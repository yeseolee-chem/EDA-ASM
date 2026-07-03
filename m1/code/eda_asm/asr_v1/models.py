"""ASR head architectures (v1).

Two heads, sharing the sign-constrained output layer:

- ``BaselineB0`` — mean-pool per-atom features for R and for P, concat,
  feed through a small MLP, then the sign-constrained 5-component head.
  This is the floor that ``ModelM1`` must beat to justify the extra
  parameters at this data scale.

- ``ModelM1`` — multi-head cross-attention between R-atom features and
  P-atom features, followed by attention pooling and the same head.
  No atom-mapping required for v1: cross-attention is set-to-set
  (queries are P atoms, keys/values are R atoms). The fusion is
  symmetric (mean of R→P and P→R).

Both consume **precomputed** backbone features (frozen NequIP), so the
head training is fast and decoupled from the GNN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== Sign-constrained output head ==========================================


@dataclass(frozen=True)
class LabelStandardizer:
    """Component-wise (mean, std) for standardizing labels during training."""

    mean: torch.Tensor   # shape (5,)
    std: torch.Tensor    # shape (5,)

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean.to(y)) / self.std.to(y)

    def unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z) + self.mean.to(z)


class SignConstrainedHead(nn.Module):
    """Final 5-component head with per-channel sign constraint.

    Component order (must match data.ASR_COMPONENTS):
      0  E_strain  ≥ 0  → +softplus
      1  Pauli     ≥ 0  → +softplus
      2  V_elst    ≤ 0  → -softplus
      3  E_orb     ≤ 0  → -softplus
      4  E_disp    ≤ 0  → -softplus

    The signed magnitudes are produced directly in kcal/mol (not in the
    standardized space) so the constraint is enforced on the true target.
    """

    SIGNS = (+1.0, +1.0, -1.0, -1.0, -1.0)

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 5),
        )
        signs = torch.tensor(self.SIGNS, dtype=torch.float32).view(1, 5)
        self.register_buffer("signs", signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(x)                         # (B, 5)
        mag = F.softplus(raw)                     # ≥ 0
        return mag * self.signs                   # apply sign per channel


# ===== Input standardization =================================================


class InputStandardizer(nn.Module):
    """Per-feature z-score standardization for frozen-backbone features.

    Stats are NON-trainable buffers, fit once per training fold from the
    train-fold concatenation of R and P features. They are persisted as
    part of the model state_dict so the same scaling is reapplied at
    inference / when the artifact is reused by V2.

    Until ``fit_from`` (or ``set_stats``) is called, this layer is the
    identity (mean=0, std=1).
    """

    def __init__(self, feature_dim: int, eps: float = 1e-6):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.feature_dim))
        self.register_buffer("std", torch.ones(self.feature_dim))
        self.register_buffer("_fitted", torch.zeros(1, dtype=torch.bool))

    @property
    def fitted(self) -> bool:
        return bool(self._fitted.item())

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.detach().to(self.mean).view(self.feature_dim)
        std = std.detach().to(self.std).view(self.feature_dim).clamp_min(self.eps)
        self.mean.copy_(mean)
        self.std.copy_(std)
        self._fitted.fill_(True)

    def fit_from(self, features: Sequence[torch.Tensor]) -> None:
        """Fit from a list of (n_atoms_i, F) tensors. Pools atoms across all."""
        stacked = torch.cat([f.detach() for f in features], dim=0)
        if stacked.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: standardizer={self.feature_dim} vs data={stacked.shape[1]}"
            )
        self.set_stats(stacked.mean(dim=0), stacked.std(dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ===== Pooling helpers =======================================================


def _mean_pool(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool (B, N, D) under a (B, N) bool mask. Empty rows are 0."""
    m = mask.unsqueeze(-1).to(feats.dtype)              # (B, N, 1)
    s = (feats * m).sum(dim=1)                          # (B, D)
    n = m.sum(dim=1).clamp_min(1.0)                     # (B, 1)
    return s / n


# ===== Baseline B0 ===========================================================


class BaselineB0(nn.Module):
    """Mean-pool MLP — the data-efficiency floor."""

    def __init__(
        self,
        feature_dim: int,
        d_hidden: int = 64,
        head_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_std = InputStandardizer(feature_dim)
        # Per-side feature transform (shared weights — R and P are processed identically)
        self.atom_proj = nn.Sequential(
            nn.Linear(feature_dim, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Reaction descriptor: concat([R̄, P̄, P̄ - R̄])
        self.head = SignConstrainedHead(
            d_in=3 * d_hidden, d_hidden=head_hidden, dropout=dropout,
        )

    def forward(
        self,
        R_feat: torch.Tensor,        # (B, N_R, F)
        R_mask: torch.Tensor,        # (B, N_R)  bool
        P_feat: torch.Tensor,        # (B, N_P, F)
        P_mask: torch.Tensor,        # (B, N_P)  bool
    ) -> torch.Tensor:
        R_feat = self.input_std(R_feat)
        P_feat = self.input_std(P_feat)
        rh = self.atom_proj(R_feat)
        ph = self.atom_proj(P_feat)
        r_bar = _mean_pool(rh, R_mask)
        p_bar = _mean_pool(ph, P_mask)
        rxn = torch.cat([r_bar, p_bar, p_bar - r_bar], dim=-1)
        return self.head(rxn)


# ===== Model M1 — cross-attention ============================================


class _AttentionPool(nn.Module):
    """Single-query attention pool with masking."""

    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D); mask: (B, N) bool
        scores = (self.proj(x) * self.q).sum(dim=-1)              # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1).unsqueeze(-1)               # (B, N, 1)
        # If a row was fully masked (shouldn't happen), zero it out
        any_valid = mask.any(dim=-1, keepdim=True).unsqueeze(-1)
        out = (x * w).sum(dim=1)
        return torch.where(any_valid.squeeze(-1), out, torch.zeros_like(out))


class ModelM1(nn.Module):
    """Cross-attention head (set-to-set, no atom mapping required).

    Pipeline:
      project (R, P) features → d_model.
      run multi-head cross-attention twice: queries=P keys=R, and queries=R keys=P.
      attention-pool each side, concat with the symmetric difference,
      and apply the sign-constrained head.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        n_heads: int = 2,
        head_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.feature_dim = feature_dim
        self.input_std = InputStandardizer(feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.xattn_p_from_r = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.xattn_r_from_p = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)
        self.pool_p = _AttentionPool(d_model)
        self.pool_r = _AttentionPool(d_model)
        self.head = SignConstrainedHead(
            d_in=3 * d_model, d_hidden=head_hidden, dropout=dropout,
        )

    def forward(
        self,
        R_feat: torch.Tensor,
        R_mask: torch.Tensor,
        P_feat: torch.Tensor,
        P_mask: torch.Tensor,
    ) -> torch.Tensor:
        R_feat = self.input_std(R_feat)
        P_feat = self.input_std(P_feat)
        rh = self.proj(R_feat)
        ph = self.proj(P_feat)

        # nn.MultiheadAttention uses key_padding_mask where True = IGNORE
        kpm_r = ~R_mask
        kpm_p = ~P_mask

        p2 = self.xattn_p_from_r(query=ph, key=rh, value=rh,
                                 key_padding_mask=kpm_r, need_weights=False)[0]
        r2 = self.xattn_r_from_p(query=rh, key=ph, value=ph,
                                 key_padding_mask=kpm_p, need_weights=False)[0]
        ph_out = self.norm_p(ph + p2)
        rh_out = self.norm_r(rh + r2)
        p_vec = self.pool_p(ph_out, P_mask)
        r_vec = self.pool_r(rh_out, R_mask)
        rxn = torch.cat([r_vec, p_vec, p_vec - r_vec], dim=-1)
        return self.head(rxn)
