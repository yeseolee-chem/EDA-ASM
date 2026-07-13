"""Δ-learning heads — same R/TS/P inputs as the RTSP heads, but the final
linear layer is **unconstrained** (no softplus, no sign mask). The training
loop passes ``y = baseline + head(features)`` to the loss; the head learns
the *residual* between the deterministic ridge baseline and the DFT labels.

Two heads, mirroring B0_RTSP / M1_RTSP architecturally:
  - ``BaselineB0Delta``  — mean-pool concat over (R, TS, P).
  - ``ModelM1Delta``     — TS-centric cross-attention over (R, TS, P).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import InputStandardizer, _AttentionPool, _mean_pool


class _DeltaHead(nn.Module):
    """Plain MLP → 5 unbounded outputs (the residual)."""

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
        # Initialize the last layer near zero so the model starts at
        # y ≈ baseline and learns small corrections.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BaselineB0Delta(nn.Module):
    """Mean-pool MLP residual model. Output is the *delta* only;
    the final ``y = baseline + delta`` is added by the training loop."""

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
        self.atom_proj = nn.Sequential(
            nn.Linear(feature_dim, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = _DeltaHead(
            d_in=6 * d_hidden, d_hidden=head_hidden, dropout=dropout,
        )

    def forward(
        self,
        R_feat, R_mask, TS_feat, TS_mask, P_feat, P_mask,
    ) -> torch.Tensor:
        R_feat = self.input_std(R_feat)
        TS_feat = self.input_std(TS_feat)
        P_feat = self.input_std(P_feat)
        rh = self.atom_proj(R_feat)
        th = self.atom_proj(TS_feat)
        ph = self.atom_proj(P_feat)
        r_bar = _mean_pool(rh, R_mask)
        ts_bar = _mean_pool(th, TS_mask)
        p_bar = _mean_pool(ph, P_mask)
        rxn = torch.cat([
            r_bar, ts_bar, p_bar,
            ts_bar - r_bar, ts_bar - p_bar, p_bar - r_bar,
        ], dim=-1)
        return self.head(rxn)


class ModelM1Delta(nn.Module):
    """TS-centric cross-attention residual model. Same wiring as M1_RTSP,
    with the sign-constrained head replaced by the unbounded delta head."""

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
        mha = dict(embed_dim=d_model, num_heads=n_heads, dropout=dropout,
                   batch_first=True)
        self.xattn_ts_from_r = nn.MultiheadAttention(**mha)
        self.xattn_ts_from_p = nn.MultiheadAttention(**mha)
        self.xattn_r_from_ts = nn.MultiheadAttention(**mha)
        self.xattn_p_from_ts = nn.MultiheadAttention(**mha)
        self.norm_ts = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        self.pool_ts = _AttentionPool(d_model)
        self.pool_r = _AttentionPool(d_model)
        self.pool_p = _AttentionPool(d_model)
        self.head = _DeltaHead(
            d_in=6 * d_model, d_hidden=head_hidden, dropout=dropout,
        )

    def forward(
        self,
        R_feat, R_mask, TS_feat, TS_mask, P_feat, P_mask,
    ) -> torch.Tensor:
        R_feat = self.input_std(R_feat)
        TS_feat = self.input_std(TS_feat)
        P_feat = self.input_std(P_feat)
        rh = self.proj(R_feat)
        th = self.proj(TS_feat)
        ph = self.proj(P_feat)

        kpm_r = ~R_mask
        kpm_p = ~P_mask
        kpm_ts = ~TS_mask

        ts_from_r = self.xattn_ts_from_r(
            query=th, key=rh, value=rh, key_padding_mask=kpm_r, need_weights=False,
        )[0]
        ts_from_p = self.xattn_ts_from_p(
            query=th, key=ph, value=ph, key_padding_mask=kpm_p, need_weights=False,
        )[0]
        r_from_ts = self.xattn_r_from_ts(
            query=rh, key=th, value=th, key_padding_mask=kpm_ts, need_weights=False,
        )[0]
        p_from_ts = self.xattn_p_from_ts(
            query=ph, key=th, value=th, key_padding_mask=kpm_ts, need_weights=False,
        )[0]

        th_out = self.norm_ts(th + 0.5 * (ts_from_r + ts_from_p))
        rh_out = self.norm_r(rh + r_from_ts)
        ph_out = self.norm_p(ph + p_from_ts)

        ts_vec = self.pool_ts(th_out, TS_mask)
        r_vec = self.pool_r(rh_out, R_mask)
        p_vec = self.pool_p(ph_out, P_mask)

        rxn = torch.cat([
            r_vec, ts_vec, p_vec,
            ts_vec - r_vec, ts_vec - p_vec, p_vec - r_vec,
        ], dim=-1)
        return self.head(rxn)
