"""3-way (R, TS, P) ASR heads — v1.

Adds the TS geometry as an explicit input alongside R and P. Motivation:
the original R/P-only models hit a ~9.4 kcal/mol overall MAE on N=250
dipolar, with Pauli/V_elst/E_orb errors at 50–65% of label std. Those
channels are dominated by the TS-frame intermolecular distances/orientations,
which the R/P-only backbone has to *infer* — supplying the DFT-converged TS
directly removes that bottleneck.

Two heads, sharing the sign-constrained output:

- ``BaselineB0RTSP`` — per-side mean-pool, then concat
    [r̄, ts̄, p̄, ts̄−r̄, ts̄−p̄, p̄−r̄] → MLP head. The 3 difference vectors
    expose strain-like and reaction-direction signals explicitly.

- ``ModelM1RTSP`` — TS-centric set-to-set cross-attention. TS attends to R
    and to P; R and P each attend back to TS. Symmetric in R↔P. No atom
    mapping required (the cross-attention is set-to-set).

Both consume precomputed backbone features, identical to the 2-way pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    InputStandardizer,
    SignConstrainedHead,
    _AttentionPool,
    _mean_pool,
)


# ===== Baseline B0 (RTSP) =====================================================


class BaselineB0RTSP(nn.Module):
    """Mean-pool MLP over (R, TS, P)."""

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
        # concat([r̄, ts̄, p̄, ts̄−r̄, ts̄−p̄, p̄−r̄]) = 6 × d_hidden
        self.head = SignConstrainedHead(
            d_in=6 * d_hidden, d_hidden=head_hidden, dropout=dropout,
        )

    def forward(
        self,
        R_feat: torch.Tensor,   R_mask: torch.Tensor,
        TS_feat: torch.Tensor,  TS_mask: torch.Tensor,
        P_feat: torch.Tensor,   P_mask: torch.Tensor,
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


# ===== Model M1 (RTSP, TS-centric cross-attention) ============================


class ModelM1RTSP(nn.Module):
    """TS-centric multi-head cross-attention over (R, TS, P).

    Architecture:
        project R, TS, P → d_model
        TS ←attn─ R     (queries=TS, keys/values=R)
        TS ←attn─ P     (queries=TS, keys/values=P)
        R  ←attn─ TS    (queries=R,  keys/values=TS)
        P  ←attn─ TS    (queries=P,  keys/values=TS)
        residual + LayerNorm on each updated side; the two TS updates are
            summed before residual+norm so TS stays a single tensor.
        attention-pool each → r_vec, ts_vec, p_vec
        head_in = concat([r_vec, ts_vec, p_vec,
                          ts_vec-r_vec, ts_vec-p_vec, p_vec-r_vec])
        → SignConstrainedHead
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
        # TS ← R, TS ← P, R ← TS, P ← TS
        mha_kwargs = dict(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            batch_first=True,
        )
        self.xattn_ts_from_r = nn.MultiheadAttention(**mha_kwargs)
        self.xattn_ts_from_p = nn.MultiheadAttention(**mha_kwargs)
        self.xattn_r_from_ts = nn.MultiheadAttention(**mha_kwargs)
        self.xattn_p_from_ts = nn.MultiheadAttention(**mha_kwargs)
        self.norm_ts = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        self.pool_ts = _AttentionPool(d_model)
        self.pool_r = _AttentionPool(d_model)
        self.pool_p = _AttentionPool(d_model)
        self.head = SignConstrainedHead(
            d_in=6 * d_model, d_hidden=head_hidden, dropout=dropout,
        )

    def forward(
        self,
        R_feat: torch.Tensor,  R_mask: torch.Tensor,
        TS_feat: torch.Tensor, TS_mask: torch.Tensor,
        P_feat: torch.Tensor,  P_mask: torch.Tensor,
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

        # Residual + LayerNorm. TS receives two updates → average them so the
        # residual scale doesn't double versus R/P.
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
