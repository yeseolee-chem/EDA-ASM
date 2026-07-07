"""Delta training with an externally-supplied baseline vector.

Wraps the reactot `training_delta` logic but bypasses its internal
`LinearBaseline.fit(D_train, Y_train) -> predict(D_all)` step. Instead, the
caller passes in an `(N, 5)` `baseline_all_np` where:

  baseline_all_np[train_idx] = b_oof(train)      # anti-leakage OOF baseline
  baseline_all_np[val_idx]   = b_full(val)       # baseline retrained on all
                                                 # of outer-train

This preserves the delta head architecture / optimiser / schedule (identical
across arms B and C) — only the baseline provider changes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reactot training_delta pieces
import sys
from pathlib import Path
_M3 = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/m3/code")
sys.path.insert(0, str(_M3))
from eda_asm.asr_v1.training_delta import (  # noqa: E402
    CachedFeatureBundleDelta, _IndexedDatasetDelta, _collate_delta,
    TrainConfigDelta, _component_mae,
)


@dataclass
class FoldResultCustom:
    val_mae_per_component: np.ndarray
    val_mae_overall: float
    val_mae_baseline_only: np.ndarray
    best_epoch: int
    final_epoch: int
    early_stopped: bool
    history: list[dict]


def train_one_delta_custom(
    bundle: CachedFeatureBundleDelta,
    model_factory: Callable[[int], nn.Module],
    train_idx: list[int],
    val_idx: list[int],
    cfg: TrainConfigDelta,
    baseline_all_np: np.ndarray,   # (N, 5) — pre-computed baseline
    seed: int = 0,
    grad_clip: float = 5.0,
) -> tuple[nn.Module, FoldResultCustom]:
    """Same training loop as `train_one_model_delta`, but the baseline for
    every reaction is supplied by the caller (no ridge fit inside)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    baseline_all = torch.from_numpy(baseline_all_np.astype(np.float32))

    Y_np = bundle.labels.numpy()
    val_base_mae = np.abs(baseline_all_np[val_idx] - Y_np[val_idx]).mean(axis=0)

    # SPEC §4: loss = mean over batch of  mean_c |ŷ_c − y_c| / σ_c
    # σ_c = per-channel std of train-fold labels (fixed for the whole run).
    sigma_c = Y_np[train_idx].std(axis=0)
    sigma_c = np.where(sigma_c < 1e-6, 1.0, sigma_c).astype(np.float32)
    sigma_c_t = torch.from_numpy(sigma_c).to(cfg.device)

    model = model_factory(bundle.feature_dim).to(cfg.device)
    if hasattr(model, "input_std"):
        train_features = (
            [bundle.R_features[i] for i in train_idx]
            + [bundle.P_features[i] for i in train_idx]   # spec: R+P only, TS excluded
        )
        model.input_std.fit_from(train_features)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    train_ds = _IndexedDatasetDelta(bundle, train_idx, baseline_all)
    val_ds = _IndexedDatasetDelta(bundle, val_idx, baseline_all)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=_collate_delta, drop_last=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=_collate_delta, num_workers=0)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    final_epoch = 0
    early_stopped = False
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_seen = 0.0, 0
        for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in train_dl:
            R_feat = R_feat.to(cfg.device); R_mask = R_mask.to(cfg.device)
            T_feat = T_feat.to(cfg.device); T_mask = T_mask.to(cfg.device)
            P_feat = P_feat.to(cfg.device); P_mask = P_mask.to(cfg.device)
            y = y.to(cfg.device); b = b.to(cfg.device)
            delta = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
            pred = b + delta
            # σ_c-normalised L1 (SPEC §4).
            loss = (torch.abs(pred - y) / sigma_c_t).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item() * y.shape[0]; n_seen += y.shape[0]
        train_mae = ep_loss / max(n_seen, 1)

        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in val_dl:
                R_feat = R_feat.to(cfg.device); R_mask = R_mask.to(cfg.device)
                T_feat = T_feat.to(cfg.device); T_mask = T_mask.to(cfg.device)
                P_feat = P_feat.to(cfg.device); P_mask = P_mask.to(cfg.device)
                b = b.to(cfg.device)
                delta = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
                pred = b + delta
                val_preds.append(pred.cpu()); val_targs.append(y)
        vp = torch.cat(val_preds, dim=0); vt = torch.cat(val_targs, dim=0)
        per_comp = _component_mae(vp, vt)
        val_mae = per_comp.mean().item()

        history.append({"epoch": epoch, "train_mae": train_mae,
                        "val_mae": val_mae,
                        "val_mae_per_comp": per_comp.tolist()})

        final_epoch = epoch
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stop_patience:
                early_stopped = True
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    val_preds, val_targs = [], []
    with torch.no_grad():
        for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in val_dl:
            R_feat = R_feat.to(cfg.device); R_mask = R_mask.to(cfg.device)
            T_feat = T_feat.to(cfg.device); T_mask = T_mask.to(cfg.device)
            P_feat = P_feat.to(cfg.device); P_mask = P_mask.to(cfg.device)
            b = b.to(cfg.device)
            val_preds.append(
                (b + model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)).cpu()
            )
            val_targs.append(y)
    vp = torch.cat(val_preds, dim=0); vt = torch.cat(val_targs, dim=0)
    per_comp = _component_mae(vp, vt).numpy()
    return model, FoldResultCustom(
        val_mae_per_component=per_comp,
        val_mae_overall=float(per_comp.mean()),
        val_mae_baseline_only=val_base_mae,
        best_epoch=best_epoch, final_epoch=final_epoch,
        early_stopped=early_stopped, history=history,
    )
