"""Training loop + dataset bundle for Δ-learning heads.

Differences vs ``training_rtsp``:
  - Bundle stores a ``descriptors`` tensor (N, 6) per reaction, precomputed
    by the cache script.
  - Per fold, a ``LinearBaseline`` is fit on TRAIN-FOLD descriptors+labels and
    applied to the entire N to produce ``baseline (N, 5)``. This is stored
    in the model's state_dict so inference can re-use the same baseline.
  - The training target for the head is ``label - baseline`` (the residual).
    The head output is unconstrained; the final prediction is
    ``y = baseline + head(features)``.

``kfold_indices`` is reused from ``training`` so folds match the 2-way /
RTSP runs at the same (N, seed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .baseline_physics import LinearBaseline, N_DESCRIPTORS
from .training import kfold_indices  # re-exported


@dataclass
class CachedFeatureBundleDelta:
    reaction_ids: list[str]
    R_features: list[torch.Tensor]
    TS_features: list[torch.Tensor]
    P_features: list[torch.Tensor]
    labels: torch.Tensor                 # (N, 5)
    descriptors: torch.Tensor            # (N, n_desc)
    feature_dim: int

    @classmethod
    def load(cls, path: str | Path) -> "CachedFeatureBundleDelta":
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(
            reaction_ids=obj["reaction_ids"],
            R_features=obj["R_features"],
            TS_features=obj["TS_features"],
            P_features=obj["P_features"],
            labels=obj["labels"].float(),
            descriptors=obj["descriptors"].float(),
            feature_dim=int(obj["feature_dim"]),
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "reaction_ids": self.reaction_ids,
                "R_features": self.R_features,
                "TS_features": self.TS_features,
                "P_features": self.P_features,
                "labels": self.labels,
                "descriptors": self.descriptors,
                "feature_dim": self.feature_dim,
            },
            str(path),
        )

    def __len__(self) -> int:
        return len(self.reaction_ids)


class _IndexedDatasetDelta(Dataset):
    """Returns (R_feat, TS_feat, P_feat, label, baseline) per item."""

    def __init__(self, bundle: CachedFeatureBundleDelta, indices: list[int],
                 baseline_all: torch.Tensor):
        self.bundle = bundle
        self.indices = list(indices)
        self.baseline_all = baseline_all          # (N, 5) on CPU

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        return (
            self.bundle.R_features[j],
            self.bundle.TS_features[j],
            self.bundle.P_features[j],
            self.bundle.labels[j],
            self.baseline_all[j],
        )


def _collate_delta(batch):
    Rs, Ts, Ps, ys, bs = zip(*batch)
    n_R = max(x.shape[0] for x in Rs)
    n_T = max(x.shape[0] for x in Ts)
    n_P = max(x.shape[0] for x in Ps)
    F_dim = Rs[0].shape[1]
    B = len(batch)

    def pack(seq, n):
        x = torch.zeros(B, n, F_dim, dtype=torch.float32)
        m = torch.zeros(B, n, dtype=torch.bool)
        for i, s in enumerate(seq):
            x[i, : s.shape[0]] = s
            m[i, : s.shape[0]] = True
        return x, m

    R_feat, R_mask = pack(Rs, n_R)
    T_feat, T_mask = pack(Ts, n_T)
    P_feat, P_mask = pack(Ps, n_P)
    y = torch.stack(list(ys), dim=0).float()
    b = torch.stack(list(bs), dim=0).float()
    return R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b


@dataclass
class TrainConfigDelta:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    early_stop_patience: int = 30
    device: str = "cpu"
    baseline_ridge_alpha: float = 1.0
    grad_clip_norm: float = 5.0     # Spec: grad-clip 5.0


@dataclass
class FoldResultDelta:
    train_indices: list[int]
    val_indices: list[int]
    val_mae_per_component: np.ndarray       # final y = baseline + delta
    val_mae_overall: float
    val_mae_baseline_only: np.ndarray       # diagnostic: baseline alone
    best_epoch: int
    final_epoch: int
    early_stopped: bool
    history: list[dict]
    baseline_state: dict                    # LinearBaseline.state_dict()


def _component_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=0)


def train_one_model_delta(
    bundle: CachedFeatureBundleDelta,
    model_factory: Callable[[int], nn.Module],
    train_idx: list[int],
    val_idx: list[int],
    cfg: TrainConfigDelta,
    seed: int = 0,
) -> tuple[nn.Module, FoldResultDelta]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1) Fit LinearBaseline on train-fold descriptors + labels.
    D_np = bundle.descriptors.numpy()
    Y_np = bundle.labels.numpy()
    D_train = D_np[train_idx]
    Y_train = Y_np[train_idx]
    bl = LinearBaseline(alpha=cfg.baseline_ridge_alpha).fit(D_train, Y_train)
    baseline_all_np = bl.predict(D_np)                          # (N, 5)
    baseline_all = torch.from_numpy(baseline_all_np).float()
    # Diagnostic: baseline-alone validation MAE.
    val_base_mae = np.abs(baseline_all_np[val_idx] - Y_np[val_idx]).mean(axis=0)

    # 2) Build the ML model.
    model = model_factory(bundle.feature_dim).to(cfg.device)

    # Spec: μ_k, σ_k fit from train-fold R and P features only (TS excluded).
    if hasattr(model, "input_std"):
        train_features = (
            [bundle.R_features[i] for i in train_idx]
            + [bundle.P_features[i] for i in train_idx]
        )
        model.input_std.fit_from(train_features)

    # Spec: per-channel σ_c from train-fold labels; loss = mean |ŷ_c − y_c| / σ_c
    sigma_c = bundle.labels[train_idx].std(dim=0).clamp_min(1e-6).to(cfg.device)  # (5,)

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
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            y = y.to(cfg.device); b = b.to(cfg.device)
            delta = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
            pred = b + delta
            # Spec loss: per-channel σ_c-normalized L1 (mean over channels + batch)
            loss = (pred - y).abs().div(sigma_c).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()
            ep_loss += loss.item() * y.shape[0]; n_seen += y.shape[0]
        train_mae = ep_loss / max(n_seen, 1)

        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in val_dl:
                R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
                T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
                P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
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
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            b = b.to(cfg.device)
            val_preds.append(
                (b + model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)).cpu()
            )
            val_targs.append(y)
    vp = torch.cat(val_preds, dim=0); vt = torch.cat(val_targs, dim=0)
    per_comp = _component_mae(vp, vt).numpy()
    result = FoldResultDelta(
        train_indices=train_idx, val_indices=val_idx,
        val_mae_per_component=per_comp,
        val_mae_overall=float(per_comp.mean()),
        val_mae_baseline_only=val_base_mae,
        best_epoch=best_epoch, final_epoch=final_epoch,
        early_stopped=early_stopped, history=history,
        baseline_state=bl.state_dict(),
    )
    return model, result
