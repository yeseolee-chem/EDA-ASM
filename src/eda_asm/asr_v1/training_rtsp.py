"""Training loop + dataset bundle for the 3-way (R, TS, P) ASR heads.

Mirrors ``training.py`` but holds a per-reaction TS feature tensor in
addition to R and P. Shares ``kfold_indices`` from ``training`` so fold
assignments are identical between the 2-way and 3-way pipelines at the
same (N, seed).
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

from .training import kfold_indices  # re-exported for callers


# ===== Cached-feature dataset (RTSP) =========================================


@dataclass
class CachedFeatureBundleRTSP:
    reaction_ids: list[str]
    R_features: list[torch.Tensor]
    TS_features: list[torch.Tensor]
    P_features: list[torch.Tensor]
    labels: torch.Tensor
    feature_dim: int

    @classmethod
    def load(cls, path: str | Path) -> "CachedFeatureBundleRTSP":
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(
            reaction_ids=obj["reaction_ids"],
            R_features=obj["R_features"],
            TS_features=obj["TS_features"],
            P_features=obj["P_features"],
            labels=obj["labels"].float(),
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
                "feature_dim": self.feature_dim,
            },
            str(path),
        )

    def __len__(self) -> int:
        return len(self.reaction_ids)


class _IndexedDatasetRTSP(Dataset):
    def __init__(self, bundle: CachedFeatureBundleRTSP, indices: list[int]):
        self.bundle = bundle
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        return (
            self.bundle.R_features[j],
            self.bundle.TS_features[j],
            self.bundle.P_features[j],
            self.bundle.labels[j],
        )


def _collate_rtsp(batch):
    Rs, Ts, Ps, ys = zip(*batch)
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
    return R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y


# ===== Train one model =======================================================


@dataclass
class TrainConfigRTSP:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    early_stop_patience: int = 30
    device: str = "cpu"


@dataclass
class FoldResultRTSP:
    train_indices: list[int]
    val_indices: list[int]
    val_mae_per_component: np.ndarray
    val_mae_overall: float
    best_epoch: int
    final_epoch: int
    early_stopped: bool
    history: list[dict]


def _component_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=0)


def train_one_model_rtsp(
    bundle: CachedFeatureBundleRTSP,
    model_factory: Callable[[int], nn.Module],
    train_idx: list[int],
    val_idx: list[int],
    cfg: TrainConfigRTSP,
    seed: int = 0,
) -> tuple[nn.Module, FoldResultRTSP]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_factory(bundle.feature_dim).to(cfg.device)

    # Fit InputStandardizer on train-fold R ∪ TS ∪ P features.
    if hasattr(model, "input_std"):
        train_features = (
            [bundle.R_features[i] for i in train_idx]
            + [bundle.TS_features[i] for i in train_idx]
            + [bundle.P_features[i] for i in train_idx]
        )
        model.input_std.fit_from(train_features)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    train_ds = _IndexedDatasetRTSP(bundle, train_idx)
    val_ds = _IndexedDatasetRTSP(bundle, val_idx)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=_collate_rtsp, drop_last=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=_collate_rtsp, num_workers=0)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    final_epoch = 0
    early_stopped = False
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y in train_dl:
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            y = y.to(cfg.device)
            pred = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
            loss = F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.shape[0]
            n_seen += y.shape[0]
        train_mae = ep_loss / max(n_seen, 1)

        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y in val_dl:
                R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
                T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
                P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
                pred = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
                val_preds.append(pred.cpu())
                val_targs.append(y)
        vp = torch.cat(val_preds, dim=0)
        vt = torch.cat(val_targs, dim=0)
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
        for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y in val_dl:
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            val_preds.append(model(R_feat, R_mask, T_feat, T_mask,
                                   P_feat, P_mask).cpu())
            val_targs.append(y)
    vp = torch.cat(val_preds, dim=0)
    vt = torch.cat(val_targs, dim=0)
    per_comp = _component_mae(vp, vt).numpy()
    result = FoldResultRTSP(
        train_indices=train_idx, val_indices=val_idx,
        val_mae_per_component=per_comp,
        val_mae_overall=float(per_comp.mean()),
        best_epoch=best_epoch, final_epoch=final_epoch,
        early_stopped=early_stopped, history=history,
    )
    return model, result
