"""Training loop, ensemble, and K-fold CV for ASR v1 heads.

Operates on **precomputed** per-atom backbone features (cached as a
``.pt`` file by ``scripts/asr_v1/cache_features.py``) so the GNN is
never touched during head training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ===== Cached-feature dataset ===============================================


@dataclass
class CachedFeatureBundle:
    """In-memory holder for all reactions' precomputed features."""

    reaction_ids: list[str]                  # length N
    R_features: list[torch.Tensor]            # each (n_R_i, F)
    P_features: list[torch.Tensor]            # each (n_P_i, F)
    labels: torch.Tensor                      # (N, 5) float32, kcal/mol
    feature_dim: int

    @classmethod
    def load(cls, path: str | Path) -> "CachedFeatureBundle":
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(
            reaction_ids=obj["reaction_ids"],
            R_features=obj["R_features"],
            P_features=obj["P_features"],
            labels=obj["labels"].float(),
            feature_dim=int(obj["feature_dim"]),
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "reaction_ids": self.reaction_ids,
                "R_features": self.R_features,
                "P_features": self.P_features,
                "labels": self.labels,
                "feature_dim": self.feature_dim,
            },
            str(path),
        )

    def __len__(self) -> int:
        return len(self.reaction_ids)


class _IndexedDataset(Dataset):
    def __init__(self, bundle: CachedFeatureBundle, indices: list[int]):
        self.bundle = bundle
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        return (
            self.bundle.R_features[j],
            self.bundle.P_features[j],
            self.bundle.labels[j],
        )


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    Rs, Ps, ys = zip(*batch)
    n_R = max(x.shape[0] for x in Rs)
    n_P = max(x.shape[0] for x in Ps)
    F_dim = Rs[0].shape[1]
    B = len(batch)

    R_feat = torch.zeros(B, n_R, F_dim, dtype=torch.float32)
    P_feat = torch.zeros(B, n_P, F_dim, dtype=torch.float32)
    R_mask = torch.zeros(B, n_R, dtype=torch.bool)
    P_mask = torch.zeros(B, n_P, dtype=torch.bool)
    for i, (r, p, _) in enumerate(batch):
        R_feat[i, : r.shape[0]] = r
        P_feat[i, : p.shape[0]] = p
        R_mask[i, : r.shape[0]] = True
        P_mask[i, : p.shape[0]] = True
    y = torch.stack(list(ys), dim=0).float()
    return R_feat, R_mask, P_feat, P_mask, y


# ===== Train one model =======================================================


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    early_stop_patience: int = 30
    device: str = "cpu"


@dataclass
class FoldResult:
    train_indices: list[int]
    val_indices: list[int]
    val_mae_per_component: np.ndarray      # (5,)
    val_mae_overall: float
    best_epoch: int          # epoch at which validation MAE was minimal
    final_epoch: int         # last epoch actually trained (stop point)
    early_stopped: bool      # True iff training halted before max_epochs
    history: list[dict]


def _component_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=0)


def train_one_model(
    bundle: CachedFeatureBundle,
    model_factory: Callable[[int], nn.Module],
    train_idx: list[int],
    val_idx: list[int],
    cfg: TrainConfig,
    seed: int = 0,
) -> tuple[nn.Module, FoldResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_factory(bundle.feature_dim).to(cfg.device)

    # Fit input standardizer on TRAIN FOLD ONLY (R and P features pooled).
    # This baked-in standardization is persisted in the state_dict, so V2
    # can reuse the same statistics by loading the artifact.
    if hasattr(model, "input_std"):
        train_features = [bundle.R_features[i] for i in train_idx] + \
                         [bundle.P_features[i] for i in train_idx]
        model.input_std.fit_from(train_features)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = _IndexedDataset(bundle, train_idx)
    val_ds = _IndexedDataset(bundle, val_idx)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=_collate, drop_last=False, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    final_epoch = 0
    early_stopped = False
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        for R_feat, R_mask, P_feat, P_mask, y in train_dl:
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            y = y.to(cfg.device)
            pred = model(R_feat, R_mask, P_feat, P_mask)
            loss = F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.shape[0]
            n_seen += y.shape[0]
        train_mae = ep_loss / max(n_seen, 1)

        # Validation
        model.eval()
        val_preds: list[torch.Tensor] = []
        val_targs: list[torch.Tensor] = []
        with torch.no_grad():
            for R_feat, R_mask, P_feat, P_mask, y in val_dl:
                R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
                P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
                pred = model(R_feat, R_mask, P_feat, P_mask)
                val_preds.append(pred.cpu())
                val_targs.append(y)
        vp = torch.cat(val_preds, dim=0)
        vt = torch.cat(val_targs, dim=0)
        per_comp_mae = _component_mae(vp, vt)
        val_mae = per_comp_mae.mean().item()

        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "val_mae_per_comp": per_comp_mae.tolist(),
            }
        )

        final_epoch = epoch
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stop_patience:
                early_stopped = True
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    # Recompute per-comp MAE at the best checkpoint
    model.eval()
    val_preds = []
    val_targs = []
    with torch.no_grad():
        for R_feat, R_mask, P_feat, P_mask, y in val_dl:
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            val_preds.append(model(R_feat, R_mask, P_feat, P_mask).cpu())
            val_targs.append(y)
    vp = torch.cat(val_preds, dim=0)
    vt = torch.cat(val_targs, dim=0)
    per_comp = _component_mae(vp, vt).numpy()
    result = FoldResult(
        train_indices=train_idx,
        val_indices=val_idx,
        val_mae_per_component=per_comp,
        val_mae_overall=float(per_comp.mean()),
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        early_stopped=early_stopped,
        history=history,
    )
    return model, result


def train_ensemble(
    bundle: CachedFeatureBundle,
    model_factory: Callable[[int], nn.Module],
    train_idx: list[int],
    val_idx: list[int],
    cfg: TrainConfig,
    n_models: int = 5,
    base_seed: int = 0,
) -> tuple[list[nn.Module], list[FoldResult]]:
    models: list[nn.Module] = []
    results: list[FoldResult] = []
    for k in range(n_models):
        m, r = train_one_model(
            bundle, model_factory, train_idx, val_idx, cfg, seed=base_seed + k,
        )
        models.append(m)
        results.append(r)
    return models, results


def kfold_indices(n: int, k: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """K-fold split returning Python ints (JSON-serializable downstream)."""
    rng = np.random.default_rng(seed)
    perm = [int(x) for x in rng.permutation(n)]
    folds = np.array_split(perm, k)
    splits = []
    for i in range(k):
        val = [int(x) for x in folds[i]]
        train = [int(x) for j in range(k) if j != i for x in folds[j]]
        splits.append((train, val))
    return splits
