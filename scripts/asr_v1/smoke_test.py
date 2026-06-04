"""ASR v1 smoke test — verifies the full pipeline end-to-end on 5 reactions.

Steps:
  1. Load NequIP backbone, extract features for 5 reactions (R and P each).
  2. Verify per-atom features have non-degenerate variance (Plan §4.3).
  3. Train Baseline B0 for ~20 epochs with M=2 ensemble on a 4/1 split.
  4. Train Model M1 for ~20 epochs with M=2 ensemble on the same split.
  5. Print per-component MAE for both, and a sanity check on output signs.

This must pass before launching the full CV / learning curve / AL job.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from eda_asm.asr_v1.backbone import NequIPFeatureExtractor
from eda_asm.asr_v1.data import ASR_COMPONENTS, iter_reaction_pairs, load_label_table
from eda_asm.asr_v1.models import BaselineB0, ModelM1
from eda_asm.asr_v1.training import (
    CachedFeatureBundle,
    TrainConfig,
    train_one_model,
)


SIGN_RULE = {0: +1, 1: +1, 2: -1, 3: -1, 4: -1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml")
    ap.add_argument("--n-reactions", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo = Path.cwd()
    print(f"[smoke] cwd = {repo}")

    # ===== 1. Backbone + feature extraction =====
    df = load_label_table(repo / cfg["labels_parquet"], family=cfg["family"])
    df = df.head(args.n_reactions).copy()
    print(f"[smoke] selecting {len(df)} reactions: {df['reaction_id'].tolist()}")

    backbone = NequIPFeatureExtractor(
        config_path=repo / cfg["backbone"]["config"],
        checkpoint_path=repo / cfg["backbone"]["checkpoint"],
    )
    print(f"[smoke] backbone device={backbone.device} feature_dim={backbone.feature_dim}")

    t0 = time.time()
    rids, Rfs, Pfs, ys = [], [], [], []
    for s in iter_reaction_pairs(df, dipolar_root=repo / cfg["dipolar_root"]):
        R_feat = backbone.extract(s.R_atoms)
        P_feat = backbone.extract(s.P_atoms)
        print(f"  {s.reaction_id}: R={tuple(R_feat.shape)}, P={tuple(P_feat.shape)}, "
              f"label={s.label.round(2).tolist()}")
        rids.append(s.reaction_id)
        Rfs.append(R_feat)
        Pfs.append(P_feat)
        ys.append(torch.from_numpy(s.label))
    dt = time.time() - t0
    print(f"[smoke] feature extraction: {dt:.1f}s for {len(rids)} reactions")

    # ===== 2. Degeneracy check (Plan §4.3) =====
    flat = torch.cat([f.flatten() for f in Rfs + Pfs])
    print(f"[smoke] feature stats: mean={flat.mean():.4f}, std={flat.std():.4f}, "
          f"min={flat.min():.4f}, max={flat.max():.4f}, n_nan={torch.isnan(flat).sum().item()}")
    R_concat = torch.cat(Rfs, dim=0)
    per_dim_var = R_concat.var(dim=0)
    n_zero = (per_dim_var < 1e-8).sum().item()
    print(f"[smoke] R per-feature variance: min={per_dim_var.min():.4g} "
          f"max={per_dim_var.max():.4g} n_near_zero={n_zero}/{backbone.feature_dim}")
    assert torch.isnan(flat).sum() == 0, "NaN in backbone features!"
    assert n_zero < backbone.feature_dim, "ALL feature dims are degenerate!"

    bundle = CachedFeatureBundle(
        reaction_ids=rids, R_features=Rfs, P_features=Pfs,
        labels=torch.stack(ys, dim=0).float(),
        feature_dim=backbone.feature_dim,
    )
    print(f"[smoke] bundle ready: {len(bundle)} reactions, "
          f"labels mean={bundle.labels.mean(dim=0).round(decimals=2).tolist()}")

    # ===== 3. + 4. Train B0 and M1 on a tiny split =====
    n = len(bundle)
    train_idx = list(range(n - 1))
    val_idx = [n - 1]
    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=min(8, n - 1),
        lr=1.0e-3,
        weight_decay=1.0e-3,
        early_stop_patience=args.epochs,         # no early stop in smoke
        device="cpu",
    )

    def factory_b0(F: int) -> torch.nn.Module:
        return BaselineB0(
            feature_dim=F,
            d_hidden=cfg["baseline_b0"]["d_hidden"],
            head_hidden=cfg["baseline_b0"]["head_hidden"],
            dropout=cfg["baseline_b0"]["dropout"],
        )

    def factory_m1(F: int) -> torch.nn.Module:
        return ModelM1(
            feature_dim=F,
            d_model=cfg["model_m1"]["d_model"],
            n_heads=cfg["model_m1"]["n_heads"],
            head_hidden=cfg["model_m1"]["head_hidden"],
            dropout=cfg["model_m1"]["dropout"],
        )

    for name, factory in [("B0", factory_b0), ("M1", factory_m1)]:
        print(f"\n[smoke] training {name} for {args.epochs} epochs on "
              f"{len(train_idx)}/{len(val_idx)} split…")
        t0 = time.time()
        model, res = train_one_model(bundle, factory, train_idx, val_idx, tcfg, seed=0)
        dt = time.time() - t0
        print(f"[smoke] {name} done in {dt:.1f}s")
        print(f"  best epoch        = {res.best_epoch}/{args.epochs}")
        print(f"  val MAE overall   = {res.val_mae_overall:.3f} kcal/mol")
        print(f"  val MAE per comp  = {dict(zip(ASR_COMPONENTS, res.val_mae_per_component.round(3).tolist()))}")
        # Sign check on the held-out prediction
        model.eval()
        with torch.no_grad():
            r = bundle.R_features[val_idx[0]].unsqueeze(0)
            p = bundle.P_features[val_idx[0]].unsqueeze(0)
            rm = torch.ones(r.shape[:2], dtype=torch.bool)
            pm = torch.ones(p.shape[:2], dtype=torch.bool)
            pred = model(r, rm, p, pm)[0].tolist()
        print(f"  pred (kcal/mol)   = {[round(x, 2) for x in pred]}")
        print(f"  true (kcal/mol)   = {[round(x, 2) for x in bundle.labels[val_idx[0]].tolist()]}")
        for k, p_k in enumerate(pred):
            assert SIGN_RULE[k] * p_k >= 0, f"sign constraint violated for component {k}: {p_k}"
        print(f"  ✓ sign constraints satisfied")

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
