"""5-fold CV × M-ensemble training of an ASR v1 head (B0 or M1).

Loads the precomputed feature cache, runs K-fold CV with an M-ensemble
inside each fold, saves all checkpoints + per-fold and per-ensemble
metrics. Prints a final per-component MAE in kcal/mol (mean ± std across
folds, averaged over the ensemble).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from eda_asm.asr_v1.data import ASR_COMPONENTS
from eda_asm.asr_v1.models import BaselineB0, ModelM1
from eda_asm.asr_v1.training import (
    CachedFeatureBundle,
    TrainConfig,
    kfold_indices,
    train_one_model,
)


def _build_factory(model_name: str, cfg: dict):
    if model_name == "b0":
        c = cfg["baseline_b0"]
        return lambda F: BaselineB0(
            feature_dim=F, d_hidden=c["d_hidden"],
            head_hidden=c["head_hidden"], dropout=c["dropout"],
        )
    if model_name == "m1":
        c = cfg["model_m1"]
        return lambda F: ModelM1(
            feature_dim=F, d_model=c["d_model"], n_heads=c["n_heads"],
            head_hidden=c["head_hidden"], dropout=c["dropout"],
        )
    raise ValueError(f"unknown model: {model_name}")


def _resolve_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml")
    ap.add_argument("--model", choices=["b0", "m1"], required=True)
    ap.add_argument("--output-dir", default=None,
                    help="defaults to outputs/asr_v1/<model>/")
    ap.add_argument("--features", default=None,
                    help="override feature_cache from config (used by the "
                         "backbone-comparison spec)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo = Path.cwd()
    out_root = Path(args.output_dir or f"outputs/asr_v1/{args.model}")
    out_root.mkdir(parents=True, exist_ok=True)

    feature_cache_path = (
        Path(args.features) if args.features else repo / cfg["feature_cache"]
    )
    # Stamp the effective feature path into the config dict so summary.json
    # records exactly which cache was used.
    cfg["feature_cache"] = str(feature_cache_path)
    print(f"[train_cv] model={args.model}  output={out_root}")
    print(f"[train_cv] features={feature_cache_path}")
    bundle = CachedFeatureBundle.load(feature_cache_path)
    n = len(bundle)
    print(f"[train_cv] loaded {n} reactions, feature_dim={bundle.feature_dim}")

    tcfg = TrainConfig(
        epochs=int(cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        early_stop_patience=int(cfg["train"]["early_stop_patience"]),
        device=_resolve_device(cfg["train"]["device"]),
    )
    print(f"[train_cv] device={tcfg.device}  epochs={tcfg.epochs}  "
          f"bs={tcfg.batch_size}  lr={tcfg.lr}  wd={tcfg.weight_decay}")

    factory = _build_factory(args.model, cfg)
    K = int(cfg["cv"]["n_folds"])
    M = int(cfg["ensemble"]["n_models"])
    base_seed = int(cfg["ensemble"]["base_seed"])
    splits = kfold_indices(n, K, seed=int(cfg["cv"]["seed"]))

    fold_records = []
    fold_mean_mae = []         # (K, 5) — ensemble mean per fold
    t_start = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fold_dir = out_root / f"fold_{fold_i:02d}"
        fold_dir.mkdir(exist_ok=True)
        print(f"\n=== fold {fold_i+1}/{K}  train={len(train_idx)}  val={len(val_idx)} ===")

        ens_preds_val = []     # (M, n_val, 5)
        ens_per_comp = []      # (M, 5)
        ens_epoch_info = []    # list of (best_epoch, final_epoch, early_stopped)
        for m_idx in range(M):
            seed = base_seed + 1000 * fold_i + m_idx
            t0 = time.time()
            model, res = train_one_model(
                bundle, factory, train_idx, val_idx, tcfg, seed=seed,
            )
            dt = time.time() - t0
            stop_tag = "early" if res.early_stopped else "max"
            print(f"  [m{m_idx}] best_ep={res.best_epoch:3d}  "
                  f"stop_ep={res.final_epoch:3d} ({stop_tag})  "
                  f"val_MAE={res.val_mae_overall:.3f}  "
                  f"per_comp={[f'{x:.2f}' for x in res.val_mae_per_component]}  "
                  f"({dt:.1f}s)")
            torch.save(model.state_dict(), fold_dir / f"model_m{m_idx}.pt")
            ens_per_comp.append(res.val_mae_per_component)
            ens_epoch_info.append(
                {"best_epoch": res.best_epoch, "final_epoch": res.final_epoch,
                 "early_stopped": res.early_stopped}
            )

            # Compute predictions on val set for this ensemble member
            model.eval()
            with torch.no_grad():
                preds_this = []
                for j in val_idx:
                    r = bundle.R_features[j].unsqueeze(0).to(tcfg.device)
                    p = bundle.P_features[j].unsqueeze(0).to(tcfg.device)
                    rm = torch.ones(r.shape[:2], dtype=torch.bool, device=tcfg.device)
                    pm = torch.ones(p.shape[:2], dtype=torch.bool, device=tcfg.device)
                    preds_this.append(model(r, rm, p, pm)[0].cpu())
            ens_preds_val.append(torch.stack(preds_this, dim=0))     # (n_val, 5)

        # Ensemble-mean prediction on this fold's val
        ens_stack = torch.stack(ens_preds_val, dim=0)                # (M, n_val, 5)
        ens_mean = ens_stack.mean(dim=0)                             # (n_val, 5)
        targets = bundle.labels[val_idx]                             # (n_val, 5)
        per_comp_mae = (ens_mean - targets).abs().mean(dim=0).numpy()
        fold_mean_mae.append(per_comp_mae)
        print(f"  >> ensemble per-comp MAE: "
              f"{[f'{x:.3f}' for x in per_comp_mae]} (overall {per_comp_mae.mean():.3f})")

        # Persist per-fold record
        rec = {
            "fold": fold_i,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_indices": val_idx,
            "ensemble_per_member_per_comp_mae": [x.tolist() for x in ens_per_comp],
            "ensemble_per_member_epoch_info": ens_epoch_info,
            "ensemble_mean_per_comp_mae_kcal": per_comp_mae.tolist(),
            "ensemble_mean_overall_mae_kcal": float(per_comp_mae.mean()),
            "max_epochs_config": tcfg.epochs,
            "early_stop_patience": tcfg.early_stop_patience,
        }
        fold_records.append(rec)
        (fold_dir / "fold_metrics.json").write_text(json.dumps(rec, indent=2))

    elapsed = time.time() - t_start
    fold_mean_mae = np.stack(fold_mean_mae, axis=0)                  # (K, 5)
    mean_per_comp = fold_mean_mae.mean(axis=0)
    std_per_comp = fold_mean_mae.std(axis=0)

    summary = {
        "model": args.model,
        "n_reactions": n,
        "feature_dim": bundle.feature_dim,
        "k_folds": K,
        "ensemble_size": M,
        "component_order": list(ASR_COMPONENTS),
        "per_component_mae_kcal_mean": mean_per_comp.tolist(),
        "per_component_mae_kcal_std":  std_per_comp.tolist(),
        "overall_mae_kcal_mean": float(mean_per_comp.mean()),
        "overall_mae_kcal_std":  float(fold_mean_mae.mean(axis=1).std()),
        "train_seconds": elapsed,
        "device": tcfg.device,
        "config": cfg,
        "folds": fold_records,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[train_cv] DONE in {elapsed:.1f}s   "
          f"→ overall MAE = {summary['overall_mae_kcal_mean']:.3f} "
          f"± {summary['overall_mae_kcal_std']:.3f} kcal/mol")
    print("[train_cv] per-component MAE (mean ± std over folds):")
    for name, m, s in zip(ASR_COMPONENTS, mean_per_comp, std_per_comp):
        print(f"  {name:>18s} : {m:7.3f} ± {s:.3f} kcal/mol")


if __name__ == "__main__":
    main()
