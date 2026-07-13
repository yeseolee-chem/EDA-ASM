"""5-fold CV × M-ensemble training for Δ-learning heads."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from eda_asm.asr_v1.data import ASR_COMPONENTS
from eda_asm.asr_v1.models_delta import BaselineB0Delta, ModelM1Delta
from eda_asm.asr_v1.training_delta import (
    CachedFeatureBundleDelta,
    TrainConfigDelta,
    kfold_indices,
    train_one_model_delta,
)


def _build_factory(model_name: str, cfg: dict):
    if model_name == "b0_delta":
        c = cfg["baseline_b0"]
        return lambda F: BaselineB0Delta(
            feature_dim=F, d_hidden=c["d_hidden"],
            head_hidden=c["head_hidden"], dropout=c["dropout"],
        )
    if model_name == "m1_delta":
        c = cfg["model_m1"]
        return lambda F: ModelM1Delta(
            feature_dim=F, d_model=c["d_model"], n_heads=c["n_heads"],
            head_hidden=c["head_hidden"], dropout=c["dropout"],
        )
    raise ValueError(f"unknown delta model: {model_name}")


def _resolve_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml")
    ap.add_argument("--model", choices=["b0_delta", "m1_delta"], required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo = Path.cwd()
    out_root = Path(args.output_dir or f"outputs/asr_v1/{args.model}")
    out_root.mkdir(parents=True, exist_ok=True)

    feature_cache_path = (
        Path(args.features) if args.features else repo / cfg["feature_cache"]
    )
    cfg["feature_cache"] = str(feature_cache_path)
    print(f"[train_cv_delta] model={args.model}  output={out_root}")
    print(f"[train_cv_delta] features={feature_cache_path}")
    bundle = CachedFeatureBundleDelta.load(feature_cache_path)
    n = len(bundle)
    print(f"[train_cv_delta] loaded {n} reactions, feature_dim={bundle.feature_dim}, "
          f"descriptors={tuple(bundle.descriptors.shape)}")

    tcfg = TrainConfigDelta(
        epochs=int(cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        early_stop_patience=int(cfg["train"]["early_stop_patience"]),
        device=_resolve_device(cfg["train"]["device"]),
        baseline_ridge_alpha=float(cfg.get("delta_baseline", {}).get("ridge_alpha", 1.0)),
    )
    print(f"[train_cv_delta] device={tcfg.device}  epochs={tcfg.epochs}  "
          f"bs={tcfg.batch_size}  lr={tcfg.lr}  wd={tcfg.weight_decay}  "
          f"ridge_alpha={tcfg.baseline_ridge_alpha}")

    factory = _build_factory(args.model, cfg)
    K = int(cfg["cv"]["n_folds"])
    M = int(cfg["ensemble"]["n_models"])
    base_seed = int(cfg["ensemble"]["base_seed"])
    splits = kfold_indices(n, K, seed=int(cfg["cv"]["seed"]))

    fold_records = []
    fold_mean_mae = []
    fold_base_mae = []
    t_start = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fold_dir = out_root / f"fold_{fold_i:02d}"
        fold_dir.mkdir(exist_ok=True)
        print(f"\n=== fold {fold_i+1}/{K}  train={len(train_idx)}  val={len(val_idx)} ===")

        ens_preds_val = []
        ens_per_comp = []
        ens_epoch_info = []
        base_mae_this_fold = None
        for m_idx in range(M):
            seed = base_seed + 1000 * fold_i + m_idx
            t0 = time.time()
            model, res = train_one_model_delta(
                bundle, factory, train_idx, val_idx, tcfg, seed=seed,
            )
            dt = time.time() - t0
            stop_tag = "early" if res.early_stopped else "max"
            print(f"  [m{m_idx}] best_ep={res.best_epoch:3d}  "
                  f"stop_ep={res.final_epoch:3d} ({stop_tag})  "
                  f"val_MAE={res.val_mae_overall:.3f}  "
                  f"(baseline-only={float(res.val_mae_baseline_only.mean()):.3f})  "
                  f"per_comp={[f'{x:.2f}' for x in res.val_mae_per_component]}  "
                  f"({dt:.1f}s)")
            torch.save(
                {"model": model.state_dict(), "baseline": res.baseline_state},
                fold_dir / f"model_m{m_idx}.pt",
            )
            ens_per_comp.append(res.val_mae_per_component)
            ens_epoch_info.append(
                {"best_epoch": res.best_epoch, "final_epoch": res.final_epoch,
                 "early_stopped": res.early_stopped}
            )
            base_mae_this_fold = res.val_mae_baseline_only

            # Per-member ensemble prediction
            from eda_asm.asr_v1.baseline_physics import LinearBaseline
            bl = LinearBaseline()
            bl.load_state_dict(res.baseline_state)
            baseline_all = torch.from_numpy(
                bl.predict(bundle.descriptors.numpy())
            ).float()
            model.eval()
            with torch.no_grad():
                preds_this = []
                for j in val_idx:
                    r = bundle.R_features[j].unsqueeze(0).to(tcfg.device)
                    t = bundle.TS_features[j].unsqueeze(0).to(tcfg.device)
                    p = bundle.P_features[j].unsqueeze(0).to(tcfg.device)
                    rm = torch.ones(r.shape[:2], dtype=torch.bool, device=tcfg.device)
                    tm = torch.ones(t.shape[:2], dtype=torch.bool, device=tcfg.device)
                    pm = torch.ones(p.shape[:2], dtype=torch.bool, device=tcfg.device)
                    delta = model(r, rm, t, tm, p, pm)[0].cpu()
                    pred = baseline_all[j] + delta
                    preds_this.append(pred)
            ens_preds_val.append(torch.stack(preds_this, dim=0))

        ens_stack = torch.stack(ens_preds_val, dim=0)
        ens_mean = ens_stack.mean(dim=0)
        targets = bundle.labels[val_idx]
        per_comp_mae = (ens_mean - targets).abs().mean(dim=0).numpy()
        fold_mean_mae.append(per_comp_mae)
        fold_base_mae.append(base_mae_this_fold)
        print(f"  >> ensemble per-comp MAE: "
              f"{[f'{x:.3f}' for x in per_comp_mae]} "
              f"(overall {per_comp_mae.mean():.3f}; "
              f"baseline-only {float(base_mae_this_fold.mean()):.3f})")

        rec = {
            "fold": fold_i,
            "n_train": len(train_idx), "n_val": len(val_idx),
            "val_indices": val_idx,
            "ensemble_per_member_per_comp_mae": [x.tolist() for x in ens_per_comp],
            "ensemble_per_member_epoch_info": ens_epoch_info,
            "ensemble_mean_per_comp_mae_kcal": per_comp_mae.tolist(),
            "ensemble_mean_overall_mae_kcal": float(per_comp_mae.mean()),
            "baseline_only_per_comp_mae_kcal": base_mae_this_fold.tolist(),
            "baseline_only_overall_mae_kcal": float(base_mae_this_fold.mean()),
            "max_epochs_config": tcfg.epochs,
            "early_stop_patience": tcfg.early_stop_patience,
        }
        fold_records.append(rec)
        (fold_dir / "fold_metrics.json").write_text(json.dumps(rec, indent=2))

    elapsed = time.time() - t_start
    fold_mean_mae = np.stack(fold_mean_mae, axis=0)
    fold_base_mae = np.stack(fold_base_mae, axis=0)
    mean_per_comp = fold_mean_mae.mean(axis=0)
    std_per_comp = fold_mean_mae.std(axis=0)

    summary = {
        "model": args.model,
        "input_mode": "delta_rtsp",
        "n_reactions": n,
        "feature_dim": bundle.feature_dim,
        "k_folds": K, "ensemble_size": M,
        "component_order": list(ASR_COMPONENTS),
        "per_component_mae_kcal_mean": mean_per_comp.tolist(),
        "per_component_mae_kcal_std":  std_per_comp.tolist(),
        "overall_mae_kcal_mean": float(mean_per_comp.mean()),
        "overall_mae_kcal_std":  float(fold_mean_mae.mean(axis=1).std()),
        "baseline_only_per_component_mae_kcal_mean": fold_base_mae.mean(axis=0).tolist(),
        "baseline_only_overall_mae_kcal_mean": float(fold_base_mae.mean()),
        "train_seconds": elapsed,
        "device": tcfg.device,
        "config": cfg,
        "folds": fold_records,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[train_cv_delta] DONE in {elapsed:.1f}s   "
          f"→ overall MAE = {summary['overall_mae_kcal_mean']:.3f} ± "
          f"{summary['overall_mae_kcal_std']:.3f} kcal/mol  "
          f"(baseline-only {summary['baseline_only_overall_mae_kcal_mean']:.3f})")
    print("[train_cv_delta] per-component MAE (mean ± std over folds):")
    for name, m, s in zip(ASR_COMPONENTS, mean_per_comp, std_per_comp):
        print(f"  {name:>18s} : {m:7.3f} ± {s:.3f} kcal/mol")


if __name__ == "__main__":
    main()
