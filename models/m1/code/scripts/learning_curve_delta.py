"""Learning curve for Δ-learning heads."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from eda_asm.asr_v1.baseline_physics import LinearBaseline
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
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--features", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo = Path.cwd()
    out_root = Path(args.output_dir or f"outputs/asr_v1/learning_curve_{args.model}")
    out_root.mkdir(parents=True, exist_ok=True)

    feature_cache_path = (
        Path(args.features) if args.features else repo / cfg["feature_cache"]
    )
    cfg["feature_cache"] = str(feature_cache_path)
    print(f"[lc_delta] features={feature_cache_path}")
    bundle = CachedFeatureBundleDelta.load(feature_cache_path)
    n = len(bundle)
    K = int(cfg["cv"]["n_folds"])
    M = int(cfg["ensemble"]["n_models"])
    base_seed = int(cfg["ensemble"]["base_seed"])
    splits = kfold_indices(n, K, seed=int(cfg["cv"]["seed"]))
    train_fold_size = max(len(s[0]) for s in splits)

    raw_sizes = args.sizes if args.sizes else cfg["learning_curve_train_sizes"]
    sizes = sorted({min(s, train_fold_size) for s in raw_sizes if s > 0})
    print(f"[lc_delta] model={args.model}  N={n}  train-fold={train_fold_size}")
    print(f"[lc_delta] sizes: {raw_sizes} → {sizes}")

    tcfg = TrainConfigDelta(
        epochs=int(cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        early_stop_patience=int(cfg["train"]["early_stop_patience"]),
        device=_resolve_device(cfg["train"]["device"]),
        baseline_ridge_alpha=float(cfg.get("delta_baseline", {}).get("ridge_alpha", 1.0)),
    )
    factory = _build_factory(args.model, cfg)

    rng = np.random.default_rng(int(cfg["cv"]["seed"]) + 7777)

    curve = []
    t_start = time.time()
    for N_train in sizes:
        per_fold = []
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            sub = rng.permutation(len(train_idx))[:N_train].tolist()
            sub_train_idx = [train_idx[k] for k in sub]
            ens_preds = []
            for m_idx in range(M):
                seed = base_seed + 1000 * fold_i + m_idx + 7 * N_train
                model, res = train_one_model_delta(
                    bundle, factory, sub_train_idx, val_idx, tcfg, seed=seed,
                )
                bl = LinearBaseline()
                bl.load_state_dict(res.baseline_state)
                baseline_all = torch.from_numpy(
                    bl.predict(bundle.descriptors.numpy())
                ).float()
                model.eval()
                with torch.no_grad():
                    preds = []
                    for j in val_idx:
                        r = bundle.R_features[j].unsqueeze(0).to(tcfg.device)
                        t = bundle.TS_features[j].unsqueeze(0).to(tcfg.device)
                        p = bundle.P_features[j].unsqueeze(0).to(tcfg.device)
                        rm = torch.ones(r.shape[:2], dtype=torch.bool, device=tcfg.device)
                        tm = torch.ones(t.shape[:2], dtype=torch.bool, device=tcfg.device)
                        pm = torch.ones(p.shape[:2], dtype=torch.bool, device=tcfg.device)
                        delta = model(r, rm, t, tm, p, pm)[0].cpu()
                        preds.append(baseline_all[j] + delta)
                ens_preds.append(torch.stack(preds, dim=0))
            ens_mean = torch.stack(ens_preds, dim=0).mean(dim=0)
            target = bundle.labels[val_idx]
            per_comp = (ens_mean - target).abs().mean(dim=0).numpy()
            per_fold.append(per_comp)
            print(f"  N={N_train:3d}  fold={fold_i}  per-comp MAE = "
                  f"{[f'{x:.2f}' for x in per_comp]}")

        per_fold = np.stack(per_fold, axis=0)
        record = {
            "N_train": N_train,
            "per_component_mae_kcal_mean": per_fold.mean(axis=0).tolist(),
            "per_component_mae_kcal_std":  per_fold.std(axis=0).tolist(),
            "overall_mae_kcal_mean": float(per_fold.mean(axis=0).mean()),
            "overall_mae_kcal_std":  float(per_fold.mean(axis=1).std()),
        }
        curve.append(record)
        print(f"  >> N={N_train:3d}: overall MAE = "
              f"{record['overall_mae_kcal_mean']:.3f} ± "
              f"{record['overall_mae_kcal_std']:.3f}")

    elapsed = time.time() - t_start
    summary = {
        "model": args.model, "input_mode": "delta_rtsp",
        "n_total": n, "k_folds": K, "ensemble_size": M,
        "train_fold_size": train_fold_size, "sizes": sizes,
        "component_order": list(ASR_COMPONENTS),
        "curve": curve, "train_seconds": elapsed,
        "device": tcfg.device, "config": cfg,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[lc_delta] DONE in {elapsed:.1f}s — saved {out_root/'summary.json'}")
    print("[lc_delta] Learning curve (overall MAE kcal/mol):")
    for r in curve:
        print(f"  N={r['N_train']:3d}: {r['overall_mae_kcal_mean']:.3f} ± "
              f"{r['overall_mae_kcal_std']:.3f}")


if __name__ == "__main__":
    main()
