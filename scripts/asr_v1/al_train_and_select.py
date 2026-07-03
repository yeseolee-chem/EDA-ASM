"""Train Δ-M1 5-ensemble on the current LABELED set (mixed dipolar + qmrxn
families) and pick the K most-uncertain candidates from the AL pool.

Outputs (per round_dir):
  cached_features_labeled.pt        — Δ-bundle for the current labels
  ensemble/model_m{0..4}.pt         — trained members (state + baseline)
  predictions.parquet               — per-pool predictions (mean, std)
  picks.csv                         — selected K reactions in seed-CSV format
  picks_summary.json                — round provenance + uncertainty range
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor
from eda_asm.asr_v1.baseline_physics import LinearBaseline, compute_descriptors
from eda_asm.asr_v1.data import ASR_COMPONENTS
from eda_asm.asr_v1.data_multi import (
    iter_seed_rows,
    load_seed_csv,
    normalize_family,
)
from eda_asm.asr_v1.models_delta import ModelM1Delta
from eda_asm.asr_v1.training_delta import (
    CachedFeatureBundleDelta,
    TrainConfigDelta,
    train_one_model_delta,
)


def _cache_labeled_features(
    seed_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    backbone: MACEOFFFeatureExtractor,
    cache_path: Path,
) -> CachedFeatureBundleDelta:
    if cache_path.exists():
        b = CachedFeatureBundleDelta.load(cache_path)
        if len(b.reaction_ids) == len(labels_df) and set(b.reaction_ids) == set(labels_df["reaction_id"]):
            print(f"[train_select] reusing labeled cache: {cache_path} (N={len(b)})")
            return b
        print(f"[train_select] cache mismatch ({len(b)} vs {len(labels_df)}), rebuilding")

    # Join labels with seed to resolve paths per family.
    lab = labels_df[["reaction_id", "family"] + list(ASR_COMPONENTS)].copy()
    lab["seed_family"] = lab["family"].map(normalize_family)
    seed = seed_df.copy()
    joined = lab.merge(seed, left_on=["reaction_id", "seed_family"],
                       right_on=["reaction_id", "family"], how="left",
                       suffixes=("_lab", "_seed"))
    missing_path = joined["path_r"].isna()
    if missing_path.any():
        bad = joined[missing_path]["reaction_id"].tolist()
        raise RuntimeError(f"{len(bad)} labelled reactions missing from seed CSV: {bad[:5]}")

    Rs, Ts, Ps, descs, lbls, ids = [], [], [], [], [], []
    t0 = time.time()
    label_cols = list(ASR_COMPONENTS)
    # Use iter_seed_rows so the dipolar / qmrxn split is consistent.
    for i, sample in enumerate(iter_seed_rows(joined.assign(family=joined["seed_family"]),
                                              labels_df=labels_df)):
        if sample.label is None:
            continue
        Rs.append(backbone.extract(sample.R_atoms))
        Ts.append(backbone.extract(sample.TS_atoms))
        Ps.append(backbone.extract(sample.P_atoms))
        descs.append(compute_descriptors(sample.R_atoms, sample.TS_atoms, sample.P_atoms))
        lbls.append(sample.label); ids.append(sample.reaction_id)
        if (i + 1) % 50 == 0:
            print(f"  labeled [{i+1}/{len(joined)}]  ({time.time()-t0:.0f}s)")

    bundle = CachedFeatureBundleDelta(
        reaction_ids=ids, R_features=Rs, TS_features=Ts, P_features=Ps,
        labels=torch.from_numpy(np.stack(lbls, axis=0)).float(),
        descriptors=torch.from_numpy(np.stack(descs, axis=0)).float(),
        feature_dim=backbone.feature_dim,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(cache_path)
    print(f"[train_select] wrote labeled cache: {cache_path} (N={len(bundle)})")
    return bundle


def _train_ensemble(
    bundle: CachedFeatureBundleDelta, cfg: dict, device: str, ens_dir: Path,
    n_members: int, base_seed: int,
):
    n = len(bundle)
    rng = np.random.default_rng(base_seed)
    members = []
    ens_dir.mkdir(parents=True, exist_ok=True)
    for m_idx in range(n_members):
        perm = rng.permutation(n)
        n_val = max(int(0.10 * n), 4)
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()
        c_m1 = cfg["model_m1"]
        c_train = cfg["train"]
        factory = lambda F: ModelM1Delta(
            feature_dim=F, d_model=c_m1["d_model"], n_heads=c_m1["n_heads"],
            head_hidden=c_m1["head_hidden"], dropout=c_m1["dropout"],
        )
        tcfg = TrainConfigDelta(
            epochs=int(c_train["epochs"]),
            batch_size=int(c_train["batch_size"]),
            lr=float(c_train["lr"]),
            weight_decay=float(c_train["weight_decay"]),
            early_stop_patience=int(c_train["early_stop_patience"]),
            device=device,
            baseline_ridge_alpha=float(cfg.get("delta_baseline", {}).get("ridge_alpha", 1.0)),
        )
        t0 = time.time()
        model, res = train_one_model_delta(
            bundle, factory, train_idx, val_idx, tcfg, seed=base_seed + m_idx,
        )
        print(f"  [ens m{m_idx}] best_ep={res.best_epoch:3d}  stop_ep={res.final_epoch:3d}  "
              f"val_MAE={res.val_mae_overall:.3f}  ({time.time()-t0:.0f}s)")
        torch.save(
            {"model": model.state_dict(), "baseline": res.baseline_state},
            ens_dir / f"model_m{m_idx}.pt",
        )
        members.append((model, res.baseline_state))
    return members


def _predict_pool(members, pool_bundle: CachedFeatureBundleDelta, device: str):
    N = len(pool_bundle); n_mem = len(members)
    preds = np.zeros((n_mem, N, 5), dtype=np.float32)
    desc_np = pool_bundle.descriptors.numpy()
    for mi, (model, bl_state) in enumerate(members):
        bl = LinearBaseline(); bl.load_state_dict(bl_state)
        baseline = torch.from_numpy(bl.predict(desc_np)).float()
        model.eval()
        with torch.no_grad():
            for j in range(N):
                r = pool_bundle.R_features[j].unsqueeze(0).to(device)
                t = pool_bundle.TS_features[j].unsqueeze(0).to(device)
                p = pool_bundle.P_features[j].unsqueeze(0).to(device)
                rm = torch.ones(r.shape[:2], dtype=torch.bool, device=device)
                tm = torch.ones(t.shape[:2], dtype=torch.bool, device=device)
                pm = torch.ones(p.shape[:2], dtype=torch.bool, device=device)
                delta = model(r, rm, t, tm, p, pm)[0].cpu()
                preds[mi, j] = (baseline[j] + delta).numpy()
    return preds.mean(axis=0), preds.std(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1_maceoff_delta_n250.yaml")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--pool", default="outputs/asr_v1/al/pool_features.pt")
    ap.add_argument("--seed-csv",
                    default="ADF_250/seed_selection/initial_seed_v1/selected_reactions.csv")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--round-dir", default=None)
    ap.add_argument("--n-pick", type=int, default=40)
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    repo = Path.cwd()
    labels_path = Path(args.labels) if args.labels else repo / cfg["labels_parquet"]
    seed_path = repo / args.seed_csv
    round_dir = Path(args.round_dir or f"outputs/asr_v1/al/round_{args.round:02d}")
    round_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(labels_path)
    print(f"[train_select] round {args.round}: {len(labels)} labeled  "
          f"families={labels['family'].value_counts().to_dict()}  target picks {args.n_pick}")

    seed = load_seed_csv(seed_path)

    backbone = MACEOFFFeatureExtractor(
        model_size=cfg.get("backbone", {}).get("model_size", "medium"),
        device=device,
        default_dtype=cfg.get("backbone", {}).get("default_dtype", "float32"),
        num_layers=cfg.get("backbone", {}).get("num_layers", -1),
    )

    labeled_cache = round_dir / "cached_features_labeled.pt"
    labeled_bundle = _cache_labeled_features(seed, labels, backbone, labeled_cache)

    ens_dir = round_dir / "ensemble"
    members = _train_ensemble(
        labeled_bundle, cfg, device, ens_dir,
        n_members=args.n_ensemble, base_seed=1000 * args.round,
    )

    pool = CachedFeatureBundleDelta.load(args.pool)
    labeled_ids = set(labels["reaction_id"])
    keep = [i for i, rid in enumerate(pool.reaction_ids) if rid not in labeled_ids]
    if len(keep) < args.n_pick:
        raise RuntimeError(f"pool too small: {len(keep)} remain, need ≥ {args.n_pick}")
    pool_sub = CachedFeatureBundleDelta(
        reaction_ids=[pool.reaction_ids[i] for i in keep],
        R_features=[pool.R_features[i] for i in keep],
        TS_features=[pool.TS_features[i] for i in keep],
        P_features=[pool.P_features[i] for i in keep],
        labels=pool.labels[keep],
        descriptors=pool.descriptors[keep],
        feature_dim=pool.feature_dim,
    )
    print(f"[train_select] pool after labeled-removal: {len(pool_sub)}")

    t0 = time.time()
    pred_mean, pred_std = _predict_pool(members, pool_sub, device)
    print(f"[train_select] predicted on pool in {time.time()-t0:.1f}s")

    # Per-channel label std (from current labels — mixed families).
    label_std = np.array(
        [float(labels[c].std(ddof=1)) for c in ASR_COMPONENTS], dtype=np.float32,
    )
    # Uncertainty = Σ_c std_c / label_std_c
    scores = (pred_std / np.maximum(label_std, 1e-6)).sum(axis=1)
    picks_idx = np.argsort(-scores)[: args.n_pick]

    # Predictions table for ALL pool reactions (audit).
    pred_rows = []
    for j, rid in enumerate(pool_sub.reaction_ids):
        row = {"reaction_id": rid, "uncertainty_score": float(scores[j])}
        for ci, name in enumerate(ASR_COMPONENTS):
            row[f"{name}_pred_mean"] = float(pred_mean[j, ci])
            row[f"{name}_pred_std"] = float(pred_std[j, ci])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_parquet(round_dir / "predictions.parquet")

    # Mini seed CSV for the K picks (look up in master seed CSV).
    seed_lookup = {row["reaction_id"]: row for _, row in seed.iterrows()}
    pick_rows = []
    for pi in picks_idx:
        rid = pool_sub.reaction_ids[pi]
        if rid not in seed_lookup:
            print(f"  WARNING: pick {rid} not in seed CSV — skipping")
            continue
        pick_rows.append(seed_lookup[rid])
    picks_df = pd.DataFrame(pick_rows)
    picks_df.to_csv(round_dir / "picks.csv", index=False)

    summary = {
        "round": args.round,
        "n_labeled_at_start": int(len(labels)),
        "n_pool_after_filter": int(len(pool_sub)),
        "n_pick": int(args.n_pick),
        "n_ensemble": int(args.n_ensemble),
        "label_std": label_std.tolist(),
        "families_picked": picks_df["family"].value_counts().to_dict(),
        "uncertainty_min": float(scores[picks_idx].min()),
        "uncertainty_max": float(scores[picks_idx].max()),
        "device": device,
        "picks": [
            {
                "reaction_id": pool_sub.reaction_ids[int(pi)],
                "family": str(picks_df.iloc[k]["family"]) if k < len(picks_df) else "?",
                "uncertainty_score": float(scores[int(pi)]),
                "pred_mean": pred_mean[int(pi)].tolist(),
                "pred_std": pred_std[int(pi)].tolist(),
            }
            for k, pi in enumerate(picks_idx)
        ],
    }
    (round_dir / "picks_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train_select] wrote picks: {round_dir/'picks.csv'} ({len(picks_df)} rows)  "
          f"families={summary['families_picked']}")
    print(f"[train_select] uncertainty: min={summary['uncertainty_min']:.3f}  "
          f"max={summary['uncertainty_max']:.3f}")


if __name__ == "__main__":
    main()
