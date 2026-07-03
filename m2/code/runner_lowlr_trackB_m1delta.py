"""Track B / m1_delta + geom6 baseline, with low-LR + larger budget.

Per user spec (2026-06-24):
  - cross-attention + delta-learning (m1_delta) only
  - baseline = geom6 (d1~d6 physics ridge, NO xTB)
  - LR: 1e-3 → 1e-5
  - max epochs: 200 → 3000
  - early-stop patience: 30 → 500

One SLURM array task = one (fold, member) cell.
SLURM_ARRAY_TASK_ID = fold * 5 + member (range 0..24).

Output per cell:
  trackB_lowlr_geom6/m1_delta/fold{F}/member{M}.json
  {
    "fold": F, "member": M,
    "n_train", "n_val", "n_test",
    "reaction_ids": [test rids],
    "y_true":       [[5]],
    "y_pred":       [[5]],
    "val_mae_per_channel", "test_mae_per_channel", "test_mae_mean_kcal",
    "best_epoch", "final_epoch", "early_stopped",
    "hp": {lr, epochs, patience, ...}
  }
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/asr_v1"))

from eda_asm.asr_v1.data import ASR_COMPONENTS
from eda_asm.asr_v1.models_delta import ModelM1Delta
from eda_asm.asr_v1.baseline_physics import LinearBaseline
from eda_asm.asr_v1.training_delta import (
    CachedFeatureBundleDelta, TrainConfigDelta, train_one_model_delta,
)

HERE = Path(__file__).resolve().parent
# BASELINE selectable via env var: geom6 (no xTB) | xtb | xtb_geom6 (xTB+d1~d6)
BASELINE = os.environ.get("BASELINE", "geom6")
BUNDLE = HERE / "bundles" / f"features_v6_delta_{BASELINE}.pt"
# SUBSAMPLES_TAG controls which fold-split dir is used.
#   default = trackB (full 789 pool, includes OOD)
#   no_ood  = trackB_no_ood (4 OOD strain-channel outliers removed from train AND test)
SUBSAMPLES_TAG = os.environ.get("SUBSAMPLES_TAG", "trackB")
SUBSAMPLES = ROOT / "outputs/asr_v1/phase3/subsamples" / SUBSAMPLES_TAG
OUT_TAG = os.environ.get("OUT_TAG", "lowlr")  # output dir suffix
OUT_BASE = HERE / f"trackB_{OUT_TAG}_{BASELINE}" / "m1_delta"

# === User-requested HP (low LR + very large budget per 2026-06-24 update) ===
LR_LOW = 1.0e-5
EPOCHS_MAX = 100_000      # 100k epochs (vs 3000 prior)
PATIENCE = 10_000         # 10k patience (vs 500 prior)
BATCH = 16
WD = 1.0e-3

# Architecture HP (unchanged from spec)
M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
RIDGE_ALPHA = 1.0
# size_*.json filename in subsamples dir. The no_ood variant kept the same
# filename `size_509.json` so the same runner works for both pools (contents
# differ: 509 original vs 506-508 after OOD removal).
SIZE_FULL = 509
SEED_BASE = 42


def slice_delta(bundle: CachedFeatureBundleDelta, rids: list[str]) -> CachedFeatureBundleDelta:
    rid_to_i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    keep = [rid_to_i[r] for r in rids if r in rid_to_i]
    return CachedFeatureBundleDelta(
        reaction_ids=[bundle.reaction_ids[i] for i in keep],
        R_features=[bundle.R_features[i] for i in keep],
        TS_features=[bundle.TS_features[i] for i in keep],
        P_features=[bundle.P_features[i] for i in keep],
        labels=bundle.labels[keep],
        descriptors=bundle.descriptors[keep],
        feature_dim=bundle.feature_dim,
    )


def load_indices_trackB(fold: int):
    test_rids = json.load(open(SUBSAMPLES / f"fold{fold}" / "test_rids.json"))
    train_rids = json.load(open(SUBSAMPLES / f"fold{fold}" / f"size_{SIZE_FULL}.json"))
    return train_rids, test_rids


def make_val_split(train_pos, fold, member):
    """Member-dependent val carve-out for ensemble variance."""
    rng = np.random.default_rng(SEED_BASE + fold * 100 + member * 17)
    arr = list(train_pos); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    return arr[n_val:], arr[:n_val]


def evaluate_predictions(model, bundle: CachedFeatureBundleDelta, test_pos, baseline_pred_np, device):
    """Return (y_true, y_pred, per-channel MAE)."""
    model.eval()
    y_true = bundle.labels[test_pos].numpy()
    preds = []
    with torch.no_grad():
        for i in test_pos:
            r_f = bundle.R_features[i].to(device).unsqueeze(0)
            t_f = bundle.TS_features[i].to(device).unsqueeze(0)
            p_f = bundle.P_features[i].to(device).unsqueeze(0)
            r_m = torch.ones(r_f.shape[:2], dtype=torch.bool, device=device)
            t_m = torch.ones(t_f.shape[:2], dtype=torch.bool, device=device)
            p_m = torch.ones(p_f.shape[:2], dtype=torch.bool, device=device)
            delta = model(r_f, r_m, t_f, t_m, p_f, p_m).cpu().numpy().flatten()
            preds.append(baseline_pred_np[i] + delta)
    y_pred = np.array(preds)
    test_mae = np.abs(y_pred - y_true).mean(axis=0)
    return y_true, y_pred, test_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--member", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # SLURM array mapping: 1 task = 1 fold = 5 members (sequential).
    # If --fold not given, run all 5 members for SLURM_ARRAY_TASK_ID's fold.
    if args.fold is None:
        args.fold = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        print(f"[array] task_id={args.fold} → fold={args.fold}, running all 5 members", flush=True)
    if args.member is None:
        # Loop all members for this fold
        for m in range(5):
            args2 = argparse.Namespace(**vars(args))
            args2.member = m
            run_one(args2)
        return
    run_one(args)


def run_one(args):
    assert 0 <= args.fold <= 4 and 0 <= args.member <= 4
    out_dir = OUT_BASE / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member{args.member}.json"

    if out_path.exists():
        print(f"[skip] {out_path} exists — already done", flush=True)
        return

    print(f"=== fold={args.fold} member={args.member} | baseline={BASELINE} | "
          f"lr={LR_LOW} epochs={EPOCHS_MAX} patience={PATIENCE} ===", flush=True)
    t0 = time.time()

    # Load bundle
    full = CachedFeatureBundleDelta.load(BUNDLE)
    print(f"  bundle: {len(full.reaction_ids)} reactions, feat_dim={full.feature_dim}", flush=True)

    # Load fold indices
    train_rids, test_rids = load_indices_trackB(args.fold)
    keep = list(set(train_rids) | set(test_rids))
    bundle = slice_delta(full, keep)

    rid_to_i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    train_pos_all = [rid_to_i[r] for r in train_rids if r in rid_to_i]
    test_pos = [rid_to_i[r] for r in test_rids if r in rid_to_i]
    train_pos, val_pos = make_val_split(train_pos_all, args.fold, args.member)

    print(f"  N: train={len(train_pos)} val={len(val_pos)} test={len(test_pos)}", flush=True)

    # Build cross-attention + Δ-learning model
    factory = lambda F: ModelM1Delta(feature_dim=F, **M1_HP)

    cfg = TrainConfigDelta(
        epochs=EPOCHS_MAX,
        batch_size=BATCH,
        lr=LR_LOW,
        weight_decay=WD,
        early_stop_patience=PATIENCE,
        device=args.device,
        baseline_ridge_alpha=RIDGE_ALPHA,
    )
    seed = SEED_BASE + args.member * 1000 + args.fold * 100

    model, fr = train_one_model_delta(bundle, factory, train_pos, val_pos, cfg, seed=seed)

    # Reconstruct baseline predictions
    bl = LinearBaseline(); bl.load_state_dict(fr.baseline_state)
    baseline_all = bl.predict(bundle.descriptors.numpy())

    # Evaluate on test
    y_true, y_pred, test_mae = evaluate_predictions(model, bundle, test_pos, baseline_all, args.device)
    test_rids_in_order = [bundle.reaction_ids[i] for i in test_pos]

    result = {
        "fold": args.fold, "member": args.member, "seed": seed,
        "n_train": len(train_pos), "n_val": len(val_pos), "n_test": len(test_pos),
        "reaction_ids": test_rids_in_order,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "components": list(ASR_COMPONENTS),
        "val_mae_per_channel": list(map(float, fr.val_mae_per_component)),
        "val_mae_baseline_only": list(map(float, fr.val_mae_baseline_only)),
        "test_mae_per_channel": list(map(float, test_mae)),
        "test_mae_mean_kcal": float(test_mae.mean()),
        "best_epoch": int(fr.best_epoch),
        "final_epoch": int(fr.final_epoch),
        "early_stopped": bool(fr.early_stopped),
        "elapsed_s": time.time() - t0,
        "hp": {
            "lr": LR_LOW, "epochs_max": EPOCHS_MAX,
            "early_stop_patience": PATIENCE,
            "batch_size": BATCH, "weight_decay": WD,
            "arch": "m1_delta (cross-attn + Δ-learning)",
            "baseline": BASELINE,
            "ridge_alpha": RIDGE_ALPHA,
            "d_model": M1_HP["d_model"], "n_heads": M1_HP["n_heads"],
        },
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  test MAE: {[f'{v:.2f}' for v in test_mae]}", flush=True)
    print(f"  best_epoch={fr.best_epoch} final_epoch={fr.final_epoch} early_stopped={fr.early_stopped}", flush=True)
    print(f"  wrote → {out_path}  ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
