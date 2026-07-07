"""Arm B / C per-(fold, arm) runner for the A/B/C ablation.

One cell = one (arm ∈ {B, C}, outer fold ∈ 0..4). SLURM_ARRAY_TASK_ID
maps as   task_id = arm_idx * 5 + fold_idx   (arm_idx: B=0, C=1).

Per-cell contract:
  1. Load fixed splits (outer_folds.json) — same for A/B/C.
  2. On the outer-train slice, produce anti-leakage OOF baseline via
     inner K'=5 cross-fit (`b_oof`).
  3. Fit `b_full` on the full outer-train slice.
  4. Assemble baseline_all: rows in train_idx → b_oof, rows in val_idx →
     b_full.predict.
  5. Train the delta head (identical arch/config in B and C).
  6. Evaluate on val_idx → append to per-fold JSON.

Output:
  results/cells/{arm}/fold{F}.json   with y_true / y_pred / baseline_only.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "m3" / "code"))

from baselines import BASELINE_REGISTRY, cross_fit_oof  # noqa: E402
from delta_trainer import train_one_delta_custom  # noqa: E402
from eda_asm.asr_v1.models_delta import ModelM1Delta  # noqa: E402
from eda_asm.asr_v1.training_delta import (  # noqa: E402
    CachedFeatureBundleDelta, TrainConfigDelta,
)


CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
SEED = 42
INNER_K = 5

# δ hyper-parameters — identical for B and C.
LR_LOW = 1.0e-5
EPOCHS_MAX = int(1e5)
PATIENCE = 10_000
BATCH = 16
WD = 1.0e-3
M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)

ARM_TO_BASELINE = {"B": "ridge", "C": "xgb"}


def parse_task(task_id: int) -> tuple[str, int]:
    """task_id 0..9 → (arm, fold). B (ridge_delta) task_id 0..4, C 5..9."""
    if task_id < 0 or task_id > 9:
        raise SystemExit(f"task_id out of range: {task_id}")
    return ("B" if task_id < 5 else "C", task_id % 5)


def slice_delta(bundle, keep_pos):
    return CachedFeatureBundleDelta(
        reaction_ids=[bundle.reaction_ids[i] for i in keep_pos],
        R_features=[bundle.R_features[i] for i in keep_pos],
        TS_features=[bundle.TS_features[i] for i in keep_pos],
        P_features=[bundle.P_features[i] for i in keep_pos],
        labels=bundle.labels[keep_pos],
        descriptors=bundle.descriptors[keep_pos],
        feature_dim=bundle.feature_dim,
    )


def make_val_split(train_pos, fold, seed=SEED):
    """15% val carve-out (deterministic per fold)."""
    rng = np.random.default_rng(seed + fold * 101)
    arr = list(train_pos); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    return arr[n_val:], arr[:n_val]


def eval_delta(model, bundle, val_pos, baseline_all, device):
    """Return ŷ, y arrays over val_pos."""
    model.eval()
    y_true = bundle.labels[val_pos].numpy()
    preds = []
    with torch.no_grad():
        for i in val_pos:
            r = bundle.R_features[i].to(device).unsqueeze(0)
            t = bundle.TS_features[i].to(device).unsqueeze(0)
            p = bundle.P_features[i].to(device).unsqueeze(0)
            rm = torch.ones(r.shape[:2], dtype=torch.bool, device=device)
            tm = torch.ones(t.shape[:2], dtype=torch.bool, device=device)
            pm = torch.ones(p.shape[:2], dtype=torch.bool, device=device)
            delta = model(r, rm, t, tm, p, pm).cpu().numpy().flatten()
            preds.append(baseline_all[i] + delta)
    return y_true, np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None,
                    help="0..9 (arm × fold). Overrides --arm/--fold if given.")
    ap.add_argument("--arm", choices=["B", "C"], default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--descriptor-set", choices=["m1", "m2", "m3"], default="m3")
    ap.add_argument("--bundle", type=str, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    tid = args.task
    if tid is None:
        tid_env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid_env is not None:
            tid = int(tid_env)
    if tid is not None:
        args.arm, args.fold = parse_task(tid)
    if args.arm is None or args.fold is None:
        raise SystemExit("Need --task or (--arm and --fold).")

    if args.bundle is None:
        args.bundle = str(REPO / f"pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_{args.descriptor_set}.pt")

    out_dir = HERE / "results" / "cells" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{args.fold}.json"
    if out_path.exists():
        print(f"[skip] {out_path} already exists", flush=True)
        return

    print(f"=== arm={args.arm} fold={args.fold} descriptor={args.descriptor_set} "
          f"device={args.device} ===", flush=True)
    t0 = time.time()

    # ---------- Data ----------
    splits = json.load(open(HERE / "splits" / "outer_folds.json"))
    assert splits["seed"] == SEED
    fd = next(f for f in splits["folds"] if f["fold"] == args.fold)
    train_rids = fd["train_rids"]
    test_rids = fd["test_rids"]

    full = CachedFeatureBundleDelta.load(args.bundle)
    rid_to_i = {r: i for i, r in enumerate(full.reaction_ids)}
    keep = [rid_to_i[r] for r in train_rids + test_rids if r in rid_to_i]
    bundle = slice_delta(full, keep)
    rid2j = {r: j for j, r in enumerate(bundle.reaction_ids)}
    train_pos_all = [rid2j[r] for r in train_rids if r in rid2j]
    test_pos = [rid2j[r] for r in test_rids if r in rid2j]
    train_pos, val_pos = make_val_split(train_pos_all, args.fold, seed=SEED)

    print(f"  N: train={len(train_pos)} val={len(val_pos)} test={len(test_pos)} "
          f"| bundle_feat_dim={bundle.feature_dim}", flush=True)

    # ---------- Baseline ----------
    baseline_fn = BASELINE_REGISTRY[ARM_TO_BASELINE[args.arm]]
    X = bundle.descriptors.numpy()
    Y = bundle.labels.numpy()

    # b_oof over the outer-train pool (train + val — both are non-test).
    outer_train_pos = np.array(train_pos + val_pos)
    b_oof_outer = cross_fit_oof(
        baseline_fn, X[outer_train_pos], Y[outer_train_pos],
        k=INNER_K, seed=SEED + args.fold,
        fit_kwargs=({"seed": SEED + args.fold}
                    if ARM_TO_BASELINE[args.arm] == "xgb" else {}),
    )
    # b_full → predict on the test (unseen) positions.
    b_full = baseline_fn(X[outer_train_pos], Y[outer_train_pos],
                        **({"seed": SEED + args.fold}
                           if ARM_TO_BASELINE[args.arm] == "xgb" else {}))
    b_test = b_full.predict(X[test_pos])

    baseline_all = np.zeros_like(Y, dtype=np.float32)
    baseline_all[outer_train_pos] = b_oof_outer
    baseline_all[test_pos] = b_test

    # Gate #3 — δ target must be non-trivially non-zero.
    r_train = Y[train_pos] - baseline_all[train_pos]
    r_train_ratio = float(
        np.median(np.abs(r_train)) / max(np.median(np.abs(Y[train_pos])), 1e-6)
    )
    print(f"  median|r_train| / median|y_train| = {r_train_ratio:.3f} "
          f"(gate #3: >0.02 expected)", flush=True)
    assert r_train_ratio > 0.02, "gate #3 failed — δ target ≈ 0"

    # ---------- Delta training ----------
    factory = lambda F: ModelM1Delta(feature_dim=F, **M1_HP)
    cfg = TrainConfigDelta(
        epochs=EPOCHS_MAX, batch_size=BATCH, lr=LR_LOW, weight_decay=WD,
        early_stop_patience=PATIENCE, device=args.device,
        baseline_ridge_alpha=1.0,
    )
    seed = SEED + args.fold * 100
    model, fr = train_one_delta_custom(
        bundle=bundle, model_factory=factory,
        train_idx=train_pos, val_idx=val_pos, cfg=cfg,
        baseline_all_np=baseline_all, seed=seed,
    )

    # ---------- Test evaluation ----------
    y_true_te, y_pred_te = eval_delta(model, bundle, test_pos, baseline_all, args.device)
    y_true_va, y_pred_va = eval_delta(model, bundle, val_pos,  baseline_all, args.device)
    test_rids_ordered = [bundle.reaction_ids[i] for i in test_pos]
    val_rids_ordered = [bundle.reaction_ids[i] for i in val_pos]

    # Baseline-only diagnostic
    baseline_only_test = baseline_all[test_pos]

    result = {
        "arm": args.arm,
        "baseline": ARM_TO_BASELINE[args.arm],
        "descriptor_set": args.descriptor_set,
        "fold": args.fold,
        "seed": seed,
        "n_train": len(train_pos), "n_val": len(val_pos), "n_test": len(test_pos),
        "gate3_r_train_ratio": r_train_ratio,
        "components": CH,
        "test_rids": test_rids_ordered,
        "y_true_test": y_true_te.tolist(),
        "y_pred_test": y_pred_te.tolist(),
        "baseline_only_test": baseline_only_test.tolist(),
        "val_rids": val_rids_ordered,
        "y_true_val": y_true_va.tolist(),
        "y_pred_val": y_pred_va.tolist(),
        "val_mae_per_channel": list(map(float, fr.val_mae_per_component)),
        "val_mae_baseline_only": list(map(float, fr.val_mae_baseline_only)),
        "best_epoch": int(fr.best_epoch),
        "final_epoch": int(fr.final_epoch),
        "early_stopped": bool(fr.early_stopped),
        "elapsed_s": time.time() - t0,
        "hp": {
            "lr": LR_LOW, "epochs_max": EPOCHS_MAX,
            "early_stop_patience": PATIENCE, "batch_size": BATCH,
            "weight_decay": WD, "arch": "m1_delta (cross-attn)",
            "grad_clip": 5.0, "inner_k": INNER_K,
            "d_model": M1_HP["d_model"], "n_heads": M1_HP["n_heads"],
        },
    }
    out_path.write_text(json.dumps(result, indent=2))
    per_ch = np.abs(y_pred_te - y_true_te).mean(axis=0)
    print(f"  test MAE per channel: {per_ch.round(3).tolist()}", flush=True)
    print(f"  wrote → {out_path}  ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
