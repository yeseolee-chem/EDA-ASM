"""SPEC_06 — 2-step learner: 28-d per-channel XGB base (b) + MACE cross-attention δ.

Per outer fold f:
  1) X28 = hstack[m3 24-d, d25, d26, d27, d28]  (spec05 no-sum 28-d composition)
  2) b_oof = cross_fit_oof("xgb", X28[train], Y[train], K=5, seed=…)  — δ target
  3) b_full = fit_full("xgb", X28[train], Y[train]); b_val = predict on test
  4) train δ (ModelM1Delta) on residual (y_train − b_oof); ŷ_test = b_val + δ(test)

Idempotent: skip if member{M}.json exists.
Cross-fit gate #3: abort if median|r_train| < 0.05 · median|y − ȳ|.

CLI:
  --fold {0..4}    (default: $SLURM_ARRAY_TASK_ID)
  --member N       (default: 0)
  --device cuda|cpu
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

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec06_2step_xgb28_delta"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
sys.path.insert(0, str(SPEC / "code"))

from eda_asm.asr_v1.models_delta import ModelM1Delta
from eda_asm.asr_v1.baseline_physics import LinearBaseline
from eda_asm.asr_v1.training_delta import (
    CachedFeatureBundleDelta, TrainConfigDelta, train_one_model_delta,
)
from baselines import cross_fit_oof, fit_full, predict_full  # noqa: E402
from descriptors28 import build_X28  # noqa: E402

BUNDLE_PT = Path(os.environ.get(
    "BUNDLE_PT",
    "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
FOLDS_JSON = SPEC / "splits/outer_folds.json"
OUT_ROOT = SPEC / "oof/xgb28_delta"

LR = 1.0e-5
EPOCHS_MAX = 100_000
PATIENCE = 10_000
BATCH = 16
WD = 1.0e-3
M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
SEED_BASE = 42
CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]


def slice_bundle(bundle, rids_keep_set):
    keep = [i for i, r in enumerate(bundle.reaction_ids) if r in rids_keep_set]
    return CachedFeatureBundleDelta(
        reaction_ids=[bundle.reaction_ids[i] for i in keep],
        R_features=[bundle.R_features[i] for i in keep],
        TS_features=[bundle.TS_features[i] for i in keep],
        P_features=[bundle.P_features[i] for i in keep],
        labels=bundle.labels[keep],
        descriptors=bundle.descriptors[keep],
        feature_dim=bundle.feature_dim,
    )


def train_one_cell(bundle_full, X, Y, rids, fold_idx, tr_idx, te_idx,
                   member, device):
    # spec02 xgb_delta uses seed = SEED_BASE + member*1000 + fold*100 + 50.
    # Same recipe here (this is the "xgb" arm at 28-d).
    seed = SEED_BASE + member * 1000 + fold_idx * 100 + 50
    print(f"[xgb28+delta] fold{fold_idx} member{member} seed={seed}", flush=True)

    b_oof = cross_fit_oof("xgb", X[tr_idx], Y[tr_idx], K=5, seed=seed)
    r_train = Y[tr_idx] - b_oof
    med_r = float(np.median(np.abs(r_train)))
    med_y = float(np.median(np.abs(Y[tr_idx] - Y[tr_idx].mean(0))))
    print(f"  median|r_train|={med_r:.3f}  median|y-ybar|={med_y:.3f}  "
          f"ratio={med_r/max(med_y,1e-9):.3f}", flush=True)
    if med_r < 0.05 * med_y:
        raise RuntimeError(
            f"cross-fit gate #3 failed: med|r_train|={med_r:.4f} vs "
            f"med|y-ybar|={med_y:.4f} (ratio<{0.05})"
        )

    b_full = fit_full("xgb", X[tr_idx], Y[tr_idx])
    b_val = predict_full("xgb", b_full, X[te_idx])

    # quick sanity: b_oof NMAE per channel (should ≈ spec05 xgb_28d fold0 numbers)
    for c_i, ch in enumerate(CHANNELS):
        yt_c = Y[tr_idx][:, c_i]
        yp_c = b_oof[:, c_i]
        mad = float(np.mean(np.abs(yt_c - yt_c.mean())))
        nm = float(np.mean(np.abs(yt_c - yp_c)) / (mad + 1e-12))
        print(f"  b_oof(train) NMAE[{ch}]={nm:.3f}", flush=True)

    rids_tr = set(rids[tr_idx].tolist()); rids_te = set(rids[te_idx].tolist())
    keep_set = rids_tr | rids_te
    bundle = slice_bundle(bundle_full, keep_set)
    r2i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    train_positions_all = [r2i[r] for r in rids[tr_idx] if r in r2i]
    test_positions = [r2i[r] for r in rids[te_idx] if r in r2i]

    rng = np.random.default_rng(seed)
    arr = list(train_positions_all); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    train_pos, val_pos = arr[n_val:], arr[:n_val]

    injected_train = {rids[tr_idx][i]: b_oof[i] for i in range(len(tr_idx))}
    injected_test  = {rids[te_idx][i]: b_val[i] for i in range(len(te_idx))}
    injected_all = {**injected_train, **injected_test}

    class FixedBaseline(LinearBaseline):
        """Impersonates LinearBaseline but returns a precomputed b vector
        aligned to bundle.reaction_ids (spec02 trick)."""
        def __init__(self, rid_map, bundle_rids):
            super().__init__(alpha=1.0)
            arr = np.array([rid_map[r] for r in bundle_rids], dtype=np.float32)
            self._arr = arr
            self.W_ = np.zeros((1, 5), dtype=np.float32)
            self.d_in_ = 1
            self.d_mean_ = np.zeros(1, dtype=np.float32)
            self.d_std_ = np.ones(1, dtype=np.float32)

        def fit(self, D_train, Y_train):
            return self

        def predict(self, D):
            n = D.shape[0]
            if n == self._arr.shape[0]:
                return self._arr.copy()
            return np.zeros((n, 5), dtype=np.float32)

    from eda_asm.asr_v1 import training_delta as tdmod
    orig_lb = tdmod.LinearBaseline
    tdmod.LinearBaseline = lambda **kw: FixedBaseline(injected_all, bundle.reaction_ids)

    try:
        factory = lambda F: ModelM1Delta(feature_dim=F, **M1_HP)
        cfg = TrainConfigDelta(
            epochs=EPOCHS_MAX, batch_size=BATCH, lr=LR, weight_decay=WD,
            early_stop_patience=PATIENCE, device=device, baseline_ridge_alpha=1.0,
        )
        model, fr = train_one_model_delta(bundle, factory, train_pos, val_pos, cfg, seed=seed)
    finally:
        tdmod.LinearBaseline = orig_lb

    model.eval()
    yt = bundle.labels[test_positions].numpy()
    baseline_test = np.array([injected_test[r] for r in
                              [bundle.reaction_ids[i] for i in test_positions]])
    preds = []
    with torch.no_grad():
        for i in test_positions:
            r_f = bundle.R_features[i].to(device).unsqueeze(0)
            t_f = bundle.TS_features[i].to(device).unsqueeze(0)
            p_f = bundle.P_features[i].to(device).unsqueeze(0)
            r_m = torch.ones(r_f.shape[:2], dtype=torch.bool, device=device)
            t_m = torch.ones(t_f.shape[:2], dtype=torch.bool, device=device)
            p_m = torch.ones(p_f.shape[:2], dtype=torch.bool, device=device)
            delta = model(r_f, r_m, t_f, t_m, p_f, p_m).cpu().numpy().flatten()
            preds.append(delta)
    delta_test = np.array(preds)
    yp = baseline_test + delta_test
    return (yt, yp,
            [bundle.reaction_ids[i] for i in test_positions],
            baseline_test, delta_test,
            fr, med_r, med_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.fold is None:
        args.fold = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    print(f"[SPEC_06] fold={args.fold}  member={args.member}  device={args.device}", flush=True)

    out_dir = OUT_ROOT / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member{args.member}.json"
    if out_path.exists():
        print(f"[skip] {out_path}", flush=True); return

    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy().astype(np.float64)
    Y = b["labels"].numpy().astype(np.float64)
    X28, ok = build_X28(rids, X24)
    n_ok = int(ok.sum()); n_zero = int(len(ok) - n_ok)
    print(f"  X28 shape={X28.shape}  proxies_ok={n_ok}  zero_fill={n_zero}", flush=True)

    bundle_full = CachedFeatureBundleDelta(
        reaction_ids=b["reaction_ids"],
        R_features=b["R_features"], TS_features=b["TS_features"], P_features=b["P_features"],
        labels=b["labels"].float(), descriptors=b["descriptors"].float(),
        feature_dim=int(b["feature_dim"]),
    )
    folds = json.load(open(FOLDS_JSON))
    fold_info = folds[str(args.fold)]
    tr_idx = np.array([np.where(rids == r)[0][0] for r in fold_info["train"]])
    te_idx = np.array([np.where(rids == r)[0][0] for r in fold_info["test"]])

    t0 = time.time()
    yt, yp, te_rids, b_val, delta_test, fr, med_r, med_y = train_one_cell(
        bundle_full, X28, Y, rids,
        args.fold, tr_idx, te_idx, args.member, args.device,
    )
    bt = yt.sum(axis=1); bp = yp.sum(axis=1)
    result = {
        "arm": "xgb28_delta",
        "fold": args.fold, "member": args.member,
        "reaction_ids": te_rids,
        **{f"y_true_{c}": [float(v) for v in yt[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"y_pred_{c}": [float(v) for v in yp[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"b_test_{c}": [float(v) for v in b_val[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"delta_test_{c}": [float(v) for v in delta_test[:, i]] for i, c in enumerate(CHANNELS)},
        "barrier_true": [float(v) for v in bt],
        "barrier_pred": [float(v) for v in bp],
        "best_epoch": int(fr.best_epoch), "final_epoch": int(fr.final_epoch),
        "early_stopped": bool(fr.early_stopped),
        "cross_fit_median_r_train": med_r,
        "cross_fit_median_y_bar": med_y,
        "proxies_zero_filled": n_zero,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path} ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
