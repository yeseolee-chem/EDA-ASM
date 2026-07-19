"""SPEC_07 — λ-contribution sweep runner.

For each (λ, fold) cell:
  * If λ == 1.0: base-only. Fit xgb 28-d on train, predict on test, ŷ = b_test.
  * Else: cross-fit b_oof + b_full (spec06 recipe), then train ModelM1Delta with
    TrainConfigDelta(w_base=λ, w_delta=(1-λ)); ŷ_test = λ·b_test + (1-λ)·δ(test).

Output JSON schema mirrors spec06 with an added "lam" field.

Idempotent: skip if member{M}.json exists.

CLI:
  --lam FLOAT       required (e.g. 0.0, 0.25, 0.5, 0.75, 1.0)
  --fold {0..4}     default: $SLURM_ARRAY_TASK_ID
  --member N        default: 0
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
SPEC = REPO / "spec/spec07_lambda_contribution"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
sys.path.insert(0, str(REPO / "spec/spec06_2step_xgb28_delta/code"))

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
OOF_ROOT = SPEC / "oof"

LR = 1.0e-5
EPOCHS_MAX = 100_000
PATIENCE = 10_000
BATCH = 16
WD = 1.0e-3
M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
SEED_BASE = 42
CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]


def lam_tag(lam: float) -> str:
    # 0.0 -> "0p00", 0.25 -> "0p25", 1.0 -> "1p00"
    return f"{lam:.2f}".replace(".", "p")


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


def run_base_only_cell(X, Y, rids, tr_idx, te_idx, fold_idx, member):
    seed = SEED_BASE + member * 1000 + fold_idx * 100 + 50
    print(f"[λ=1 base-only] fold{fold_idx} member{member} seed={seed}", flush=True)
    b_full = fit_full("xgb", X[tr_idx], Y[tr_idx])
    b_val = predict_full("xgb", b_full, X[te_idx])
    yt = Y[te_idx]
    yp = b_val  # ŷ = b
    te_rids = [str(r) for r in rids[te_idx]]
    return yt, yp, te_rids, b_val, np.zeros_like(b_val), None


def run_delta_cell(bundle_full, X, Y, rids, tr_idx, te_idx, fold_idx, member,
                   lam, device):
    seed = SEED_BASE + member * 1000 + fold_idx * 100 + 50
    print(f"[λ={lam:.2f}] fold{fold_idx} member{member} seed={seed}", flush=True)

    b_oof = cross_fit_oof("xgb", X[tr_idx], Y[tr_idx], K=5, seed=seed)
    b_full = fit_full("xgb", X[tr_idx], Y[tr_idx])
    b_val = predict_full("xgb", b_full, X[te_idx])

    # b_oof NMAE per channel — sanity print
    for c_i, ch in enumerate(CHANNELS):
        yt_c = Y[tr_idx][:, c_i]; yp_c = b_oof[:, c_i]
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
            w_base=float(lam), w_delta=float(1.0 - lam),
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
    yp = lam * baseline_test + (1.0 - lam) * delta_test
    te_rids = [bundle.reaction_ids[i] for i in test_positions]
    return yt, yp, te_rids, baseline_test, delta_test, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.fold is None:
        args.fold = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    lam = float(args.lam)
    tag = lam_tag(lam)
    print(f"[SPEC_07] λ={lam}  tag={tag}  fold={args.fold}  member={args.member}  "
          f"device={args.device}", flush=True)

    out_dir = OOF_ROOT / f"lam{tag}" / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member{args.member}.json"
    if out_path.exists():
        print(f"[skip] {out_path}", flush=True); return

    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy().astype(np.float64)
    Y = b["labels"].numpy().astype(np.float64)
    X28, ok = build_X28(rids, X24)
    n_zero = int(len(ok) - int(ok.sum()))
    print(f"  X28={X28.shape}  proxies_zero_fill={n_zero}", flush=True)

    folds = json.load(open(FOLDS_JSON))
    fold_info = folds[str(args.fold)]
    tr_idx = np.array([np.where(rids == r)[0][0] for r in fold_info["train"]])
    te_idx = np.array([np.where(rids == r)[0][0] for r in fold_info["test"]])

    t0 = time.time()
    if lam >= 1.0 - 1e-9:
        yt, yp, te_rids, b_val, delta_test, fr = run_base_only_cell(
            X28, Y, rids, tr_idx, te_idx, args.fold, args.member,
        )
    else:
        bundle_full = CachedFeatureBundleDelta(
            reaction_ids=b["reaction_ids"],
            R_features=b["R_features"], TS_features=b["TS_features"],
            P_features=b["P_features"],
            labels=b["labels"].float(), descriptors=b["descriptors"].float(),
            feature_dim=int(b["feature_dim"]),
        )
        yt, yp, te_rids, b_val, delta_test, fr = run_delta_cell(
            bundle_full, X28, Y, rids, tr_idx, te_idx,
            args.fold, args.member, lam, args.device,
        )

    bt = yt.sum(axis=1); bp = yp.sum(axis=1)
    result = {
        "arm": "lambda_blend",
        "lam": lam,
        "fold": args.fold, "member": args.member,
        "reaction_ids": [str(r) for r in te_rids],
        **{f"y_true_{c}": [float(v) for v in yt[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"y_pred_{c}": [float(v) for v in yp[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"b_test_{c}": [float(v) for v in b_val[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"delta_test_{c}": [float(v) for v in delta_test[:, i]] for i, c in enumerate(CHANNELS)},
        "barrier_true": [float(v) for v in bt],
        "barrier_pred": [float(v) for v in bp],
        "best_epoch": int(fr.best_epoch) if fr is not None else -1,
        "final_epoch": int(fr.final_epoch) if fr is not None else -1,
        "early_stopped": bool(fr.early_stopped) if fr is not None else True,
        "proxies_zero_filled": n_zero,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path} ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
