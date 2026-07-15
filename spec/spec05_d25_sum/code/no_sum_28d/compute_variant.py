"""SPEC_05 no-sum 28-d — single-fold per-variant metrics.

Fold-0 only (per user request "fold 1개만"), no sum-consistency post-hoc.
Variants:
  xgb_24d   : m3 24-d, per-channel XGB
  xgb_28d   : m3 24-d + d25 + d26 + d27 + d28 (=28-d), per-channel XGB
  ridge_24d : m3 24-d, per-channel Ridge (α=1, z-score inputs, m3 baseline convention)
  ridge_28d : m3 24-d + d25 + d26 + d27 + d28 (=28-d), per-channel Ridge (α=1, z-score)

Each run writes one JSON with per-channel + barrier NMAE / RMSE, plus test/train counts.
Aggregation + plotting is a separate step (see plot_comparison.py).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path(os.environ.get("BUNDLE_PT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
SPLIT_ROOT = Path(os.environ.get("SPLIT_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9"))
D25_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_d25_refR.parquet"
D26_28_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet"
OUT_RES = REPO / "spec/spec05_d25_sum/results/no_sum_28d"
OUT_RES.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
SEED = 42


def nmae(yt, yp):
    mad = np.mean(np.abs(yt - yt.mean()))
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss / (tot + 1e-12))


def load_bundle_and_fold(fold_idx: int):
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy().astype(np.float64)
    Y = b["labels"].numpy().astype(np.float64)
    r2i = {r: i for i, r in enumerate(rids)}
    fd = SPLIT_ROOT / f"fold{fold_idx}"
    te = json.load(open(fd / "test_rids.json"))
    tf = sorted(fd.glob("size_*.json"),
                key=lambda p: int(p.stem.split("_")[1]), reverse=True)[0]
    tr = json.load(open(tf))
    tr_idx = np.array([r2i[r] for r in tr if r in r2i])
    te_idx = np.array([r2i[r] for r in te if r in r2i])
    return rids, X24, Y, tr_idx, te_idx


def attach_col(rids, parquet_path, col):
    df = pd.read_parquet(parquet_path).set_index("reaction_id")
    vals, ok = [], []
    for r in rids:
        if r in df.index and bool(df.loc[r, "scf_ok"]):
            v = df.loc[r, col]
            if pd.isna(v):
                vals.append(0.0); ok.append(False)
            else:
                vals.append(float(v)); ok.append(True)
        else:
            vals.append(0.0); ok.append(False)
    return np.array(vals, dtype=np.float64), np.array(ok, dtype=bool)


def build_X(variant: str, rids, X24):
    if variant.endswith("_24d"):
        X = X24
        ok = np.ones(len(rids), dtype=bool)
        return X, ok
    # 28d: 24 base + d25 + d26 + d27 + d28
    d25, ok25 = attach_col(rids, D25_PQ, "d25")
    d26, ok26 = attach_col(rids, D26_28_PQ, "d26")
    d27, ok27 = attach_col(rids, D26_28_PQ, "d27")
    d28, ok28 = attach_col(rids, D26_28_PQ, "d28")
    X = np.hstack([X24, d25[:, None], d26[:, None], d27[:, None], d28[:, None]])
    ok = ok25 & ok26 & ok27 & ok28
    return X, ok


def xgb_predict(X_tr, y_tr, X_te, seed):
    est = XGBRegressor(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        random_state=seed, n_jobs=4, objective="reg:squarederror",
        verbosity=0,
    )
    est.fit(X_tr, y_tr)
    return est.predict(X_te)


def ridge_fit_predict(X_tr, Y_tr, X_te, alpha=1.0):
    """Per-channel ridge on z-scored features, matching src/eda_asm/asr_v1/baseline_physics.LinearBaseline."""
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xn_tr = (X_tr - mean) / std
    Xn_te = (X_te - mean) / std
    A = np.concatenate([Xn_tr, np.ones((Xn_tr.shape[0], 1))], axis=1)
    Ate = np.concatenate([Xn_te, np.ones((Xn_te.shape[0], 1))], axis=1)
    n_feat = A.shape[1]
    reg = alpha * np.eye(n_feat); reg[-1, -1] = 0.0
    W = np.linalg.solve(A.T @ A + reg, A.T @ Y_tr)   # (d+1, 5)
    return Ate @ W


def per_channel_xgb(X_tr, Y_tr, X_te):
    P = np.zeros((len(X_te), 5))
    for c in range(5):
        P[:, c] = xgb_predict(X_tr, Y_tr[:, c], X_te, seed=SEED + c)
    return P


def evaluate(variant: str, fold_idx: int):
    rids, X24, Y, tr, te = load_bundle_and_fold(fold_idx)
    X, ok = build_X(variant, rids, X24)
    tr = tr[ok[tr]]; te = te[ok[te]]
    print(f"[{variant}] fold{fold_idx}  D={X.shape[1]}  n_train={len(tr)}  n_test={len(te)}", flush=True)
    if variant.startswith("xgb"):
        preds = per_channel_xgb(X[tr], Y[tr], X[te])
    elif variant.startswith("ridge"):
        preds = ridge_fit_predict(X[tr], Y[tr], X[te], alpha=1.0)
    else:
        raise SystemExit(f"unknown variant {variant}")
    yt = Y[te]; yp = preds
    metrics = {"variant": variant, "fold": fold_idx,
               "n_train": int(len(tr)), "n_test": int(len(te)),
               "D": int(X.shape[1]), "channels": {}}
    for c_i, ch in enumerate(CHANNELS):
        metrics["channels"][ch] = {"NMAE": nmae(yt[:, c_i], yp[:, c_i]),
                                    "RMSE": rmse(yt[:, c_i], yp[:, c_i]),
                                    "R2":   r2(yt[:, c_i], yp[:, c_i])}
    bt = yt.sum(axis=1); bp = yp.sum(axis=1)
    metrics["barrier"] = {"NMAE": nmae(bt, bp), "RMSE": rmse(bt, bp),
                          "R2": r2(bt, bp)}
    out = OUT_RES / f"{variant}_fold{fold_idx}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}", flush=True)
    for ch in CHANNELS + ["barrier"]:
        m = metrics["channels"][ch] if ch != "barrier" else metrics["barrier"]
        print(f"  {ch:<8s}  NMAE={m['NMAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["xgb_24d", "xgb_28d", "ridge_24d", "ridge_28d"])
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    evaluate(args.variant, args.fold)


if __name__ == "__main__":
    main()
