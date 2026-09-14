#!/usr/bin/env python3
"""train_ml_single.py — process ONE target for parallel ML array.

Usage:
    python train_ml_single.py <target_idx>   # 0..10

Runs Protocol A (Ridge/KRR/SVR/XGB) + Protocol B (Linear/Ridge/RF/GBR/XGB) on
both feature sets (E5, ESPLEY54) for a single target, writes per-target JSON
+ CSV + predictions to <ROOT>/ml_targets/. aggregate_ml.py merges them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
FEAT_PATH = ROOT / "xtb_features.parquet"
TARGETS_DIR = ROOT / "ml_targets"
TARGETS_DIR.mkdir(parents=True, exist_ok=True)
N_JOBS = int(os.environ.get("ESPLEY_NJOBS", "8"))

DIST11 = ["dist_R_dip_ab", "dist_R_dip_bc", "dist_R_dip_ac", "dist_R_dph_ab",
          "dist_TS_dip_ab", "dist_TS_dip_bc", "dist_TS_dip_ac", "dist_TS_dph_ab",
          "dist_TS_form_ad", "dist_TS_form_be", "dist_TS_diag_ae"]
MULLIKEN15 = ([f"mulliken_R_dip_{k}" for k in (0, 1, 2)]
              + [f"mulliken_R_dph_{k}" for k in (0, 1)]
              + [f"mulliken_TSdip_{k}" for k in (0, 1, 2)]
              + [f"mulliken_TSdph_{k}" for k in (0, 1)]
              + [f"mulliken_TS_{k}" for k in (0, 1, 2, 3, 4)])
APTSURR15 = ([f"aptsurr_R_dip_{k}" for k in (0, 1, 2)]
             + [f"aptsurr_R_dph_{k}" for k in (0, 1)]
             + [f"aptsurr_TSdip_{k}" for k in (0, 1, 2)]
             + [f"aptsurr_TSdph_{k}" for k in (0, 1)]
             + [f"aptsurr_TS_{k}" for k in (0, 1, 2, 3, 4)])
D_STRUCT41 = DIST11 + MULLIKEN15 + APTSURR15
E5 = ["xtb_e_barrier_kcal", "xtb_dist_dipole_kcal", "xtb_dist_dipolarophile_kcal",
      "xtb_sum_distortion_kcal", "xtb_interaction_kcal"]
CHAN8 = ["ch_strain_1", "ch_strain_2", "ch_elst", "ch_Pauli", "ch_oi",
         "ch_disp", "ch_cpcm", "ch_cds"]
ESPLEY54 = D_STRUCT41 + E5 + CHAN8
FEATURE_SETS = {"E5": E5, "ESPLEY54": ESPLEY54}

TARGETS = ["dft_barrier_kcal", "dft_d1_kcal", "dft_d2_kcal", "dft_eint_spe_kcal", "dft_e_bond_kcal",
           "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft", "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft"]
PRE_ML = {"dft_barrier_kcal": "xtb_e_barrier_kcal", "dft_d1_kcal": "xtb_dist_dipole_kcal",
          "dft_d2_kcal": "xtb_dist_dipolarophile_kcal", "dft_eint_spe_kcal": "xtb_interaction_kcal",
          "dft_e_bond_kcal": "xtb_interaction_kcal"}
SEEDS = [22, 23, 14, 1, 2]
TUNE_SEED = 23

GRIDS = {
    "Ridge": (Ridge(), {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}),
    "KRR_rbf": (KernelRidge(kernel="rbf"),
                {"kernelridge__alpha": [1e-3, 1e-2, 1e-1, 1],
                 "kernelridge__gamma": [1e-3, 1e-2, 1e-1, 1]}),
    "SVR_rbf": (SVR(kernel="rbf"),
                {"svr__C": [1, 10, 100, 1000],
                 "svr__gamma": ["scale", 1e-2, 1e-1],
                 "svr__epsilon": [0.05, 0.1, 0.5]}),
    "XGB": (XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method="hist"),
            {"xgbregressor__n_estimators": [200, 500],
             "xgbregressor__max_depth": [3, 5, 7],
             "xgbregressor__learning_rate": [0.03, 0.1]}),
}


def mets(y, p):
    r = float(np.corrcoef(y, p)[0, 1]) if (np.std(y) > 0 and np.std(p) > 0) else float("nan")
    return dict(mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)), pearson_r=r)


def split_80_10_10(n, seed):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.20, random_state=seed)
    _, te = train_test_split(rest, test_size=0.50, random_state=seed)
    return tr, te


def protocol_A(df, feats, tgt, preds_store):
    X, y = df[feats].values, df[tgt].values
    out = {}
    tr23, _ = split_80_10_10(len(df), TUNE_SEED)
    for name, (est, grid) in GRIDS.items():
        pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), est)
        gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=TUNE_SEED),
                          scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise")
        gs.fit(X[tr23], y[tr23])
        best = gs.best_params_
        tr_m, te_m, ranges = [], [], []
        for seed in SEEDS:
            tr, te = split_80_10_10(len(df), seed)
            mdl = make_pipeline(StandardScaler(with_mean=True, with_std=True), est).set_params(**best).fit(X[tr], y[tr])
            tr_m.append(mets(y[tr], mdl.predict(X[tr])))
            te_m.append(mets(y[te], mdl.predict(X[te])))
            ranges.append(float(y[te].max() - y[te].min()))
            preds_store.append(pd.DataFrame({"rxn_id": df.rxn_id.values[te], "seed": seed, "model": name,
                                             "target": tgt, "feature_set": len(feats),
                                             "y": y[te], "yhat": mdl.predict(X[te])}))
        test_mae = float(np.mean([m["mae"] for m in te_m]))
        out[name] = dict(best_params=best,
                         train_mae=float(np.mean([m["mae"] for m in tr_m])),
                         test_mae=test_mae, test_mae_sd=float(np.std([m["mae"] for m in te_m])),
                         test_rmse=float(np.mean([m["rmse"] for m in te_m])),
                         test_r2=float(np.mean([m["r2"] for m in te_m])),
                         test_pearson_r=float(np.mean([m["pearson_r"] for m in te_m])),
                         test_range=float(np.mean(ranges)),
                         test_mae_pct_range=100 * test_mae / max(float(np.mean(ranges)), 1e-9))
    return out


def protocol_B(df, feats, tgt):
    X, y = df[feats].values, df[tgt].values
    kf = KFold(5, shuffle=True, random_state=42)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge_alpha1": Ridge(alpha=1.0),
        "RandomForest_200": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=N_JOBS),
        "GBR_200_lr0.05": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        "XGB_default": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                    random_state=42, n_jobs=N_JOBS, verbosity=0, tree_method="hist"),
    }
    out = {}
    for name, est in models.items():
        oof = np.zeros_like(y)
        for tr, va in kf.split(X):
            oof[va] = make_pipeline(StandardScaler(with_mean=True, with_std=True), est).fit(X[tr], y[tr]).predict(X[va])
        out[name] = dict(pooled_oof=mets(y, oof))
    return out


def main():
    tidx = int(sys.argv[1])
    tgt = TARGETS[tidx]
    print(f"[target {tidx}/{len(TARGETS)}] {tgt}", flush=True)

    df = pd.read_parquet(FEAT_PATH)
    all_cols = sorted(set(sum(FEATURE_SETS.values(), [])) | set(TARGETS))
    ok = (df["xtb_status"] == "ok") & df[all_cols].notna().all(axis=1)
    ml = df[ok].reset_index(drop=True)
    print(f"ML-ready rows: {len(ml)}", flush=True)

    pre_ml = None
    if tgt in PRE_ML:
        feat = PRE_ML[tgt]
        d = ml[feat] - ml[tgt]
        pre_ml = dict(feature=feat, mae=float(d.abs().mean()), bias=float(d.mean()),
                      pearson_r=float(np.corrcoef(ml[feat], ml[tgt])[0, 1]))
        print(f"pre-ML {tgt} <- {feat} MAE {pre_ml['mae']:.2f}  bias {pre_ml['bias']:+.2f}", flush=True)

    result = dict(target=tgt, target_idx=tidx, n_rows_ml=len(ml), pre_ml=pre_ml,
                  protocol_A={}, protocol_B={})
    preds = []
    for fs_name, feats in FEATURE_SETS.items():
        print(f"\n=== [{fs_name}] target={tgt} ({len(feats)} features) ===", flush=True)
        A = protocol_A(ml, feats, tgt, preds)
        B = protocol_B(ml, feats, tgt)
        result["protocol_A"][fs_name] = A
        result["protocol_B"][fs_name] = B
        pre = (pre_ml or {}).get("mae", float("nan"))
        for mname, r in A.items():
            print(f"  A  {mname:8s} pre-ML {pre:6.2f}  test MAE {r['test_mae']:6.2f} ± {r['test_mae_sd']:.2f}"
                  f"  ({r['test_mae_pct_range']:.1f}% of range {r['test_range']:.1f})  r2 {r['test_r2']:+.3f}", flush=True)
        for mname, r in B.items():
            print(f"  B  {mname:20s}                 test MAE {r['pooled_oof']['mae']:6.2f}                 r2 {r['pooled_oof']['r2']:+.3f}", flush=True)

    out_json = TARGETS_DIR / f"target_{tidx:02d}_{tgt}.json"
    out_json.write_text(json.dumps(result, indent=2))
    if preds:
        pd.concat(preds, ignore_index=True).to_parquet(TARGETS_DIR / f"preds_{tidx:02d}_{tgt}.parquet", index=False)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
