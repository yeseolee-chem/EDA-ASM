#!/usr/bin/env python3
"""train_ml_single.py — process ONE target for parallel ML array.

Usage:
    python train_ml_single.py <target_idx>   # 0..11

Runs Protocol A (Ridge/KRR/SVR/XGB with per-seed GridSearchCV; y standardized
via TransformedTargetRegressor) + Protocol B (Linear/Ridge/RF/GBR/XGB, pooled
OOF) on three feature sets (ESPLEY46 / ESPLEY54 / ESPLEY73) for a single
target. Writes per-target JSON + preds parquet to <ROOT>/ml_targets/.
aggregate_ml.py merges them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
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
VALENCE15 = ([f"wbo_valence_R_dip_{k}" for k in (0, 1, 2)]
             + [f"wbo_valence_R_dph_{k}" for k in (0, 1)]
             + [f"wbo_valence_TSdip_{k}" for k in (0, 1, 2)]
             + [f"wbo_valence_TSdph_{k}" for k in (0, 1)]
             + [f"wbo_valence_TS_{k}" for k in (0, 1, 2, 3, 4)])
D_STRUCT41 = DIST11 + MULLIKEN15 + VALENCE15
E5 = ["xtb_e_barrier_kcal", "xtb_dist_dipole_kcal", "xtb_dist_dipolarophile_kcal",
      "xtb_sum_distortion_kcal", "xtb_interaction_kcal"]
CHAN8 = ["b_strain_1", "b_strain_2", "b_elst", "b_pauli", "b_oi", "b_disp", "b_cpcm", "b_cds"]
AUX19 = (["b_elst_scc", "b_disp_d4", "b_axc", "b_ct",
          "gap_ts", "gap_dip", "gap_dph", "mu_ts", "mu_dip", "mu_dph", "dmu_complexation",
          "is_charged"]                                         # N: is_charged added
         + [f"dsasa_{el}" for el in ("H", "C", "N", "O", "F", "Cl", "Br")])
ESPLEY46 = D_STRUCT41 + E5                       # ablation: 채널 블록 없음
ESPLEY54 = D_STRUCT41 + E5 + CHAN8               # 옛 55의 대응 (q_barrier만 제외)
ESPLEY73 = ESPLEY54 + AUX19                      # was ESPLEY72; +is_charged
FEATURE_SETS = {"ESPLEY46": ESPLEY46, "ESPLEY54": ESPLEY54, "ESPLEY73": ESPLEY73}

TARGETS = ["dft_barrier_kcal", "dft_d1_kcal", "dft_d2_kcal", "dft_eint_spe_kcal", "dft_e_bond_kcal",
           "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft", "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft",
           "dft_barrier_eda"]                    # B2: 8-channel closed total
PRE_ML = {"dft_barrier_kcal": "xtb_e_barrier_kcal", "dft_d1_kcal": "xtb_dist_dipole_kcal",
          "dft_d2_kcal": "xtb_dist_dipolarophile_kcal", "dft_eint_spe_kcal": "xtb_interaction_kcal",
          "dft_e_bond_kcal": "xtb_interaction_kcal",
          "dft_elst_dft": "b_elst", "dft_pauli_dft": "b_pauli", "dft_oi_dft": "b_oi",
          "dft_disp_dft": "b_disp", "dft_cpcm_dft": "b_cpcm", "dft_cds_dft": "b_cds",
          "dft_barrier_eda": "xtb_e_barrier_kcal"}
SEEDS = [22, 23, 14, 1, 2]
TUNE_SEED = 23                                    # kept only for record; N3 nested CV tunes per seed

# N1 — grids expanded down to the corners we hit in the previous run.
# N2 — every estimator wrapped in TransformedTargetRegressor(StandardScaler) → prefix "regressor__"
GRIDS = {
    "Ridge": (Ridge(),
              {"regressor__ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}),
    "KRR_rbf": (KernelRidge(kernel="rbf"),
                {"regressor__kernelridge__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                 "regressor__kernelridge__gamma": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1]}),
    "SVR_rbf": (SVR(kernel="rbf"),
                {"regressor__svr__C": [1, 10, 100, 1000],
                 "regressor__svr__gamma": ["scale", 1e-3, 1e-2, 1e-1],
                 "regressor__svr__epsilon": [0.05, 0.1, 0.5]}),
    "XGB": (XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method="hist"),
            {"regressor__xgbregressor__n_estimators": [200, 500],
             "regressor__xgbregressor__max_depth": [3, 5, 7],
             "regressor__xgbregressor__learning_rate": [0.03, 0.1]}),
}


def _make_pipe(est):
    """N2: y-standardization via TransformedTargetRegressor wrapping (StandardScaler(x) → est)."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(with_mean=True, with_std=True), est),
        transformer=StandardScaler(with_mean=True, with_std=True))


def _edge_hit(param_name, value, grid_values):
    """True if numeric value equals min or max of grid_values (ignores non-numeric like 'scale')."""
    numeric = [v for v in grid_values if isinstance(v, (int, float))]
    if not numeric or not isinstance(value, (int, float)):
        return False
    return value == min(numeric) or value == max(numeric)


def mets(y, p):
    r = float(np.corrcoef(y, p)[0, 1]) if (np.std(y) > 0 and np.std(p) > 0) else float("nan")
    mae = float(mean_absolute_error(y, p))
    mad = float(np.mean(np.abs(y - np.mean(y))))          # mean absolute deviation of the target
    return dict(mae=mae, nmae=mae / mad if mad > 0 else float("nan"),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)), pearson_r=r)


def split_80_10_10(n, seed):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.20, random_state=seed)
    _, te = train_test_split(rest, test_size=0.50, random_state=seed)
    return tr, te


def protocol_A(df, feats, tgt, preds_store):
    """N3: nested CV — GridSearchCV is run on each seed's own train fold (no shared-tuning leak)."""
    X, y = df[feats].values, df[tgt].values
    out = {}
    for name, (est, grid) in GRIDS.items():
        tr_m, te_m, ranges = [], [], []
        per_seed_best, per_seed_edge = [], []
        for seed in SEEDS:
            tr, te = split_80_10_10(len(df), seed)
            gs = GridSearchCV(_make_pipe(est), grid,
                              cv=KFold(5, shuffle=True, random_state=seed),
                              scoring="neg_mean_absolute_error",
                              n_jobs=N_JOBS, error_score="raise")
            gs.fit(X[tr], y[tr])
            best = gs.best_params_
            mdl = gs.best_estimator_
            tr_m.append(mets(y[tr], mdl.predict(X[tr])))
            te_m.append(mets(y[te], mdl.predict(X[te])))
            ranges.append(float(y[te].max() - y[te].min()))
            preds_store.append(pd.DataFrame({"rxn_id": df.rxn_id.values[te], "seed": seed, "model": name,
                                             "target": tgt, "feature_set": len(feats),
                                             "y": y[te], "yhat": mdl.predict(X[te])}))
            per_seed_best.append(best)
            per_seed_edge.append({k: _edge_hit(k, v, grid[k]) for k, v in best.items()})
        test_mae = float(np.mean([m["mae"] for m in te_m]))
        edge_hits_total = sum(sum(1 for hit in d.values() if hit) for d in per_seed_edge)
        out[name] = dict(per_seed_best_params=per_seed_best,
                         per_seed_edge_hits=per_seed_edge,
                         edge_hits_total=edge_hits_total,
                         train_mae=float(np.mean([m["mae"] for m in tr_m])),
                         train_nmae=float(np.mean([m["nmae"] for m in tr_m])),
                         test_mae=test_mae, test_mae_sd=float(np.std([m["mae"] for m in te_m])),
                         test_nmae=float(np.mean([m["nmae"] for m in te_m])),
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
    # N (hygiene): drop d1<0, d2<0, d2>50 — 라벨 위생 필터
    hygiene = (df["dft_d1_kcal"] >= 0) & (df["dft_d2_kcal"] >= 0) & (df["dft_d2_kcal"] <= 50)
    dropped = int((ok & ~hygiene).sum())
    ml = df[ok & hygiene].reset_index(drop=True)
    print(f"ML-ready rows: {len(ml)}  (hygiene dropped {dropped} rows: d1<0 / d2<0 / d2>50)", flush=True)

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
