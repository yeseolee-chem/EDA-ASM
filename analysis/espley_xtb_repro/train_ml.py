#!/usr/bin/env python3
"""train_ml.py — Espley 2024 protocol with GFN2-xTB(ALPB) features on the Coley B3LYP labels.

Protocol A (Espley-faithful; ESI §4.2–4.3 of DOI 10.1039/D4DD00224E)
    * random split 80/10/10 (train/val/test) per seed; seeds = [22, 23, 14, 1, 2]
    * models: Ridge, KRR(RBF), SVR(RBF), each after StandardScaler
    * hyperparameters: GridSearchCV(5-fold) on the seed-23 training split, then reused
    * report per target: pre-ML MAE (xTB feature of the same quantity vs DFT target),
      train MAE, test MAE (mean ± sd over 5 seeds), test target range, MAE as % of range
Protocol B (secondary): 5-fold KFold (seed 42), pooled out-of-fold metrics, models
    LinearRegression / Ridge / RandomForest / GradientBoosting (v1 behaviour, kept).

Feature sets
    E5      : the 5 xTB energies (barrier, dist_dipole, dist_dipolarophile, sum_dist, interaction)
    ESPLEY  : E5 + Mulliken charges of the 4 reacting atoms (TS / distorted / GS)
              + reacting distances (0, 1, diff)  — the analogue of Espley Table S2
Targets
    dft_barrier_kcal, dft_d1_kcal, dft_d2_kcal, dft_eint_spe_kcal (ASM-consistent),
    dft_e_bond_kcal (EDA Bond), dft_{elst,pauli,oi,disp,cpcm,cds}_dft
"""
from __future__ import annotations

import json
import os
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

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
FEAT_PATH = ROOT / "xtb_features.parquet"
OUT_REPORT = ROOT / "ml_report.json"
OUT_TABLE = ROOT / "ml_table_espley.csv"
OUT_PREDS = ROOT / "predictions.parquet"
N_JOBS = int(os.environ.get("ESPLEY_NJOBS", "8"))

E5 = ["xtb_e_barrier_kcal", "xtb_dist_dipole_kcal", "xtb_dist_dipolarophile_kcal",
      "xtb_sum_distortion_kcal", "xtb_interaction_kcal"]
CHARGES = [f"q_{s}_{r}_{n}" for s in ("ts", "dist", "gs") for r in ("dip", "dph") for n in (1, 2)] + ["q_ts_dipole_sum"]
DIST = ["reacting_distance_0_ts", "reacting_distance_1_ts", "reacting_distance_diff_ts"]
FEATURE_SETS = {"E5": E5, "ESPLEY": E5 + CHARGES + DIST}
TARGETS = ["dft_barrier_kcal", "dft_d1_kcal", "dft_d2_kcal", "dft_eint_spe_kcal", "dft_e_bond_kcal",
           "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft", "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft"]
# xTB feature that is the same physical quantity as the target (pre-ML comparison)
PRE_ML = {"dft_barrier_kcal": "xtb_e_barrier_kcal", "dft_d1_kcal": "xtb_dist_dipole_kcal",
          "dft_d2_kcal": "xtb_dist_dipolarophile_kcal", "dft_eint_spe_kcal": "xtb_interaction_kcal",
          "dft_e_bond_kcal": "xtb_interaction_kcal"}
SEEDS = [22, 23, 14, 1, 2]
TUNE_SEED = 23

GRIDS = {
    "Ridge": (Ridge(), {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}),
    "KRR_rbf": (KernelRidge(kernel="rbf"), {"kernelridge__alpha": [1e-3, 1e-2, 1e-1, 1],
                                            "kernelridge__gamma": [1e-3, 1e-2, 1e-1, 1]}),
    "SVR_rbf": (SVR(kernel="rbf"), {"svr__C": [1, 10, 100, 1000], "svr__gamma": ["scale", 1e-2, 1e-1],
                                    "svr__epsilon": [0.05, 0.1, 0.5]}),
}


def mets(y, p):
    r = float(np.corrcoef(y, p)[0, 1]) if (np.std(y) > 0 and np.std(p) > 0) else float("nan")
    return dict(mae=float(mean_absolute_error(y, p)), rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)), pearson_r=r)


def split_80_10_10(n, seed):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.20, random_state=seed)
    val, te = train_test_split(rest, test_size=0.50, random_state=seed)
    return tr, val, te


def protocol_A(df, feats, tgt, preds_store):
    X, y = df[feats].values, df[tgt].values
    out = {}
    tr23, _, _ = split_80_10_10(len(df), TUNE_SEED)
    for name, (est, grid) in GRIDS.items():
        pipe = make_pipeline(StandardScaler(), est)
        gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=TUNE_SEED),
                          scoring="neg_mean_absolute_error", n_jobs=N_JOBS)
        gs.fit(X[tr23], y[tr23])
        best = gs.best_params_
        tr_m, te_m, ranges = [], [], []
        for seed in SEEDS:
            tr, _, te = split_80_10_10(len(df), seed)
            mdl = make_pipeline(StandardScaler(), est).set_params(**best).fit(X[tr], y[tr])
            tr_m.append(mets(y[tr], mdl.predict(X[tr]))); te_m.append(mets(y[te], mdl.predict(X[te])))
            ranges.append(float(y[te].max() - y[te].min()))
            preds_store.append(pd.DataFrame({"rxn_id": df.rxn_id.values[te], "seed": seed, "model": name,
                                             "target": tgt, "y": y[te], "yhat": mdl.predict(X[te])}))
        test_mae = float(np.mean([m["mae"] for m in te_m]))
        out[name] = dict(best_params=best,
                         train_mae=float(np.mean([m["mae"] for m in tr_m])),
                         test_mae=test_mae, test_mae_sd=float(np.std([m["mae"] for m in te_m])),
                         test_rmse=float(np.mean([m["rmse"] for m in te_m])),
                         test_r2=float(np.mean([m["r2"] for m in te_m])),
                         test_pearson_r=float(np.mean([m["pearson_r"] for m in te_m])),
                         test_range=float(np.mean(ranges)), test_mae_pct_range=100 * test_mae / float(np.mean(ranges)))
    return out


def protocol_B(df, feats, tgt):
    X, y = df[feats].values, df[tgt].values
    kf = KFold(5, shuffle=True, random_state=42)
    models = {"LinearRegression": LinearRegression(), "Ridge_alpha1": Ridge(alpha=1.0),
              "RandomForest_200": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=N_JOBS),
              "GBR_200_lr0.05": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)}
    out = {}
    for name, est in models.items():
        oof = np.zeros_like(y)
        for tr, va in kf.split(X):
            oof[va] = make_pipeline(StandardScaler(), est).fit(X[tr], y[tr]).predict(X[va])
        out[name] = dict(pooled_oof=mets(y, oof))
    return out


def main():
    df = pd.read_parquet(FEAT_PATH)
    print(f"loaded {len(df)} rows; xtb_status: {dict(df['xtb_status'].value_counts())}")
    all_cols = sorted(set(sum(FEATURE_SETS.values(), [])) | set(TARGETS))
    ok = (df["xtb_status"] == "ok") & df[all_cols].notna().all(axis=1)
    ml = df[ok].reset_index(drop=True)
    print(f"ML-ready rows: {len(ml)}  (flags kept, not filtered: foreign={int(ml.flag_foreign_bond.sum())} "
          f"neg_strain={int(ml.flag_negative_strain.sum())})")

    report = dict(n_rows_input=int(len(df)), n_rows_ml=int(len(ml)), seeds=SEEDS, tune_seed=TUNE_SEED,
                  feature_sets=FEATURE_SETS, targets=TARGETS, pre_ml={}, protocol_A={}, protocol_B={})
    for tgt, feat in PRE_ML.items():
        d = ml[feat] - ml[tgt]
        report["pre_ml"][tgt] = dict(feature=feat, mae=float(d.abs().mean()), bias=float(d.mean()),
                                     pearson_r=float(np.corrcoef(ml[feat], ml[tgt])[0, 1]))
        print(f"pre-ML {tgt:20s} <- {feat:28s} MAE {d.abs().mean():6.2f}  bias {d.mean():+6.2f}")

    preds, rows = [], []
    for fs_name, feats in FEATURE_SETS.items():
        report["protocol_A"][fs_name], report["protocol_B"][fs_name] = {}, {}
        for tgt in TARGETS:
            A = protocol_A(ml, feats, tgt, preds)
            B = protocol_B(ml, feats, tgt)
            report["protocol_A"][fs_name][tgt], report["protocol_B"][fs_name][tgt] = A, B
            pre = report["pre_ml"].get(tgt, {}).get("mae", float("nan"))
            for mname, r in A.items():
                rows.append(dict(feature_set=fs_name, target=tgt, model=mname, pre_ml_mae=pre,
                                 train_mae=r["train_mae"], test_mae=r["test_mae"], test_mae_sd=r["test_mae_sd"],
                                 test_r2=r["test_r2"], test_range=r["test_range"], test_mae_pct_range=r["test_mae_pct_range"]))
                print(f"  [{fs_name:6s}] {tgt:20s} {mname:8s} pre-ML {pre:6.2f}  test MAE {r['test_mae']:6.2f} ± {r['test_mae_sd']:.2f}"
                      f"  ({r['test_mae_pct_range']:.1f}% of range {r['test_range']:.1f})  r2 {r['test_r2']:+.3f}", flush=True)
            for mname, r in B.items():
                rows.append(dict(feature_set=fs_name, target=tgt, model=mname + "_5foldOOF", pre_ml_mae=pre,
                                 train_mae=float("nan"), test_mae=r["pooled_oof"]["mae"], test_mae_sd=float("nan"),
                                 test_r2=r["pooled_oof"]["r2"], test_range=float("nan"), test_mae_pct_range=float("nan")))
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False)
    pd.concat(preds, ignore_index=True).to_parquet(OUT_PREDS, index=False)
    print(f"report -> {OUT_REPORT}\ntable  -> {OUT_TABLE}\npreds  -> {OUT_PREDS}")


if __name__ == "__main__":
    main()
