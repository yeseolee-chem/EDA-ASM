#!/usr/bin/env python3
"""train_ml.py — Espley-style ML with xTB features on Coley labels.

Features (5, from xTB SPE at Coley DFT geoms):
    xtb_e_barrier_kcal
    xtb_dist_dipole_kcal
    xtb_dist_dipolarophile_kcal
    xtb_sum_distortion_kcal
    xtb_interaction_kcal

Targets (DFT B3LYP-D3(BJ)/def2-TZVP CPCM(SMD water) from labels_all):
    dft_barrier_kcal      — primary (analogous to Espley's target)
    dft_d1_kcal           — dipole strain
    dft_d2_kcal           — dipolarophile strain
    dft_e_bond_kcal       — EDA interaction (Bond Energy)
    dft_{elst,pauli,oi,disp,cpcm,cds}_dft  — 6 channels

Models:
    LinearRegression (baseline, matches raw feature quality)
    Ridge (alpha=1.0)
    RandomForestRegressor (n_estimators=200, random_state=42)
    GradientBoostingRegressor (n_estimators=200, learning_rate=0.05, random_state=42)

CV: 5-fold KFold, shuffle=True, random_state=42.
Metrics: MAE, RMSE, r², Pearson r (per fold + aggregated).
Also reports raw feature vs target Pearson r (baseline, no ML).

No preprocessing filter: all rxns with xtb_status='ok' are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

FEAT_PATH = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features.parquet")
OUT_REPORT = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/ml_report.json")
OUT_PREDS = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/predictions.parquet")

FEATURES = [
    "xtb_e_barrier_kcal",
    "xtb_dist_dipole_kcal",
    "xtb_dist_dipolarophile_kcal",
    "xtb_sum_distortion_kcal",
    "xtb_interaction_kcal",
]
TARGETS = [
    "dft_barrier_kcal",
    "dft_d1_kcal", "dft_d2_kcal", "dft_e_bond_kcal",
    "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft",
    "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft",
]


def make_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge_alpha1":     Ridge(alpha=1.0, random_state=42),
        "RandomForest_200": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GBR_200_lr0.05":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    }


def metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2, pearson_r=r)


def main():
    df = pd.read_parquet(FEAT_PATH)
    print(f"loaded {len(df)} rows")
    print(f"xtb_status distribution: {dict(df['xtb_status'].value_counts())}")

    mask = df["xtb_status"] == "ok"
    for col in FEATURES + TARGETS:
        if col in df.columns:
            mask &= df[col].notna()
    ml_df = df[mask].reset_index(drop=True)
    print(f"ML-ready (xtb ok + feats+targets non-NaN): {len(ml_df)}")

    X = ml_df[FEATURES].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline: raw feature vs target Pearson r (no ML)
    baseline = {}
    for tgt in TARGETS:
        y = ml_df[tgt].values
        row = {}
        for f in FEATURES:
            xv = ml_df[f].values
            row[f] = float(np.corrcoef(xv, y)[0, 1]) if (np.std(xv) > 0 and np.std(y) > 0) else float("nan")
        baseline[tgt] = row

    models = make_models()
    report = {"n_rows_ml": int(len(ml_df)),
              "n_rows_input": int(len(df)),
              "features": FEATURES,
              "targets": TARGETS,
              "baseline_pearson_r_feature_vs_target": baseline,
              "models": {}}

    pred_frame = ml_df[["rxn_id"] + FEATURES + TARGETS].copy()

    for tgt in TARGETS:
        y = ml_df[tgt].values
        report["models"][tgt] = {}
        for mname, mmodel in models.items():
            fold_metrics = []
            preds_full = np.zeros_like(y, dtype=float)
            for fold, (tr, va) in enumerate(kf.split(X)):
                mdl = make_models()[mname]
                mdl.fit(X[tr], y[tr])
                yp = mdl.predict(X[va])
                fold_metrics.append(metrics(y[va], yp))
                preds_full[va] = yp
            agg = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
            agg_sd = {k + "_sd": float(np.std([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
            report["models"][tgt][mname] = {**agg, **agg_sd}
            col = f"pred_{tgt}__{mname}"
            pred_frame[col] = preds_full
            print(f"  {tgt:20s} {mname:20s}  MAE {agg['mae']:6.3f}  RMSE {agg['rmse']:6.3f}  r2 {agg['r2']:+.3f}  r {agg['pearson_r']:+.3f}")

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    pred_frame.to_parquet(OUT_PREDS, index=False)
    print(f"\nreport   -> {OUT_REPORT}")
    print(f"preds    -> {OUT_PREDS}")


if __name__ == "__main__":
    main()
