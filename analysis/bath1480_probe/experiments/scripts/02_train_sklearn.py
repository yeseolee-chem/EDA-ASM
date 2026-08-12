#!/usr/bin/env python3
"""Phase 1/2/3 shared runner: sklearn Ridge/KRR/SVR training with hyperparameter tuning.

Env variables control which experiment:
  PHASE_NAME:   phase1_espley / phase2_armB_tau1e-10 / phase3_armB_tau0.05
  LABELS_PKL:   path to labels_v1.pkl
  FEATURES_PKL: path to features_espley_v1.pkl or features_armB_v1.pkl
  TARGET_COLS:  space-separated list of label columns
  VAR_TAU:      variance threshold (0.05 or 1e-10)
  SEED:         random seed (0..4)

Runs: 3 models × N_targets per seed.
Output: results/{PHASE_NAME}/seed_{SEED}/{model}_results.pkl
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PHASE_NAME = os.environ["PHASE_NAME"]
LABELS_PKL = os.environ["LABELS_PKL"]
FEATURES_PKL = os.environ["FEATURES_PKL"]
TARGET_COLS = os.environ["TARGET_COLS"].split()
VAR_TAU = float(os.environ["VAR_TAU"])
SEED = int(os.environ["SEED"])

OUT_ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results")
OUT_DIR = OUT_ROOT / PHASE_NAME / f"seed_{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_models():
    """3 model pipelines with hyperparameter grids."""
    return {
        "ridge": (Ridge(),
                  {"alpha": np.logspace(-3, 3, 13)}),
        "krr":   (KernelRidge(kernel="rbf"),
                  {"alpha": np.logspace(-3, 3, 7),
                   "gamma": np.logspace(-3, 3, 7)}),
        "svr":   (SVR(kernel="rbf"),
                  {"C": np.logspace(-1, 3, 5),
                   "gamma": np.logspace(-3, 1, 5),
                   "epsilon": [0.01, 0.1, 1.0]}),
    }


def train_one(features_train, y_train, features_test, y_test, model, param_grid, seed):
    pipe = Pipeline([
        ("var", VarianceThreshold(threshold=VAR_TAU)),
        ("scale", StandardScaler()),
        ("est", model),
    ])
    # Prefix grid keys with 'est__'
    grid = {f"est__{k}": v for k, v in param_grid.items()}
    gs = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_absolute_error",
                       n_jobs=1, refit=True)
    gs.fit(features_train, y_train)
    y_pred = gs.predict(features_test)
    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "best_params": gs.best_params_,
        "best_cv_score": -gs.best_score_,
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2": r2_score(y_test, y_pred),
    }


def main():
    print(f"=== PHASE={PHASE_NAME}  SEED={SEED}  VAR_TAU={VAR_TAU} ===")
    print(f"Targets ({len(TARGET_COLS)}): {TARGET_COLS}")

    labels_df = pd.read_pickle(LABELS_PKL)
    feats_df = pd.read_pickle(FEATURES_PKL)

    # merge on reaction_number
    df = labels_df.merge(feats_df, on="reaction_number")
    print(f"Merged: {df.shape}")

    feature_cols = [c for c in feats_df.columns if c != "reaction_number"]
    print(f"Feature cols: {len(feature_cols)}")

    fold_ids = df["fold_id"].unique()
    print(f"Folds: {sorted(fold_ids)}")

    # 5-fold outer CV: for each fold, use it as test, others as train
    models_dict = build_models()

    all_results = {}  # model → target → list of per-fold results
    for model_name, (model_class, param_grid) in models_dict.items():
        print(f"\n=== Model: {model_name} ===")
        model_results = {}
        for target in TARGET_COLS:
            fold_results = []
            for fold in sorted(fold_ids):
                mask_test = df["fold_id"] == fold
                mask_train = ~mask_test

                X_train = df.loc[mask_train, feature_cols].values
                y_train = df.loc[mask_train, target].values
                X_test = df.loc[mask_test, feature_cols].values
                y_test = df.loc[mask_test, target].values
                # drop NaN in labels
                mask_train_valid = ~np.isnan(y_train)
                mask_test_valid = ~np.isnan(y_test)
                X_train, y_train = X_train[mask_train_valid], y_train[mask_train_valid]
                X_test, y_test = X_test[mask_test_valid], y_test[mask_test_valid]

                if len(y_train) < 10 or len(y_test) < 2:
                    print(f"  {target} fold{fold}: SKIP (train={len(y_train)}, test={len(y_test)})")
                    continue

                # fresh model instance
                from sklearn.base import clone
                m = clone(model_class)
                t0 = time.time()
                res = train_one(X_train, y_train, X_test, y_test, m, param_grid, SEED)
                dt = time.time() - t0
                res["fold"] = fold
                res["target"] = target
                res["model"] = model_name
                res["seed"] = SEED
                res["reaction_number_test"] = df.loc[mask_test, "reaction_number"].values[mask_test_valid]
                res["wall_sec"] = dt
                fold_results.append(res)
                print(f"  {target} fold{fold}: MAE={res['test_mae']:.4f}  R²={res['test_r2']:.3f}  ({dt:.0f}s)")
            model_results[target] = fold_results
        all_results[model_name] = model_results

        # Save per-model results
        out_path = OUT_DIR / f"{model_name}_results.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model_results, f)
        print(f"Saved {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for model_name, model_results in all_results.items():
        for target, fold_results in model_results.items():
            if fold_results:
                maes = [r["test_mae"] for r in fold_results]
                print(f"  {model_name:6s} {target:30s}  MAE={np.mean(maes):.4f}±{np.std(maes):.4f}")


if __name__ == "__main__":
    main()
