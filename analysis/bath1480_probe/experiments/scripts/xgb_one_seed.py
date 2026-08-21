#!/usr/bin/env python3
"""XGBoost regressor per (phase, seed). Uses paper's 80/10/10 split + StandardScaler
on X and y. Trains XGB with fixed HPs + early stopping on validation.

Env:
  DATASET: path to phase{2,3,5}_dataset.pkl
  SEED:    random seed for train/test/val split
  OUT_DIR: output dir (mkdir -p'd)
"""
import os, sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATASET = Path(os.environ["DATASET"])
SEED = int(os.environ["SEED"])
OUT_DIR = Path(os.environ["OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed XGB HPs (regression baseline)
XGB_HPS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "random_state": SEED,
    "verbosity": 0,
}


def get_dft_features(df):
    """Match paper's _get_dft_features."""
    remove_targets = [c for c in df.columns if "_dft" in c]
    y = df[remove_targets]
    non_num_meta = ["reaction_number", "elements_frag1", "elements_frag2"]
    X = df.drop(columns=remove_targets)
    # Drop non-numeric identifiers if any
    drop_from_X = [c for c in non_num_meta if c in X.columns]
    X = X.drop(columns=drop_from_X)
    # Ensure only numeric
    X = X.select_dtypes(include=[np.number])
    return X, y


def paper_split(X, y, seed):
    """80/10/10 train/test/val with random_state=seed (matches paper)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
             "y_train": y_train, "y_val": y_val, "y_test": y_test}


def main():
    print(f"=== XGB one seed  DATASET={DATASET.name}  SEED={SEED}  OUT_DIR={OUT_DIR} ===")
    df = pd.read_pickle(DATASET)
    X, y = get_dft_features(df)
    print(f"X: {X.shape}, y: {y.shape}, targets: {list(y.columns)}")

    split = paper_split(X, y, SEED)
    print(f"train/val/test: {split['X_train'].shape[0]}/{split['X_val'].shape[0]}/{split['X_test'].shape[0]}")

    # Standardize X
    sc_X = StandardScaler()
    X_train_s = sc_X.fit_transform(split["X_train"])
    X_val_s = sc_X.transform(split["X_val"])
    X_test_s = sc_X.transform(split["X_test"])

    results = {}
    for target in y.columns:
        yt_train_raw = split["y_train"][target].values.reshape(-1, 1)
        yt_val_raw = split["y_val"][target].values.reshape(-1, 1)
        yt_test_raw = split["y_test"][target].values.reshape(-1, 1)

        # Standardize y (per target, matches paper)
        sc_y = StandardScaler()
        yt_train = sc_y.fit_transform(yt_train_raw).flatten()
        yt_val = sc_y.transform(yt_val_raw).flatten()
        yt_test_scaled = sc_y.transform(yt_test_raw).flatten()

        t0 = time.time()
        model = XGBRegressor(**XGB_HPS)
        model.fit(X_train_s, yt_train, eval_set=[(X_val_s, yt_val)], verbose=False)
        dt = time.time() - t0

        yp_test_scaled = model.predict(X_test_s)
        yp_val_scaled = model.predict(X_val_s)
        # inverse transform to raw scale
        yp_test = sc_y.inverse_transform(yp_test_scaled.reshape(-1, 1)).flatten()
        yp_val = sc_y.inverse_transform(yp_val_scaled.reshape(-1, 1)).flatten()

        yt_test_real = yt_test_raw.flatten()
        yt_val_real = yt_val_raw.flatten()

        mae_test = np.mean(np.abs(yt_test_real - yp_test))
        mae_val = np.mean(np.abs(yt_val_real - yp_val))
        ss_res = np.sum((yt_test_real - yp_test) ** 2)
        ss_tot = np.sum((yt_test_real - yt_test_real.mean()) ** 2)
        r2_test = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        results[target] = {
            "y_test_true": yt_test_real,
            "y_test_pred": yp_test,
            "y_val_true": yt_val_real,
            "y_val_pred": yp_val,
            "test_mae": mae_test,
            "val_mae": mae_val,
            "test_r2": r2_test,
            "wall_sec": dt,
            "best_iteration": model.best_iteration,
        }
        print(f"  {target:<28} MAE_test={mae_test:.4f}  R²={r2_test:.3f}  ({dt:.0f}s, best_iter={model.best_iteration})")

    out_pkl = OUT_DIR / "xgb_results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"saved: {out_pkl}")


if __name__ == "__main__":
    main()
