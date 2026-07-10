"""Shared baseline module for SPEC_02 A/B/C ablation.

Provides:
  - fit_ridge(X_train, Y_train) -> per-channel ridge model
  - fit_xgb(X_train, Y_train)   -> per-channel XGB model
  - cross_fit_oof(kind, X_train, Y_train, K=5) -> (N_train, 5) OOF preds
  - fit_full(kind, X_train, Y_train), predict_full(kind, model, X)

Cross-fit rule (SPEC gate #3): delta training target = y - b_oof(X); outer val
predictions use b_full(X) trained on all outer train.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
RIDGE_ALPHA = 1.0


def fit_ridge(X, Y):
    sc = StandardScaler().fit(X)
    Xn = sc.transform(X)
    models = [Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(Xn, Y[:, c])
              for c in range(Y.shape[1])]
    return {"scaler": sc, "models": models}


def predict_ridge(model, X):
    Xn = model["scaler"].transform(X)
    return np.column_stack([m.predict(Xn) for m in model["models"]])


def fit_xgb(X, Y):
    models = []
    for c in range(Y.shape[1]):
        m = XGBRegressor(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=5, tree_method="hist",
            random_state=SEED + c, n_jobs=4,
            objective="reg:squarederror", verbosity=0,
        )
        m.fit(X, Y[:, c])
        models.append(m)
    return {"models": models}


def predict_xgb(model, X):
    return np.column_stack([m.predict(X) for m in model["models"]])


BASELINE_FUNCS = {
    "ridge": (fit_ridge, predict_ridge),
    "xgb":   (fit_xgb,   predict_xgb),
}


def cross_fit_oof(baseline_kind, X_train, Y_train, K=5, seed=SEED):
    fit_fn, pred_fn = BASELINE_FUNCS[baseline_kind]
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    oof = np.zeros_like(Y_train, dtype=np.float32)
    for i_tr, i_va in kf.split(X_train):
        m = fit_fn(X_train[i_tr], Y_train[i_tr])
        oof[i_va] = pred_fn(m, X_train[i_va])
    return oof


def fit_full(baseline_kind, X_train, Y_train):
    fit_fn, _ = BASELINE_FUNCS[baseline_kind]
    return fit_fn(X_train, Y_train)


def predict_full(baseline_kind, model, X):
    _, pred_fn = BASELINE_FUNCS[baseline_kind]
    return pred_fn(model, X)
