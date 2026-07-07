"""Ridge / XGB per-channel baselines + inner K'=5 cross-fit helper.

Interfaces:
  fit_ridge(X_train, Y_train) -> ChannelModel
  fit_xgb(X_train, Y_train)   -> ChannelModel
  ChannelModel.predict(X)     -> (N, 5)

  cross_fit_oof(model_fn, X, Y, k=5, seed=0) -> b_oof (N, 5)

Notes
-----
- Both baselines are per-channel; predict returns (N, 5).
- Ridge applies z-score(X) fitted on the training slice, then a per-channel
  intercept-augmented ridge with α = 1 (matches the historical
  LinearBaseline). We use a NumPy solver so behaviour is deterministic and
  torch-free.
- XGB uses the default hyperparameters specified in the SPEC:
      n_estimators=800, max_depth=4, learning_rate=0.03,
      subsample=0.8, colsample_bytree=0.8,
      reg_lambda=1.0, min_child_weight=5
  early stopping = False (fixed budget; no inner val leakage into the fit).
- Deterministic behaviour: numpy + torch + xgboost seeds all set via `seed`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


N_CHANNELS = 5


# ============================================================
# Ridge (numpy) — identical convention to eda_asm.LinearBaseline
# ============================================================


@dataclass
class RidgeChannelModel:
    W: np.ndarray          # (d_in + 1, N_CHANNELS)
    mean: np.ndarray       # (d_in,)
    std: np.ndarray        # (d_in,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        Xa = np.concatenate([Xn, np.ones((Xn.shape[0], 1))], axis=1)
        return (Xa @ self.W).astype(np.float32)


def fit_ridge(X_train: np.ndarray, Y_train: np.ndarray,
              alpha: float = 1.0, **_) -> RidgeChannelModel:
    D = np.asarray(X_train, dtype=np.float64)
    Y = np.asarray(Y_train, dtype=np.float64)
    mean = D.mean(axis=0)
    std = D.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Dn = (D - mean) / std
    Xa = np.concatenate([Dn, np.ones((Dn.shape[0], 1))], axis=1)
    n_feat = Xa.shape[1]
    reg = alpha * np.eye(n_feat)
    reg[-1, -1] = 0.0        # do not regularise the intercept
    W = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Y)
    return RidgeChannelModel(W=W, mean=mean, std=std)


# ============================================================
# XGB — per-channel MultiOutput
# ============================================================


@dataclass
class XGBChannelModel:
    models: Sequence   # length N_CHANNELS

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float32)
        out = np.zeros((Xa.shape[0], N_CHANNELS), dtype=np.float32)
        for c, m in enumerate(self.models):
            out[:, c] = m.predict(Xa)
        return out


def fit_xgb(X_train: np.ndarray, Y_train: np.ndarray, seed: int = 42,
            **_) -> XGBChannelModel:
    from xgboost import XGBRegressor
    Xa = np.asarray(X_train, dtype=np.float32)
    Ya = np.asarray(Y_train, dtype=np.float32)
    models = []
    for c in range(N_CHANNELS):
        m = XGBRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=5,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed + c,
            n_jobs=1,
            verbosity=0,
        )
        m.fit(Xa, Ya[:, c])
        models.append(m)
    return XGBChannelModel(models=models)


# ============================================================
# Inner K'=5 out-of-fold predictor stack
# ============================================================


def cross_fit_oof(
    model_fn: Callable[[np.ndarray, np.ndarray], object],
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    seed: int = 0,
    fit_kwargs: dict | None = None,
) -> np.ndarray:
    """K-fold cross-fit OOF baseline over rows of X, Y.

    Returns an (N, 5) array where the value at row i is the prediction of a
    baseline trained on the other K-1 inner-folds (i.e. i was in the held-out
    inner fold). This is the anti-leakage baseline used as the δ training
    target.

    We use sklearn.KFold (random shuffle, seeded) rather than stratified
    because inner folds inherit the family-stratification from the outer-train
    slice already.
    """
    from sklearn.model_selection import KFold
    fit_kwargs = fit_kwargs or {}
    N = X.shape[0]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros((N, N_CHANNELS), dtype=np.float32)
    for tr, va in kf.split(np.arange(N)):
        m = model_fn(X[tr], Y[tr], **fit_kwargs)
        oof[va] = m.predict(X[va])
    return oof


BASELINE_REGISTRY = {"ridge": fit_ridge, "xgb": fit_xgb}
