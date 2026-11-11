"""abs_xgbcore.py — ABS3 one-direction XGB fit (docs/specs/ABS3_MODELS.md 0-3, 0-4): the joblib worker of xgb_abs.py.
numpy only at import (sklearn / xgboost imported inside the worker), so the module loads in every process.

Per fit: X columns z-scored with the fit's training reactions (mean, std ddof 0; columns with std < SCALE_EPS = 1e-12
dropped: the d1_train.Scale rule), y standardised (y - mean) / std over the same reactions; estimator =
clone(train_ml_single.GRIDS["XGB"][0]).set_params(**cell) (the pair XGB estimator: random_state 42, n_jobs 1, hist), one
fit; kcal prediction = mean + std * predict(z) (float64)."""
from __future__ import annotations

import numpy as np

SCALE_EPS = 1e-12                                     # = d1pair_common.SCALE_EPS (d1_train.Scale)


def scale(X, rows):
    """training-reaction statistics of X: (mu, s, keep) with keep = s >= SCALE_EPS."""
    x = X[np.asarray(rows, dtype=int)]
    mu, s = x.mean(axis=0), x.std(axis=0)
    return mu, s, s >= SCALE_EPS


def stats_sha(mu, s, keep):
    """sha256 of one fit's X statistics (mu, s float64, keep bool; C-T compares it with a recomputation)."""
    import hashlib
    h = hashlib.sha256()
    for a in (np.asarray(mu, dtype=np.float64), np.asarray(s, dtype=np.float64), np.asarray(keep, dtype=bool)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def z(X, rows, mu, s, keep):
    x = X[np.asarray(rows, dtype=int)]
    return (x[:, keep] - mu[keep]) / s[keep]


def task(paths, ztr, ys, ymu, ysd, params, evals):
    """one fit (joblib worker): returns ({name: kcal prediction of evals[name]}, seconds)."""
    import sys as _s
    import time as _t
    _s.dont_write_bytecode = True
    for p in paths:
        if p not in _s.path:
            _s.path.insert(0, p)
    import train_ml_single as T
    from sklearn.base import clone
    t0 = _t.time()
    m = clone(T.GRIDS["XGB"][0]).set_params(**params).fit(ztr, ys)
    out = {k: np.asarray(m.predict(e), dtype=float) * float(ysd) + float(ymu) for k, e in evals.items()}
    return out, _t.time() - t0
