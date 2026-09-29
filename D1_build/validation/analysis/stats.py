#!/usr/bin/env python3
"""stats.py — VALIDATION_SPEC §6.5: pooled SD, t-based CI of a mean, RMS, Wilson CI, bootstrap helper."""
from __future__ import annotations

import numpy as np


def pooled_sd(groups):                     # groups: 반응별 반복값 배열 리스트
    groups = [np.asarray(g, float) for g in groups]
    num = sum(((g - g.mean()) ** 2).sum() for g in groups if len(g) > 1)
    den = sum(len(g) - 1 for g in groups if len(g) > 1)
    return float(np.sqrt(num / den)) if den else float("nan")


def mean_ci(x, level=0.95):
    """(mean, lo, hi): t distribution with n-1 degrees of freedom."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan"))
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * se
    return float(m), float(m - h), float(m + h)


def rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt((x ** 2).mean())) if len(x) else float("nan")


def wilson(k, n, level=0.95):
    from scipy import stats
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def percentile_ci(samples, level=0.95):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return float("nan"), float("nan")
    return float(np.percentile(s, 50 * (1 - level))), float(np.percentile(s, 100 - 50 * (1 - level)))
