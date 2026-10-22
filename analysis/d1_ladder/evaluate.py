#!/usr/bin/env python3
"""D1-LADDER evaluation (docs/specs/D1LADDER.md 4-5, 11). sbatch only.

Now: the LINE reference row (spec 4-5): per target t, beta_t = sum(dH_t * y_t) / sum(dH_t^2) on exactly the given pairs,
no intercept (antisymmetric; = D1-HF Phase A report.py 'a'), c_ghost beta = 0 (no HF counterpart); prediction
beta * dH. Contract used by tests/run_tests.py L-10 (C-11 / C-18): line_beta(dH, y) -> beta [10] in TARGETS order.
Later (E1): the Phase E metrics, bends and comparisons reuse FU2 Phase E (fu2/evaluate.py bends_tab, compare) and
d1_eval.boot_diff / clopper / family_eval, hammett_eval.calls once FU2 Phase E is committed (spec 11); not written yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lad_common as L  # noqa: E402

TARGETS = list(L.DP.TARGETS)                             # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
NO_HF = ("c_ghost",)


def line_beta(dH, y):
    """beta [len(TARGETS)]: zero-intercept least squares per target column on the given pairs (rows); c_ghost and any
    column with sum(dH^2) == 0 give 0. dH, y: arrays [n_pairs, len(TARGETS)] (NaN rows must be removed by the caller)."""
    dH, y = np.asarray(dH, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if dH.shape != y.shape or dH.ndim != 2 or dH.shape[1] != len(TARGETS):
        raise ValueError(f"line_beta: dH {dH.shape} / y {y.shape}, need [n, {len(TARGETS)}]")
    beta = np.zeros(len(TARGETS), dtype=np.float64)
    for j, t in enumerate(TARGETS):
        if t in NO_HF:
            continue
        den = float((dH[:, j] ** 2).sum())
        beta[j] = float((dH[:, j] * y[:, j]).sum() / den) if den > 0 else 0.0
    return beta


def line_predict(beta, dH):
    """LINE prediction beta * dH [n_pairs, len(TARGETS)]."""
    return np.asarray(dH, dtype=np.float64) * np.asarray(beta, dtype=np.float64)[None, :]


if __name__ == "__main__":
    raise SystemExit(__doc__)
