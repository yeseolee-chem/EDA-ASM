"""Detect migrating atoms from R / P BE matrices.

The spec defines a migrating atom k as one whose row in ΔBE has both a
negative and a positive off-diagonal entry, with totals balanced. We
additionally require at least one *fully* broken bond (R-bond gone in P)
and one *fully* new bond (no R-bond, present in P) so that the routine
isolates true ligand transfers (H shifts, hypervalent shuttles, [1,j]
migrations) instead of every atom whose bond order merely changes (which
would catch every atom in a Diels-Alder, breaking partitioning entirely).
"""
from __future__ import annotations

import numpy as np


def detect_migrating_atoms(
    B_R: np.ndarray,
    B_P: np.ndarray,
) -> list[dict]:
    """Return a list of migrating-atom records.

    Each record is a dict::

        {"atom": k, "from": [i, ...], "to": [j, ...],
         "loss": int, "gain": int}

    where ``loss == gain`` is the magnitude of the balanced rearrangement at
    atom *k*.

    Only atoms with at least one *completely broken* bond and one *completely
    new* bond are reported (see module docstring).
    """
    if B_R.shape != B_P.shape or B_R.ndim != 2 or B_R.shape[0] != B_R.shape[1]:
        raise ValueError("B_R and B_P must be square arrays of the same shape")
    delta = B_P - B_R
    n = delta.shape[0]
    out: list[dict] = []
    for k in range(n):
        losses: list[tuple[int, int]] = []
        gains: list[tuple[int, int]] = []
        full_break = False
        full_form = False
        for j in range(n):
            if j == k:
                continue
            d = int(delta[k, j])
            if d < 0:
                losses.append((j, -d))
                if B_P[k, j] == 0 and B_R[k, j] > 0:
                    full_break = True
            elif d > 0:
                gains.append((j, d))
                if B_R[k, j] == 0 and B_P[k, j] > 0:
                    full_form = True
        if not losses or not gains:
            continue
        loss_total = sum(amt for _, amt in losses)
        gain_total = sum(amt for _, amt in gains)
        if loss_total != gain_total:
            continue
        if not (full_break and full_form):
            continue
        out.append(
            {
                "atom": k,
                "from": [j for j, _ in losses],
                "to": [j for j, _ in gains],
                "loss": loss_total,
                "gain": gain_total,
            }
        )
    return out


def reactive_bonds_from_delta(delta_be: np.ndarray) -> list[tuple[int, int]]:
    """Return sorted unique (i, j) bond pairs where ΔB ≠ 0 (i < j)."""
    if delta_be.ndim != 2 or delta_be.shape[0] != delta_be.shape[1]:
        raise ValueError("delta_be must be a square 2-D array")
    n = delta_be.shape[0]
    out: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if delta_be[i, j] != 0:
                out.add((i, j))
    return sorted(out)
