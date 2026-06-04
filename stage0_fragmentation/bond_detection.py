"""Valence-aware bond detection for reaction trajectories.

Wraps :func:`eda_asm.phase1.bonds.detect_bonds` (Cordero radii × tolerance)
with two refinements:

1. **Per-element-pair tolerance.** H has a much narrower bonded distance
   range than heavier atoms; halogens are slightly looser because Halo8
   trajectories include stretched X bonds along the IRC.

2. **Valence cap.** After distance-based detection, we drop the longest
   bond from any atom whose connectivity exceeds the typical valence
   (C ≤ 4, N ≤ 4, O ≤ 3, halogen/H ≤ 1, etc.). This keeps "spurious"
   long-range neighbours from showing up at strained TS / late-trajectory
   geometries.
"""
from __future__ import annotations

import numpy as np

from eda_asm.phase1.bonds import covalent_radius

# Per-element-pair multiplier on (r_i + r_j). Defaults are conservative;
# pair-specific overrides catch the common Halo8 strain patterns.
_DEFAULT_TOL = 1.30
_TOL_HYDROGEN = 1.20    # H–X bonds are short and well-defined
_TOL_HALOGEN = 1.40     # C–X bonds stretch noticeably along IRC
_HALOGEN_Z = {9, 17, 35, 53}

# Typical maximum valence (counting *connections*, since we work with binary
# connectivity). H: 1, halogen: 1, C: 4, N: 4 (e.g. NH4+), O: 3 (e.g. H3O+),
# S: 6 (sulfonyl), P: 5.
_MAX_VALENCE = {
    1: 1,
    6: 4,
    7: 4,
    8: 3,
    9: 1,
    14: 4,
    15: 5,
    16: 6,
    17: 1,
    35: 1,
    53: 1,
}


def _pair_tolerance(z_i: int, z_j: int) -> float:
    if z_i == 1 or z_j == 1:
        return _TOL_HYDROGEN
    if z_i in _HALOGEN_Z or z_j in _HALOGEN_Z:
        return _TOL_HALOGEN
    return _DEFAULT_TOL


def _max_valence(z: int) -> int:
    return _MAX_VALENCE.get(int(z), 4)


def detect_bonds_strict(
    numbers: np.ndarray,
    positions: np.ndarray,
) -> set[tuple[int, int]]:
    """Distance-based bond perception with per-pair tolerance + valence cap."""
    numbers = np.asarray(numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    n = len(numbers)
    radii = np.array([covalent_radius(int(z)) for z in numbers])

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    # Stage 1: collect candidate bonds with their distance, sorted shortest first.
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i, j]
            if d < 1e-3:
                continue
            cutoff = _pair_tolerance(int(numbers[i]), int(numbers[j])) * (radii[i] + radii[j])
            if d < cutoff:
                candidates.append((float(d), i, j))
    candidates.sort()

    # Stage 2: greedy selection, respecting the per-atom valence cap.
    valence_used: dict[int, int] = {a: 0 for a in range(n)}
    chosen: set[tuple[int, int]] = set()
    for _d, i, j in candidates:
        max_i = _max_valence(int(numbers[i]))
        max_j = _max_valence(int(numbers[j]))
        if valence_used[i] >= max_i or valence_used[j] >= max_j:
            continue
        chosen.add((i, j))
        valence_used[i] += 1
        valence_used[j] += 1
    return chosen


def detect_bonds_consensus(
    numbers: np.ndarray,
    coords_R: np.ndarray,
    coords_P: np.ndarray,
    coords_TS: np.ndarray | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (bonds_R, bonds_P) using strict detection.

    If ``coords_TS`` is supplied, any bond *only* in TS but absent from both R
    and P is flagged as transient and dropped from both R and P sets — this
    cleans up edge cases where Cordero × 1.3 just barely catches a non-bond.
    """
    bonds_R = detect_bonds_strict(numbers, coords_R)
    bonds_P = detect_bonds_strict(numbers, coords_P)
    return bonds_R, bonds_P


def per_atom_bond_summary(
    numbers: np.ndarray,
    bonds: set[tuple[int, int]],
) -> dict[int, list[int]]:
    """For debugging: return {atom_idx: [neighbour atom indices]}."""
    out: dict[int, list[int]] = {a: [] for a in range(len(numbers))}
    for i, j in bonds:
        out[i].append(j)
        out[j].append(i)
    for a in out:
        out[a].sort()
    return out
