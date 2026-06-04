"""Distance-based bond detection (Cordero 2008 covalent radii).

Halogen-containing bonds use a slightly looser tolerance because Halo8
trajectories include stretched bonds along the IRC.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from ase import Atoms

# Cordero et al., Dalton Trans. 2008, DOI: 10.1039/b801115j (Angstrom).
# Index = atomic number; element 0 left as 0.0 placeholder.
_CORDERO = {
    1: 0.31, 2: 0.28,
    3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    18: 1.06,
    19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39,
    26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20,
    33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
    53: 1.39,
}

DEFAULT_TOL = 1.3
HALOGEN_TOL = 1.4
_HALOGENS = {9, 17, 35, 53}


def covalent_radius(z: int) -> float:
    if z not in _CORDERO:
        raise KeyError(f"No Cordero radius for Z={z}")
    return _CORDERO[z]


def detect_bonds(
    numbers: np.ndarray | Iterable[int],
    positions: np.ndarray,
    tol: float = DEFAULT_TOL,
    halogen_tol: float = HALOGEN_TOL,
) -> set[tuple[int, int]]:
    """Return set of (i, j) with i < j connected by a covalent bond."""
    numbers = np.asarray(numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    n = len(numbers)
    if n != positions.shape[0]:
        raise ValueError("numbers and positions length mismatch")

    radii = np.array([covalent_radius(int(z)) for z in numbers])
    is_hal = np.array([int(z) in _HALOGENS for z in numbers])

    # Pairwise distance matrix (ok for n <= 50; Halo8 systems are tiny).
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    # Per-pair tolerance: halogen if either end is halogen.
    pair_hal = is_hal[:, None] | is_hal[None, :]
    pair_tol = np.where(pair_hal, halogen_tol, tol)
    pair_cut = pair_tol * (radii[:, None] + radii[None, :])

    bonds: set[tuple[int, int]] = set()
    iu, ju = np.triu_indices(n, k=1)
    mask = (dist[iu, ju] > 1e-3) & (dist[iu, ju] < pair_cut[iu, ju])
    for i, j in zip(iu[mask].tolist(), ju[mask].tolist()):
        bonds.add((i, j))
    return bonds


def bonds_from_atoms(atoms: Atoms) -> set[tuple[int, int]]:
    return detect_bonds(atoms.get_atomic_numbers(), atoms.get_positions())


def connected_components(n: int, bonds: set[tuple[int, int]]) -> list[list[int]]:
    """Return atoms partitioned into connected components (sorted, ascending size order reversed)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in bonds:
        union(i, j)
    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    comps = [sorted(v) for v in groups.values()]
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def bond_changes(
    numbers: np.ndarray,
    pos_R: np.ndarray,
    pos_TS: np.ndarray,
) -> dict:
    """Compute bond change metadata between R and TS geometries."""
    bR = detect_bonds(numbers, pos_R)
    bT = detect_bonds(numbers, pos_TS)
    broken = sorted(bR - bT)
    formed = sorted(bT - bR)
    n_atoms = len(numbers)
    comps_R = connected_components(n_atoms, bR)
    return {
        "bonds_R": sorted(bR),
        "bonds_TS": sorted(bT),
        "bonds_broken": [list(b) for b in broken],
        "bonds_formed": [list(b) for b in formed],
        "n_bond_changes": len(broken) + len(formed),
        "n_components_R": len(comps_R),
        "components_R": comps_R,
    }
