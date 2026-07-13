"""Phase 2c - auto fragmentation v3 for the v8 cohort.

Fundamental principle (user directive):
  If R has TWO disconnected (unbonded) molecules -> they MUST become different
  fragments (A vs B). This is non-negotiable.

  If R has ONE connected component (bonded reactants at R already) -> use
  discretionary logic to guess the reactive split (TS bond-formation cut).

Algorithm (each step verified TS-native):
  1. R BFS at cov*1.3. If exactly 2 components AND R/TS atom order matches
     (Z_R == Z_TS positionwise): direct component labels -> A/B on TS.
  2. R BFS at cov*1.5 (permissive). Same requirement, direct if 2 components.
  3. For dipolar with R != TS ordering: try both {r0-first, r1-first} splits
     verified by TS element sequence.
  4. R = 1 component: TS bond-cut. Bonds present in TS but not in R = new bonds
     forming at TS. Remove them, split TS graph into 2 components.
  5. TS BFS at cov*1.5 (2 components). Only if 4 fails.
  6. Spectral bisection on TS distance matrix (geometric heuristic, last resort).

Output: outputs/v8_review/auto_partitions.json (overwrite),
         entries with method + note + reviewed=False.
"""
from __future__ import annotations
import json
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
from ase.data import covalent_radii
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
COHORT = REPO / "outputs/v8_review/cohort_v8.parquet"
RAW    = REPO / "outputs/v8_review/raw_geoms"
OUT    = REPO / "outputs/v8_review/auto_partitions.json"
RAW_DIP = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")


def connectivity(atoms, tol):
    Z = np.asarray(atoms.get_atomic_numbers())
    pos = atoms.get_positions()
    rc = np.array([covalent_radii[int(z)] for z in Z])
    d = cdist(pos, pos)
    thresh = tol * (rc[:, None] + rc[None, :])
    A = (d > 0) & (d < thresh)
    return A


def bfs_components(atoms, tol):
    A = connectivity(atoms, tol)
    n_comp, lbl = connected_components(csgraph=csr_matrix(A), directed=False, return_labels=True)
    return n_comp, lbl


def merge_to_two(labels, positions):
    """If BFS returned >2 components, merge smaller ones into the two largest by
    proximity to their centroids."""
    n_comp = int(labels.max()) + 1
    groups = [np.where(labels == c)[0] for c in range(n_comp)]
    groups.sort(key=lambda g: -len(g))
    A = list(groups[0]); B = list(groups[1])
    cA = positions[A].mean(0); cB = positions[B].mean(0)
    for g in groups[2:]:
        cg = positions[g].mean(0)
        if np.linalg.norm(cg - cA) < np.linalg.norm(cg - cB):
            A += g.tolist()
        else:
            B += g.tolist()
    return sorted(A), sorted(B)


def element_seqs_equal(Z1, Z2):
    return len(Z1) == len(Z2) and np.array_equal(Z1, Z2)


# =================================================================
# Method 1: R has 2 components + R atom order == TS atom order
# =================================================================
def method_R_bfs_direct(R_at, TS_at, tol):
    Z_R = np.array(R_at.get_atomic_numbers())
    Z_TS = np.array(TS_at.get_atomic_numbers())
    if not element_seqs_equal(Z_R, Z_TS):
        return None
    n_comp, lbl = bfs_components(R_at, tol=tol)
    if n_comp < 2:
        return None
    if n_comp == 2:
        A = [i for i, l in enumerate(lbl) if l == lbl[0]]
        B = [i for i, l in enumerate(lbl) if l != lbl[0]]
        return A, B
    # More than 2 components: merge to 2 by centroid proximity
    return merge_to_two(lbl, R_at.get_positions())


# =================================================================
# Method 2: dipolar with mismatched R/TS ordering - use raw r0/r1 files
# Try both possible orderings against TS element sequence.
# =================================================================
def method_dipolar_by_r0_r1_elems(rid, TS_at):
    idx_str = rid.split("_")[-1]
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    d = RAW_DIP / str(idx)
    r0_files = sorted(d.glob("r0_*.xyz"))
    r1_files = sorted(d.glob("r1_*.xyz"))
    if not r0_files or not r1_files:
        return None
    r0_at = ase.io.read(str(r0_files[0]))
    r1_at = ase.io.read(str(r1_files[0]))
    Z_TS = np.array(TS_at.get_atomic_numbers())
    Z_r0 = np.array(r0_at.get_atomic_numbers())
    Z_r1 = np.array(r1_at.get_atomic_numbers())
    n0 = len(Z_r0); n1 = len(Z_r1)
    if n0 + n1 != len(Z_TS):
        return None
    # Try r0-then-r1 ordering
    if np.array_equal(np.concatenate([Z_r0, Z_r1]), Z_TS):
        return list(range(n0)), list(range(n0, n0 + n1))
    # Try r1-then-r0
    if np.array_equal(np.concatenate([Z_r1, Z_r0]), Z_TS):
        return list(range(n1)), list(range(n1, n0 + n1))
    # Element sequence doesn't align cleanly - fall through
    return None


# =================================================================
# Method 3: R = 1 component - cut at new bonds in TS (bond formation site)
# Requires R and TS to have same atom order (element sequence identical).
# =================================================================
def method_TS_bond_cut(R_at, TS_at):
    Z_R = np.array(R_at.get_atomic_numbers())
    Z_TS = np.array(TS_at.get_atomic_numbers())
    if not element_seqs_equal(Z_R, Z_TS):
        return None
    R_bonds = connectivity(R_at, tol=1.3)
    TS_bonds = connectivity(TS_at, tol=1.3)
    new_bonds = TS_bonds & ~R_bonds
    if not new_bonds.any():
        return None
    # Remove new bonds from TS graph
    TS_cut = TS_bonds & ~new_bonds
    n_comp, lbl = connected_components(csgraph=csr_matrix(TS_cut),
                                       directed=False, return_labels=True)
    if n_comp < 2:
        return None
    if n_comp == 2:
        A = [i for i, l in enumerate(lbl) if l == lbl[0]]
        B = [i for i, l in enumerate(lbl) if l != lbl[0]]
        return A, B
    return merge_to_two(lbl, TS_at.get_positions())


# =================================================================
# Method 4: TS BFS with permissive tolerance -> 2 components
# =================================================================
def method_TS_bfs(TS_at, tol=1.5):
    n_comp, lbl = bfs_components(TS_at, tol=tol)
    if n_comp < 2:
        return None
    if n_comp == 2:
        A = [i for i, l in enumerate(lbl) if l == lbl[0]]
        B = [i for i, l in enumerate(lbl) if l != lbl[0]]
        return A, B
    return merge_to_two(lbl, TS_at.get_positions())


# =================================================================
# Method 5: Spectral bisection (geometric last-resort)
# =================================================================
def method_spectral(TS_at):
    pos = TS_at.get_positions()
    d = cdist(pos, pos)
    off = d[np.triu_indices_from(d, k=1)]
    sigma = float(np.median(off))
    aff = np.exp(-d ** 2 / (sigma ** 2 + 1e-9))
    sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                            assign_labels="kmeans", random_state=42)
    lbl = sc.fit_predict(aff)
    A = [i for i, l in enumerate(lbl) if l == lbl[0]]
    B = [i for i, l in enumerate(lbl) if l != lbl[0]]
    return A, B


# =================================================================
# Main resolve
# =================================================================
def resolve(rid, family, R_at, TS_at):
    # Step 1: R BFS direct at 1.3 (main principle: 2 unbonded -> A/B)
    ab = method_R_bfs_direct(R_at, TS_at, tol=1.3)
    if ab: return (*ab, "R_bfs_1.3_direct", "two disconnected mols in R")
    ab = method_R_bfs_direct(R_at, TS_at, tol=1.5)
    if ab: return (*ab, "R_bfs_1.5_direct", "permissive threshold")

    # Step 2: dipolar mismatched ordering via raw r0/r1 element check
    if family == "dipolar":
        ab = method_dipolar_by_r0_r1_elems(rid, TS_at)
        if ab: return (*ab, "dipolar_r0_r1_elems", "element sequence aligned to r0+r1 or r1+r0")

    # Step 3: R = 1 component -> TS bond-cut (discretionary)
    ab = method_TS_bond_cut(R_at, TS_at)
    if ab: return (*ab, "TS_bond_cut", "cut at bonds forming in TS (R -> TS diff)")

    # Step 4: TS BFS
    ab = method_TS_bfs(TS_at, tol=1.5)
    if ab: return (*ab, "TS_bfs_1.5", "TS connectivity fallback")

    # Step 5: spectral
    A, B = method_spectral(TS_at)
    return A, B, "spectral", "geometric bisection (needs manual review)"


def main():
    cohort = pd.read_parquet(COHORT)
    parts = {}
    method_counts = {}
    for row in cohort.itertuples(index=False):
        rid, family = row.reaction_id, row.family
        try:
            R_at = ase.io.read(str(RAW / rid / "R.xyz"))
            TS_at = ase.io.read(str(RAW / rid / "TS.xyz"))
        except Exception as e:
            parts[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                          "method": "load_fail", "note": str(e), "reviewed": False}
            continue
        try:
            A, B, meth, note = resolve(rid, family, R_at, TS_at)
            if not A or not B:
                raise RuntimeError("empty fragment after resolve")
            # ensure full coverage + no overlap
            all_idx = set(range(len(TS_at)))
            A = [i for i in A if 0 <= i < len(TS_at)]
            B = [i for i in B if 0 <= i < len(TS_at)]
            union = set(A) | set(B)
            missing = all_idx - union
            if missing:
                B = sorted(set(B) | missing)
                note = note + f"; auto-added {len(missing)} missing atoms to B"
            if set(A) & set(B):
                overlap = set(A) & set(B)
                A = [i for i in A if i not in overlap]
                note = note + f"; removed {len(overlap)} overlapping from A"
            parts[rid] = {
                "frag_A_indices": [int(x) for x in sorted(A)],
                "frag_B_indices": [int(x) for x in sorted(B)],
                "method": meth,
                "note": note,
                "reviewed": False,
            }
            method_counts[meth] = method_counts.get(meth, 0) + 1
        except Exception as e:
            parts[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                          "method": "fail", "note": f"{type(e).__name__}: {e}",
                          "reviewed": False}
            method_counts["fail"] = method_counts.get("fail", 0) + 1

    OUT.write_text(json.dumps(parts, indent=2))
    print(f"wrote {OUT}")
    print("method counts:", method_counts)


if __name__ == "__main__":
    main()
