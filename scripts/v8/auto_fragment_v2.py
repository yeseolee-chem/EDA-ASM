"""Phase 2b - improved auto fragmentation for 207 failures.

Family-specific rescue strategies:

  dipolar: raw dataset has separate r0_*.xyz + r1_*.xyz. The concat order in
           our R.xyz + TS.xyz preserves first |r0| atoms = fragA, remaining =
           fragB. Read r0_*.xyz atom count directly (deterministic).

  qmrxn20_e2/sn2, rgd1: BFS with adaptive threshold:
       first pass: BOND_TOL = 1.3 (standard)
       second:   BOND_TOL = 1.5 (permissive)
       third:    2-cluster spectral bisection on TS distance matrix
                (splits geometry into 2 spatially-connected halves)

Overwrites failing entries in outputs/v8_review/auto_partitions.json.
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
PART   = REPO / "outputs/v8_review/auto_partitions.json"
RAW_DIP = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")


def bfs_components(atoms, tol=1.3):
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    rc = np.array([covalent_radii[int(z)] for z in Z])
    thresh = tol * (rc[:, None] + rc[None, :])
    d = cdist(pos, pos)
    A = (d > 0) & (d < thresh)
    n_comp, lbl = connected_components(csgraph=csr_matrix(A), directed=False, return_labels=True)
    return n_comp, lbl


def largest_two_split(atoms, tol=1.3):
    """Return (fragA, fragB) as index lists from a 2-cluster split. If BFS
    gives >2 components, merge smaller components into A/B by nearest centroid."""
    n_comp, lbl = bfs_components(atoms, tol=tol)
    if n_comp < 2:
        return None
    # Group atoms by component; sort by size desc
    groups = [np.where(lbl == c)[0] for c in range(n_comp)]
    groups.sort(key=lambda g: -len(g))
    if n_comp == 2:
        return groups[0].tolist(), groups[1].tolist()
    # More than 2 components: merge small ones into the closer big cluster
    pos = atoms.get_positions()
    A = list(groups[0]); B = list(groups[1])
    cA = pos[A].mean(0); cB = pos[B].mean(0)
    for g in groups[2:]:
        cg = pos[g].mean(0)
        if np.linalg.norm(cg - cA) < np.linalg.norm(cg - cB):
            A += g.tolist()
        else:
            B += g.tolist()
    return sorted(A), sorted(B)


def spectral_bisect(atoms):
    """Fallback: spectral clustering on inverse-distance affinity, 2 clusters."""
    pos = atoms.get_positions()
    d = cdist(pos, pos)
    # affinity = exp(-d^2 / sigma^2), sigma = median off-diagonal distance
    off = d[np.triu_indices_from(d, k=1)]
    sigma = float(np.median(off))
    aff = np.exp(-d ** 2 / (sigma ** 2 + 1e-9))
    sc = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="kmeans",
                            random_state=42)
    lbl = sc.fit_predict(aff)
    A = [i for i, l in enumerate(lbl) if l == lbl[0]]
    B = [i for i, l in enumerate(lbl) if l != lbl[0]]
    return A, B


def dipolar_from_r0_count(rid: str, TS_at):
    """For dipolar rid=dipolar_XXXXXX, read raw r0_*.xyz to get n_r0."""
    idx = int(rid.split("_")[-1])
    d = RAW_DIP / str(idx)
    r0_files = sorted(d.glob("r0_*.xyz"))
    if not r0_files:
        return None
    r0_at = ase.io.read(str(r0_files[0]))
    n_r0 = len(r0_at)
    n_ts = len(TS_at)
    if n_r0 <= 0 or n_r0 >= n_ts:
        return None
    A = list(range(n_r0))
    B = list(range(n_r0, n_ts))
    return A, B


def resolve(rid, family, TS_at, R_at):
    """Return (fragA, fragB, method, note)."""
    # 1) dipolar - use raw r0 file to determine split (deterministic)
    if family == "dipolar":
        ab = dipolar_from_r0_count(rid, TS_at)
        if ab: return (*ab, "auto_r0_count", "split at raw |r0|")

    # 2) BFS on R with standard tolerance (may still work in some cases)
    ab = largest_two_split(R_at, tol=1.3)
    if ab: return (*ab, "auto_R_bfs_1.3", "")

    # 3) permissive R BFS
    ab = largest_two_split(R_at, tol=1.5)
    if ab: return (*ab, "auto_R_bfs_1.5", "permissive threshold")

    # 4) TS BFS (permissive)
    ab = largest_two_split(TS_at, tol=1.5)
    if ab: return (*ab, "auto_TS_bfs_1.5", "TS BFS fallback")

    # 5) spectral bisection on TS geometry (always yields two clusters)
    A, B = spectral_bisect(TS_at)
    return A, B, "auto_spectral", "geometric bisection (chemically approximate)"


def main():
    cohort = pd.read_parquet(COHORT)
    parts = json.loads(PART.read_text())
    fixed = 0; still_fail = 0
    method_counts = {}
    for row in cohort.itertuples(index=False):
        rid, family = row.reaction_id, row.family
        cur = parts.get(rid, {})
        if cur.get("method") not in (None, "fail"): continue
        # Retry
        try:
            TS_at = ase.io.read(str(RAW / rid / "TS.xyz"))
            R_at  = ase.io.read(str(RAW / rid / "R.xyz"))
        except Exception as e:
            still_fail += 1
            continue
        try:
            A, B, meth, note = resolve(rid, family, TS_at, R_at)
            if not A or not B or set(A) & set(B):
                raise RuntimeError("empty or overlapping")
            all_idx = set(range(len(TS_at)))
            if set(A) | set(B) != all_idx:
                # ensure full coverage - add missing atoms to B
                missing = all_idx - set(A) - set(B)
                B = sorted(set(B) | missing)
                note = (note + "; " if note else "") + f"added {len(missing)} unassigned atoms to B"
            parts[rid] = {
                "frag_A_indices": sorted(A),
                "frag_B_indices": sorted(B),
                "method": meth,
                "note": note,
                "reviewed": False,
            }
            fixed += 1
            method_counts[meth] = method_counts.get(meth, 0) + 1
        except Exception as e:
            parts[rid] = {**cur, "note": f"retry fail: {e}"}
            still_fail += 1

    PART.write_text(json.dumps(parts, indent=2))
    print(f"fixed {fixed} previously-failed rxns; {still_fail} still failing.")
    print("methods used:", method_counts)


if __name__ == "__main__":
    main()
