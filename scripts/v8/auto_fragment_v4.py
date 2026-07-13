"""Phase 2 v4 - trivial auto fragmentation once R is guaranteed to have exactly
2 connected components. Each component becomes fragment A or B (TS-native).
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

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
COHORT = REPO / "outputs/v8_review/cohort_v8.parquet"
RAW    = REPO / "outputs/v8_review/raw_geoms"
OUT    = REPO / "outputs/v8_review/auto_partitions.json"


def bfs_2(atoms, tol=1.3):
    Z = np.asarray(atoms.get_atomic_numbers())
    pos = atoms.get_positions()
    rc = np.array([covalent_radii[int(z)] for z in Z])
    d = cdist(pos, pos)
    A = (d > 0) & (d < tol * (rc[:, None] + rc[None, :]))
    n_comp, lbl = connected_components(csgraph=csr_matrix(A), directed=False, return_labels=True)
    if n_comp != 2:
        return None
    lbl0 = lbl[0]
    A_idx = [i for i, l in enumerate(lbl) if l == lbl0]
    B_idx = [i for i, l in enumerate(lbl) if l != lbl0]
    return A_idx, B_idx


def main():
    cohort = pd.read_parquet(COHORT)
    parts = {}
    ok = fail = 0
    for row in cohort.itertuples(index=False):
        rid = row.reaction_id
        try:
            R = ase.io.read(str(RAW / rid / "R.xyz"))
            AB = bfs_2(R)
            if AB is None:
                raise RuntimeError("R BFS != 2 components")
            A, B = AB
            parts[rid] = {
                "frag_A_indices": [int(i) for i in sorted(A)],
                "frag_B_indices": [int(i) for i in sorted(B)],
                "method": "R_bfs_2comp",
                "note": "trivial: each R component -> A/B",
                "reviewed": False,
            }
            ok += 1
        except Exception as e:
            parts[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                          "method": "fail", "note": str(e), "reviewed": False}
            fail += 1
    OUT.write_text(json.dumps(parts, indent=2))
    print(f"wrote {OUT}  ok={ok}  fail={fail}")


if __name__ == "__main__":
    main()
