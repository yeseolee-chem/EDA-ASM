"""Phase 2 - auto fragmentation for the 800 v8 cohort.

Strategy: TS-native indices only. Fragment split based on family conventions
learned from previous v7 manual reviews (which were TS-native):

  dipolar:    two reactants combine at TS; split via bond formation site.
              Use R connectivity (bimolecular) via BFS on distance <= 1.3 * (r_cov_i + r_cov_j).
              Map R fragment membership to TS by preserving atom order (R has same atoms in same order as TS).

  qmrxn20_e2, qmrxn20_sn2: reactant complex at R has 2 connected components.
              Same atom order in R and TS. Use R connectivity BFS.

  rgd1:       R and TS have same order. Use R BFS.

For every family the split is defined on R (2 connected components) and applied
to TS indices unchanged. Output is TS-native atom indices.

Output: outputs/v8_review/auto_partitions.json  (dict: rid -> {frag_A_indices, frag_B_indices, method, note})
"""
from __future__ import annotations
import json, sys
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
RAW = REPO / "outputs/v8_review/raw_geoms"
OUT = REPO / "outputs/v8_review/auto_partitions.json"

BOND_TOL = 1.3  # bond if distance < BOND_TOL * (r_cov_A + r_cov_B)


def connectivity(atoms):
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    n = len(Z)
    rc = np.array([covalent_radii[int(z)] for z in Z])
    thresh = BOND_TOL * (rc[:, None] + rc[None, :])
    d = cdist(pos, pos)
    A = (d > 0) & (d < thresh)
    return A


def two_components(atoms):
    A = connectivity(atoms)
    G = csr_matrix(A)
    n_comp, labels = connected_components(csgraph=G, directed=False, return_labels=True)
    return n_comp, labels


def main():
    cohort = pd.read_parquet(COHORT)
    partitions = {}
    fail_counts = {"ok": 0, "empty_fragment": 0, "wrong_component_count": 0,
                   "size_mismatch": 0, "missing_geom": 0}

    for row in cohort.itertuples(index=False):
        rid, family = row.reaction_id, row.family
        note = ""; method = "auto_R_bfs"
        try:
            R_at = ase.io.read(str(RAW / rid / "R.xyz"))
            TS_at = ase.io.read(str(RAW / rid / "TS.xyz"))
        except Exception as e:
            partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                               "method": "fail", "note": f"load: {e}",
                               "reviewed": False}
            fail_counts["missing_geom"] += 1
            continue

        Z_R = R_at.get_atomic_numbers()
        Z_TS = TS_at.get_atomic_numbers()

        # For dipolar: R has r0 + r1 concat, TS may have different total but same elements
        if family == "dipolar":
            # If R and TS have same n_atoms and same element order, indices map identically
            if len(Z_R) != len(Z_TS):
                # atom count mismatch - fallback: split R by connected components and
                # assume TS atom ordering starts with r0 (first fragment) then r1 (second).
                # This is the convention used by dipolar dataset preparation.
                n_comp, labels = two_components(R_at)
                if n_comp != 2:
                    partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                                       "method": "fail", "note": f"R has {n_comp} components (need 2)",
                                       "reviewed": False}
                    fail_counts["wrong_component_count"] += 1
                    continue
                # r0 = component label 0, r1 = component label 1
                # Assume TS ordering: first |r0| atoms are r0, remaining are r1
                n_r0 = int((labels == labels[0]).sum())
                if n_r0 + int((labels == labels[-1]).sum()) != len(Z_TS):
                    partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                                       "method": "fail", "note": "TS size mismatch",
                                       "reviewed": False}
                    fail_counts["size_mismatch"] += 1
                    continue
                fragA_TS = list(range(n_r0))               # r0 atoms
                fragB_TS = list(range(n_r0, len(Z_TS)))    # r1 atoms
                note = f"R_ncomp=2 via bfs; TS ordering = r0 then r1 (assumed)"
            else:
                # Same size - use R BFS directly, TS shares atom order
                n_comp, labels = two_components(R_at)
                if n_comp != 2:
                    partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                                       "method": "fail", "note": f"R has {n_comp} components",
                                       "reviewed": False}
                    fail_counts["wrong_component_count"] += 1
                    continue
                lbl_A = labels[0]
                fragA_TS = [i for i, l in enumerate(labels) if l == lbl_A]
                fragB_TS = [i for i, l in enumerate(labels) if l != lbl_A]

        elif family in ("qmrxn20_e2", "qmrxn20_sn2"):
            # R reactant complex may have 2 components (substrate + nucleophile/base).
            # R and TS share atom order.
            n_comp, labels = two_components(R_at)
            if n_comp != 2:
                # Fallback: try TS
                n_comp_ts, labels_ts = two_components(TS_at)
                if n_comp_ts == 2:
                    method = "auto_TS_bfs"
                    labels = labels_ts
                    note = "R had wrong components; used TS BFS"
                else:
                    partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                                       "method": "fail",
                                       "note": f"R n_comp={n_comp}, TS n_comp={n_comp_ts}",
                                       "reviewed": False}
                    fail_counts["wrong_component_count"] += 1
                    continue
            lbl_A = labels[0]
            # Note: qmrxn20 P has different n_atoms (LG lost). Use TS indices size.
            n_ts = len(Z_TS)
            if len(labels) != n_ts:
                # Should be equal since R == TS atom count for qmrxn20
                partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                                   "method": "fail",
                                   "note": f"R size {len(labels)} != TS size {n_ts}",
                                   "reviewed": False}
                fail_counts["size_mismatch"] += 1
                continue
            fragA_TS = [i for i, l in enumerate(labels) if l == lbl_A]
            fragB_TS = [i for i, l in enumerate(labels) if l != lbl_A]

        elif family == "rgd1":
            # R and TS same order. R may already have 2 components or may be a single
            # bonded reactant complex. Use R connectivity.
            n_comp, labels = two_components(R_at)
            if n_comp != 2:
                partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                                   "method": "fail",
                                   "note": f"R has {n_comp} components",
                                   "reviewed": False}
                fail_counts["wrong_component_count"] += 1
                continue
            lbl_A = labels[0]
            fragA_TS = [i for i, l in enumerate(labels) if l == lbl_A]
            fragB_TS = [i for i, l in enumerate(labels) if l != lbl_A]

        else:
            partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                               "method": "fail", "note": f"unknown family {family}",
                               "reviewed": False}
            fail_counts["missing_geom"] += 1
            continue

        if not fragA_TS or not fragB_TS:
            partitions[rid] = {"frag_A_indices": [], "frag_B_indices": [],
                               "method": "fail", "note": "empty fragment",
                               "reviewed": False}
            fail_counts["empty_fragment"] += 1
            continue

        partitions[rid] = {
            "frag_A_indices": fragA_TS,
            "frag_B_indices": fragB_TS,
            "method": method,
            "note": note,
            "reviewed": False,
        }
        fail_counts["ok"] += 1

    OUT.write_text(json.dumps(partitions, indent=2))
    print(f"\nwrote {OUT}  ({len(partitions)} rxns)")
    print("counts:", fail_counts)


if __name__ == "__main__":
    main()
