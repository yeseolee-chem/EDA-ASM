"""Phase 1b - re-sample 200 x 4 = 800 rxns REQUIRING R to have exactly 2
connected components (2 unbonded molecules). No exceptions.

For dipolar: r0.xyz + r1.xyz -> translate r1 by +10 A along +x so the two
molecules are guaranteed spatially separated. R BFS will report 2 components.

For qmrxn20 e2/sn2 and rgd1: check R BFS on the raw R.xyz. Reject reactions
where BFS != 2 components. Iterate through source until 200 collected per family.

Also verifies R element sequence == TS element sequence (positionwise) so that
atom indices are consistent between R and TS (used by ORCA input writer).
Reactions whose elements do not match position-wise are still accepted for
dipolar (halved swap handled downstream) but skipped for qmrxn20/rgd1.

Wipes and rewrites outputs/v8_review/raw_geoms/ and cohort_v8.parquet.
"""
from __future__ import annotations
import random, shutil, sys
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
from ase.data import covalent_radii
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
OUT = REPO / "outputs/v8_review/raw_geoms"
COHORT = REPO / "outputs/v8_review/cohort_v8.parquet"

SEED = 42
N_PER_FAMILY = 200
DIPOLAR_TRANSLATE = np.array([10.0, 0.0, 0.0])  # translate r1 by +10 A


def n_components(atoms, tol=1.3):
    Z = np.asarray(atoms.get_atomic_numbers())
    pos = atoms.get_positions()
    rc = np.array([covalent_radii[int(z)] for z in Z])
    d = cdist(pos, pos)
    A = (d > 0) & (d < tol * (rc[:, None] + rc[None, :]))
    n_comp, _ = connected_components(csgraph=csr_matrix(A), directed=False)
    return n_comp


def _single(dir_path: Path, pattern: str):
    m = sorted(dir_path.glob(pattern))
    return m[0] if m else None


def load_dipolar(idx: int):
    """Build R = r0 + r1_translated (spatially disjoint). Element-ordering
    mismatches with TS are resolved later via Hungarian permutation in main."""
    d = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles" / str(idx)
    if not d.exists(): return None
    r0f = _single(d, "r0_*.xyz")
    r1f = _single(d, "r1_*.xyz")
    ts  = _single(d, "TS_imag_mode.xyz") or _single(d, "TS_imag_mode_*.xyz") or _single(d, "TS_*.xyz")
    p0  = _single(d, "p0_*.xyz")
    if not (r0f and r1f and ts and p0):
        return None
    r0 = ase.io.read(str(r0f))
    r1 = ase.io.read(str(r1f))
    r1_pos = r1.get_positions() + DIPOLAR_TRANSLATE
    r1 = ase.Atoms(numbers=r1.get_atomic_numbers(), positions=r1_pos)
    R_at = r0 + r1
    TS_at = ase.io.read(str(ts))
    P_at = ase.io.read(str(p0))
    return R_at, TS_at, P_at


def load_qmrxn20(subfam: str, label: str):
    root = RAW / "QMrxn20"
    ts_p = root / "transition-states" / subfam / f"{label}.xyz"
    if not ts_p.exists(): return None
    TS_at = ase.io.read(str(ts_p))
    rc = root / "reactant-complex-constrained-conformers" / subfam / label
    r_p = rc / "00.xyz"
    if not r_p.exists():
        r_p = next(iter(rc.glob("*.xyz")), None) if rc.exists() else None
    R_at = ase.io.read(str(r_p)) if r_p else None
    parts = label.split("_")
    if subfam == "e2":
        plabel = "_".join(parts[:4] + ["0", "0"])
    else:
        plabel = "_".join(parts[:4] + ["0", parts[5]])
    pd_dir = root / "product-conformers" / subfam / plabel
    p_path = pd_dir / "00.xyz"
    if not p_path.exists() and pd_dir.exists():
        p_path = next(iter(pd_dir.glob("*.xyz")), None)
    P_at = ase.io.read(str(p_path)) if p_path else None
    if R_at is None or P_at is None: return None
    return R_at, TS_at, P_at


def load_rgd1(rid: str):
    d = RAW / "rgd1" / "extracted_xyz" / rid
    if not d.exists(): return None
    if not all((d / f"{s}.xyz").exists() for s in ("R", "TS", "P")):
        return None
    return (ase.io.read(str(d / "R.xyz")),
            ase.io.read(str(d / "TS.xyz")),
            ase.io.read(str(d / "P.xyz")))


def write_xyz(atoms, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(path), atoms, plain=True)


def _adj(atoms, tol=1.3):
    Z = np.asarray(atoms.get_atomic_numbers())
    pos = atoms.get_positions()
    rc = np.array([covalent_radii[int(z)] for z in Z])
    d = cdist(pos, pos)
    return (d > 0) & (d < tol * (rc[:, None] + rc[None, :]))


def _atom_signatures(atoms, depth=2, tol=1.3):
    """Weisfeiler-Lehman-like local signature per atom based on element + BFS
    neighbourhood up to given depth. Two atoms have the same signature iff
    their local chemistry is identical."""
    Z = np.asarray(atoms.get_atomic_numbers())
    A = _adj(atoms, tol=tol)
    n = len(Z)
    # depth 0: just element
    labels = [(int(Z[i]),) for i in range(n)]
    for _ in range(depth):
        new = []
        for i in range(n):
            neigh = sorted(labels[j] for j in range(n) if A[i, j])
            new.append((labels[i], tuple(neigh)))
        labels = new
    return labels


def _permute_R_to_TS(R_at, TS_at):
    """Match R atoms to TS atoms preserving CONNECTIVITY (chemistry), given
    the constraint that <= 2 bonds change between R and TS.

    Algorithm:
      1. Compute WL-like local signatures per atom in R and TS (depth 2).
      2. Group atoms by signature. Atoms with unique signature -> direct mapping.
      3. Atoms with ambiguous signature: element-preserving Hungarian on
         position distance within each signature bucket.
      4. Reaction-centre atoms (signature changed between R and TS due to bond
         formation/breaking) are matched last by element+distance Hungarian.

    Returns permuted R atoms whose atom order matches TS positionwise, or None.
    """
    Z_R = np.array(R_at.get_atomic_numbers())
    Z_T = np.array(TS_at.get_atomic_numbers())
    if len(Z_R) != len(Z_T): return None
    if not np.array_equal(sorted(Z_R.tolist()), sorted(Z_T.tolist())): return None
    pos_R = R_at.get_positions()
    pos_T = TS_at.get_positions()
    n = len(Z_R)

    sig_R = _atom_signatures(R_at)
    sig_T = _atom_signatures(TS_at)

    # For each TS atom, list of candidate R indices whose signature matches
    from collections import defaultdict
    R_by_sig = defaultdict(list)
    for j in range(n):
        R_by_sig[sig_R[j]].append(j)

    col_ind = np.full(n, -1, dtype=int)
    used = np.zeros(n, dtype=bool)
    unmatched_T = []

    # First pass: unique signature match (deterministic)
    for i in range(n):
        cand = R_by_sig.get(sig_T[i], [])
        if len(cand) == 1 and not used[cand[0]]:
            col_ind[i] = cand[0]; used[cand[0]] = True
        else:
            unmatched_T.append(i)

    # Second pass: for ambiguous signatures, group by signature and do Hungarian
    # on position distance within each group.
    sig_groups_T = defaultdict(list)
    for i in unmatched_T:
        sig_groups_T[sig_T[i]].append(i)
    for sig, T_indices in sig_groups_T.items():
        cand = [j for j in R_by_sig.get(sig, []) if not used[j]]
        if not cand:
            continue
        if len(cand) == len(T_indices):
            # Hungarian within this signature group
            d = cdist(pos_T[T_indices], pos_R[cand])
            r_ind, c_ind = linear_sum_assignment(d)
            for k in range(len(T_indices)):
                col_ind[T_indices[r_ind[k]]] = cand[c_ind[k]]
                used[cand[c_ind[k]]] = True

    # Third pass: reaction-centre atoms (signatures differ). Match by
    # element + distance Hungarian, considering only unassigned atoms.
    remaining_T = [i for i in range(n) if col_ind[i] < 0]
    remaining_R = [j for j in range(n) if not used[j]]
    if remaining_T:
        if len(remaining_T) != len(remaining_R): return None
        d = cdist(pos_T[remaining_T], pos_R[remaining_R])
        match_mask = Z_T[remaining_T, None] == Z_R[np.asarray(remaining_R)][None, :]
        cost = np.where(match_mask, d, 1e10)
        r_ind, c_ind = linear_sum_assignment(cost)
        for k in range(len(remaining_T)):
            col_ind[remaining_T[r_ind[k]]] = remaining_R[c_ind[k]]

    # Final verification: elements match after permutation
    Z_R_perm = Z_R[col_ind]
    if not np.array_equal(Z_R_perm, Z_T): return None
    return ase.Atoms(numbers=Z_R_perm, positions=pos_R[col_ind])


def valid_R_2_components(R_at, TS_at, family: str) -> bool:
    if n_components(R_at) != 2:
        return False
    Z_R = np.array(R_at.get_atomic_numbers())
    Z_T = np.array(TS_at.get_atomic_numbers())
    return len(Z_R) == len(Z_T) and np.array_equal(Z_R, Z_T)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    all_rows = []
    reject_counts = {"dipolar": 0, "qmrxn20_e2": 0, "qmrxn20_sn2": 0, "rgd1": 0}

    # dipolar
    dip_root = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
    dip_ids = sorted([int(p.name) for p in dip_root.iterdir() if p.is_dir() and p.name.isdigit()])
    rng.shuffle(dip_ids)
    n_ok = 0
    for i in dip_ids:
        if n_ok >= N_PER_FAMILY: break
        tr = load_dipolar(i)
        if tr is None:
            reject_counts["dipolar"] += 1; continue
        R_at, TS_at, P_at = tr
        # Try to permute R to TS atom order if they mismatch
        Z_R = np.array(R_at.get_atomic_numbers())
        Z_T = np.array(TS_at.get_atomic_numbers())
        if not (len(Z_R) == len(Z_T) and np.array_equal(Z_R, Z_T)):
            R_perm = _permute_R_to_TS(R_at, TS_at)
            if R_perm is not None:
                R_at = R_perm
        if not valid_R_2_components(R_at, TS_at, "dipolar"):
            reject_counts["dipolar"] += 1; continue
        rid = f"dipolar_{i:06d}"
        out_dir = OUT / rid
        write_xyz(R_at,  out_dir / "R.xyz")
        write_xyz(TS_at, out_dir / "TS.xyz")
        write_xyz(P_at,  out_dir / "P.xyz")
        all_rows.append({"reaction_id": rid, "family": "dipolar",
                         "n_atoms_R": len(R_at), "n_atoms_TS": len(TS_at), "n_atoms_P": len(P_at)})
        n_ok += 1
    print(f"dipolar: sampled {n_ok}/{N_PER_FAMILY} (rejected {reject_counts['dipolar']})")

    # qmrxn20
    for subfam in ("e2", "sn2"):
        fam = f"qmrxn20_{subfam}"
        ts_root = RAW / "QMrxn20" / "transition-states" / subfam
        labels = sorted([p.stem for p in ts_root.glob("*.xyz")])
        rng.shuffle(labels)
        n_ok = 0
        for lab in labels:
            if n_ok >= N_PER_FAMILY: break
            tr = load_qmrxn20(subfam, lab)
            if tr is None:
                reject_counts[fam] += 1; continue
            R_at, TS_at, P_at = tr
            Z_R = np.array(R_at.get_atomic_numbers()); Z_T = np.array(TS_at.get_atomic_numbers())
            if not (len(Z_R) == len(Z_T) and np.array_equal(Z_R, Z_T)):
                R_perm = _permute_R_to_TS(R_at, TS_at)
                if R_perm is not None: R_at = R_perm
            if not valid_R_2_components(R_at, TS_at, fam):
                reject_counts[fam] += 1; continue
            rid = f"qmrxn20_{subfam}_{lab}"
            out_dir = OUT / rid
            write_xyz(R_at,  out_dir / "R.xyz")
            write_xyz(TS_at, out_dir / "TS.xyz")
            write_xyz(P_at,  out_dir / "P.xyz")
            all_rows.append({"reaction_id": rid, "family": fam,
                             "n_atoms_R": len(R_at), "n_atoms_TS": len(TS_at), "n_atoms_P": len(P_at)})
            n_ok += 1
        print(f"{fam}: sampled {n_ok}/{N_PER_FAMILY} (rejected {reject_counts[fam]})")

    # rgd1
    rgd_root = RAW / "rgd1" / "extracted_xyz"
    rgd_ids = sorted([p.name for p in rgd_root.iterdir() if p.is_dir()])
    rng.shuffle(rgd_ids)
    n_ok = 0
    for rid in rgd_ids:
        if n_ok >= N_PER_FAMILY: break
        tr = load_rgd1(rid)
        if tr is None:
            reject_counts["rgd1"] += 1; continue
        R_at, TS_at, P_at = tr
        # Try to permute R to TS atom order if they mismatch
        Z_R = np.array(R_at.get_atomic_numbers())
        Z_T = np.array(TS_at.get_atomic_numbers())
        if not (len(Z_R) == len(Z_T) and np.array_equal(Z_R, Z_T)):
            R_perm = _permute_R_to_TS(R_at, TS_at)
            if R_perm is not None:
                R_at = R_perm
        if not valid_R_2_components(R_at, TS_at, "rgd1"):
            reject_counts["rgd1"] += 1; continue
        out_dir = OUT / rid
        write_xyz(R_at,  out_dir / "R.xyz")
        write_xyz(TS_at, out_dir / "TS.xyz")
        write_xyz(P_at,  out_dir / "P.xyz")
        all_rows.append({"reaction_id": rid, "family": "rgd1",
                         "n_atoms_R": len(R_at), "n_atoms_TS": len(TS_at), "n_atoms_P": len(P_at)})
        n_ok += 1
    print(f"rgd1: sampled {n_ok}/{N_PER_FAMILY} (rejected {reject_counts['rgd1']})")

    df = pd.DataFrame(all_rows)
    df.to_parquet(COHORT, index=False)
    print(f"\nwrote {COHORT}  total = {len(df)}  ({dict(df.family.value_counts())})")
    print(f"total rejections: {reject_counts}")


if __name__ == "__main__":
    main()
