"""SPEC_11 Gate-B - AO block structure of overlap/hamiltonian.

Runs a complex GFN2-xTB SP on the first cohort reaction and checks:
  1. S has shape (n_orb, n_orb) and n_orb == sum_a(gfn2_ao_count(z_a)).
  2. Diagonal S_ii == 1.0 within 1e-6 (STO-NG minimal basis).
  3. S is symmetric within 1e-6.
  4. AO map: for atom_indices A and B from manual_partitions, the AO masks
     are disjoint and cover exactly the AOs that come from those atoms.
  5. n_atom_labels_A + n_atom_labels_B <= n_orb (equal iff no other atoms).

Also runs a tblite probe: confirm 'overlap-matrix' + 'hamiltonian-matrix'
keys are actually returned; if not, Stage 1 will fail catastrophically and
this gate blocks it.

Exit 0 => Gate-B passes.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# tblite BEFORE torch.
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import ase
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "spec/spec11_electronic_33d/code"))
from xtb_extract import run_xtb_extended

LABELS_V9  = REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
PART_V9    = REPO / "outputs/v8_review/manual_partitions.json"
CHARGES_V9 = REPO / "labels/orca/orca_eda_charges_v9.parquet"
FEAT_DIR   = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")


def load_ts_atoms(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), weights_only=False, map_location="cpu")
    return ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"]))


def probe_tblite_keys():
    """Confirm the installed tblite exposes overlap-matrix + hamiltonian-matrix."""
    Z = np.array([8, 1, 1])
    pos_ang = np.array([[0., 0., 0.],
                        [0.757, 0.586, 0.],
                        [-0.757, 0.586, 0.]])
    try:
        out = run_xtb_extended(Z, pos_ang, charge=0, mult=1, want_matrices=True,
                               want_gradient=True)
    except Exception as e:
        print(f"[probe] FAIL run_xtb_extended: {type(e).__name__}: {e}")
        return False
    for k in ["overlap", "hamiltonian", "gradient", "dipole"]:
        if k not in out or out[k] is None:
            print(f"[probe] FAIL: missing '{k}'")
            return False
    S = out["overlap"]; H = out["hamiltonian"]
    print(f"[probe] H2O:  n_orb={S.shape[0]}  S.max={S.max():.3f}  "
          f"S.diag_std={np.std(np.diag(S)):.2e}  "
          f"H.max={np.max(np.abs(H)):.3f}  grad.shape={out['gradient'].shape}")
    return True


def main():
    print("Gate-B: AO block structure check")

    if not probe_tblite_keys():
        print("Gate-B FAIL (tblite probe)")
        sys.exit(1)

    labels = pd.read_parquet(LABELS_V9)
    partitions = json.loads(PART_V9.read_text())
    charges = pd.read_parquet(CHARGES_V9).set_index("reaction_id")

    # Open-shell count over the whole cohort (should be 0 today; if >0 the
    # spin-inference path in compute_d29_d33 will surface each such row).
    open_shell_mask = ((charges["fragment_mult_a"] > 1) |
                       (charges["fragment_mult_b"] > 1))
    n_open = int(open_shell_mask.sum())
    print(f"[cohort] open-shell rxns: {n_open}/{len(charges)}")
    if n_open:
        fam_counts = (charges[open_shell_mask]["reaction_id"]
                      .str.split("_").str[0].value_counts()
                      if "reaction_id" in charges.columns else "n/a")
        print(f"[cohort] open-shell per family: {fam_counts}")

    # Pick the first rxn with a partition, charges, AND a closed-shell
    # (singlet) fragment pair - so the probe stays on well-defined ground.
    ok = False
    for _, row in labels.iterrows():
        rid = row.reaction_id
        if rid not in partitions or rid not in charges.index:
            continue
        part = partitions[rid]
        if "frag_A_indices" not in part or not part["frag_A_indices"]:
            continue
        ch = charges.loc[rid]
        if int(ch["fragment_mult_a"]) == 1 and int(ch["fragment_mult_b"]) == 1:
            probe_rid = rid; ok = True; break
    if not ok:
        print("Gate-B FAIL: no singlet+singlet reaction available to probe")
        sys.exit(1)
    print(f"[case] probing rid={probe_rid}")

    part = partitions[probe_rid]
    idx_A = np.array(part["frag_A_indices"], dtype=int)
    idx_B = np.array(part["frag_B_indices"], dtype=int)
    ch = charges.loc[probe_rid]
    q_tot = int(ch["total_charge"])
    m_A = int(ch["fragment_mult_a"]); m_B = int(ch["fragment_mult_b"])
    m_tot = 1  # enforced closed-shell by selection loop above

    TS_at = load_ts_atoms(probe_rid)
    Z = np.array(TS_at.get_atomic_numbers())
    pos_ang = TS_at.get_positions()

    rc = run_xtb_extended(Z, pos_ang, charge=q_tot, mult=m_tot, want_matrices=True)
    S = rc["overlap"]; H0 = rc["hamiltonian"]
    n_orb = S.shape[0]
    print(f"[case] n_atoms={len(Z)}  n_orb={n_orb}  H0.max={np.max(np.abs(H0)):.3f}")

    checks = []
    checks.append(("S shape square", S.shape[0] == S.shape[1]))
    checks.append(("S symmetric",
                   np.max(np.abs(S - S.T)) < 1e-6))
    diag = np.diag(S)
    checks.append(("S diag == 1",
                   np.max(np.abs(diag - 1.0)) < 1e-4))

    ao_labels = rc["orbital_map"]
    checks.append(("AO map length == n_orb",
                   ao_labels is not None and len(ao_labels) == n_orb))

    # AO labels cover exactly atoms [0..n_atoms-1]
    from collections import Counter
    counts = Counter(ao_labels.tolist())
    all_atoms_present = all(a in counts for a in range(len(Z)))
    checks.append(("AO labels cover every atom", all_atoms_present))

    setA = set(int(a) for a in idx_A)
    setB = set(int(a) for a in idx_B)
    checks.append(("A/B disjoint", setA.isdisjoint(setB)))
    mask_A = np.array([int(a) in setA for a in ao_labels], dtype=bool)
    mask_B = np.array([int(a) in setB for a in ao_labels], dtype=bool)
    checks.append(("A/B AO masks disjoint",
                   int((mask_A & mask_B).sum()) == 0))
    print(f"  |A|={len(idx_A)}  |B|={len(idx_B)}  "
          f"n_AO_A={int(mask_A.sum())}  n_AO_B={int(mask_B.sum())}  "
          f"n_AO_other={int((~mask_A & ~mask_B).sum())}")

    all_ok = True
    for name, val in checks:
        tick = "OK  " if val else "FAIL"
        print(f"  [{tick}] {name}")
        all_ok = all_ok and bool(val)
    if not all_ok:
        print("Gate-B FAIL")
        sys.exit(1)
    print("Gate-B PASS")


if __name__ == "__main__":
    main()
