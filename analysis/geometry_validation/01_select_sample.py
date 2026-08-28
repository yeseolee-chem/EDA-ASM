#!/usr/bin/env python3
"""SPEC14 Step 1 — stratified sample (core_sig × size_bin, cap 15).

GATE-1 targets (SPEC §3 Step 1):
  total selected      = 218
  core_sig × size_bin distribution:
      size_bin   S   M   L
      CCCCN     15  15  15
      CCCCO     15  15  15
      CCCNN     15  15  15
      CCCNO     15  15  15
      CCCOO     15   1   2
      CCNNN      8   1   0
      CCNNO      8   2   1
  first 20 rxn_ids: 3, 6, 52, 81, 113, 114, 121, 124, 126, 136, 155, 201,
                    266, 269, 270, 287, 289, 299, 308, 454
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
DS3 = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")

CAP = 15
SEED = 42


def core_signature(rxn_smiles):
    """5-membered ring containing both formed bonds in the product; return sorted
    atomic-symbol string (e.g., 'CCCCN'). None on failure.
    """
    reac, prod = rxn_smiles.split(">>")
    R, P = Chem.MolFromSmiles(reac), Chem.MolFromSmiles(prod)
    if R is None or P is None:
        return None

    def bondset(m):
        s = set()
        for b in m.GetBonds():
            i = b.GetBeginAtom().GetAtomMapNum()
            j = b.GetEndAtom().GetAtomMapNum()
            if i and j:
                s.add((min(i, j), max(i, j)))
        return s

    formed = bondset(P) - bondset(R)
    if len(formed) != 2:
        return None
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in P.GetAtoms() if a.GetAtomMapNum() > 0}
    try:
        fidx = {frozenset((m2i[i], m2i[j])) for i, j in formed}
    except KeyError:
        return None
    best = None
    for ring in P.GetRingInfo().AtomRings():
        rs = set(ring)
        if all(all(x in rs for x in fb) for fb in fidx):
            if best is None or len(ring) < len(best):
                best = ring
    if best is None:
        return None
    return "".join(sorted(P.GetAtomWithIdx(i).GetSymbol() for i in best))


def main():
    ds3 = pd.read_csv(DS3 / "full_dataset.csv").set_index("rxn_id")
    lab = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    lab["reaction_number"] = lab["reaction_number"].astype(int)
    lab = lab.set_index("reaction_number")

    rows = []
    for rid in lab.index:
        rid = int(rid)
        if rid not in ds3.index:
            continue
        sig = core_signature(ds3.at[rid, "rxn_smiles"])
        if sig is None:
            continue
        rows.append(dict(rxn_id=rid, core_sig=sig,
                         n_atoms=int(lab.at[rid, "n_atoms_frag1"]
                                     + lab.at[rid, "n_atoms_frag2"])))

    C = pd.DataFrame(rows)
    C["size_bin"] = pd.qcut(C["n_atoms"], 3, labels=["S", "M", "L"])
    print(f"Stratification pool: {len(C)}")
    print("Available per (core_sig × size_bin):")
    print(C.groupby(["core_sig", "size_bin"], observed=True)
          .size().unstack(fill_value=0).to_string())

    sel = pd.concat([g.sample(min(CAP, len(g)), random_state=SEED)
                     for _, g in C.groupby(["core_sig", "size_bin"], observed=True)])
    sel = sel.sort_values("rxn_id").reset_index(drop=True)
    print(f"\nSelected: {len(sel)}")
    print(sel.groupby(["core_sig", "size_bin"], observed=True)
          .size().unstack(fill_value=0).to_string())

    sel.to_csv(BASE / "artifacts" / "sample.csv", index=False)

    # Gate assertions
    expected_n = 218
    expected_first20 = [3, 6, 52, 81, 113, 114, 121, 124, 126, 136,
                        155, 201, 266, 269, 270, 287, 289, 299, 308, 454]
    got_first20 = sel["rxn_id"].tolist()[:20]

    status = "PASS"
    mismatches = []
    if len(sel) != expected_n:
        status = "FAIL"
        mismatches.append(f"n={len(sel)} != {expected_n}")
    if got_first20 != expected_first20:
        status = "FAIL"
        mismatches.append(f"first20 mismatch: got {got_first20}, expected {expected_first20}")

    with open(BASE / "artifacts" / "GATE1_STATUS.txt", "w") as f:
        f.write(f"{status} n={len(sel)}\n")
        if mismatches:
            f.write("mismatches:\n" + "\n".join(f"  {m}" for m in mismatches) + "\n")

    print(f"\n=== GATE-1 {status} ===")
    if mismatches:
        for m in mismatches:
            print(f"  MISMATCH: {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
