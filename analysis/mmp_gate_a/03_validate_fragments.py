#!/usr/bin/env python3
"""Step 3: cross-validate SMILES fragment split against eda.inp fragment labels (GATE-2).

For each reaction:
  - Parse eda.inp: atoms belong to fragment (1) or fragment (2) based on 'C(1)' / 'C(2)' labels
  - Get composition Counter for each fragment
  - From ds3 rxn_smiles reactant side, get 2 molecules, AddHs, get compositions
  - Unique-map: is there exactly one assignment of ds3 mol -> eda frag that matches?
"""
from pathlib import Path
from collections import Counter
import re
import pandas as pd
from rdkit import Chem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"
REACTIONS = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/reactions")


def parse_eda_frag_compositions(path: Path):
    """Return {1: Counter, 2: Counter} of atomic symbols (including H if present)."""
    frag_comp = {1: Counter(), 2: Counter()}
    in_xyz = False
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s.startswith("* xyz"):
                in_xyz = True
                continue
            if in_xyz:
                if s.startswith("*") or not s:
                    break
                m = re.match(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+", line)
                if m:
                    frag_comp[int(m.group(2))][m.group(1)] += 1
    return frag_comp


def mol_composition_with_h(smi: str) -> Counter:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return Counter()
    m = Chem.AddHs(m)
    return Counter(a.GetSymbol() for a in m.GetAtoms())


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl")
    n = len(cohort)
    print(f"Cohort: {n}")

    rows = []
    unique_ok = ambig = mismatch = 0
    for _, r in cohort.iterrows():
        rxn = int(r["reaction_number"])
        eda_inp = REACTIONS / f"rxn_{rxn:04d}" / "eda.inp"
        eda_frags = parse_eda_frag_compositions(eda_inp)
        reactant_side = r["rxn_smiles"].split(">>")[0]
        parts = reactant_side.split(".")
        if len(parts) != 2:
            mismatch += 1
            rows.append({"rxn_id": rxn, "status": "not_2mols", "mol0_assigned_to_frag": None})
            continue
        c0 = mol_composition_with_h(parts[0])
        c1 = mol_composition_with_h(parts[1])
        # Compare both possible assignments
        opt_A = (c0 == eda_frags[1] and c1 == eda_frags[2])
        opt_B = (c1 == eda_frags[1] and c0 == eda_frags[2])
        if opt_A and not opt_B:
            unique_ok += 1
            rows.append({"rxn_id": rxn, "status": "unique", "mol0_assigned_to_frag": 1,
                         "mol0_role": "dipole" if _dipole_like(parts[0]) else "dipolarophile",
                         "mol1_role": "dipole" if _dipole_like(parts[1]) else "dipolarophile"})
        elif opt_B and not opt_A:
            unique_ok += 1
            rows.append({"rxn_id": rxn, "status": "unique", "mol0_assigned_to_frag": 2,
                         "mol0_role": "dipole" if _dipole_like(parts[0]) else "dipolarophile",
                         "mol1_role": "dipole" if _dipole_like(parts[1]) else "dipolarophile"})
        elif opt_A and opt_B:
            ambig += 1
            rows.append({"rxn_id": rxn, "status": "ambiguous", "mol0_assigned_to_frag": None,
                         "mol0_role": "dipole" if _dipole_like(parts[0]) else "dipolarophile",
                         "mol1_role": "dipole" if _dipole_like(parts[1]) else "dipolarophile"})
        else:
            mismatch += 1
            rows.append({"rxn_id": rxn, "status": "mismatch",
                         "eda_f1": dict(eda_frags[1]), "eda_f2": dict(eda_frags[2]),
                         "mol0": dict(c0), "mol1": dict(c1)})

    frag_map = pd.DataFrame(rows)
    frag_map.to_pickle(OUT / "frag_map.pkl")

    total = unique_ok + ambig + mismatch
    rate = unique_ok / total * 100
    print(f"Unique map: {unique_ok}/{total} = {rate:.2f}%")
    print(f"Ambiguous: {ambig}  Mismatch: {mismatch}")

    if rate < 99.0:
        print(f"\n=== GATE-2 FAIL: unique_rate={rate:.2f}% < 99% ===")
        with open(OUT / "GATE2_STATUS.txt", "w") as f:
            f.write(f"FAIL unique_rate={rate:.2f}% ambig={ambig} mismatch={mismatch}\n")
    else:
        print(f"\n=== GATE-2 PASS: unique_rate={rate:.2f}% >= 99% ===")
        with open(OUT / "GATE2_STATUS.txt", "w") as f:
            f.write(f"PASS unique_rate={rate:.2f}% ambig={ambig} mismatch={mismatch}\n")


def _dipole_like(smi: str) -> bool:
    """Heuristic: dipoles have formal charges. Dipolarophiles are neutral."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return any(a.GetFormalCharge() != 0 for a in m.GetAtoms())


if __name__ == "__main__":
    main()
