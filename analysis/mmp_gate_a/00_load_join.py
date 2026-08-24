#!/usr/bin/env python3
"""Step 0: load cohort + join with ds3, verify GATE-0.

Inputs:
  analysis/bath1480_probe/experiments/cohort_v1/labels_v1.pkl   (3504 rxns, 8 channels)
  analysis/bath1480_probe/experiments/cohort_v1/reactions/rxn_XXXX/{strain.json,eda.inp,e_ab.txt}
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv (ds3, 5269 rxns)

Outputs:
  artifacts/cohort_join.pkl         (3504 rows, joined labels + rxn_smiles + solvent + temp + G_act + G_r)
  artifacts/channel_coverage.csv    (per-channel n + coverage)

GATE-0: 3504 rxns exist in reactions/, all present in ds3, atom composition matches.
"""
import sys
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
COHORT_DIR = BASE / "analysis/bath1480_probe/experiments/cohort_v1"
REACTIONS = COHORT_DIR / "reactions"
LABELS_PKL = COHORT_DIR / "labels_v1.pkl"
DS3_CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
OUT = BASE / "analysis/mmp_gate_a/artifacts"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds"]
CHANNEL_TO_COL = {
    "d1": "d1_own", "d2": "d2_own",
    "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
    "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft",
}


def parse_eda_inp_composition(path: Path) -> Counter:
    """Return Counter of atomic symbols from xyz block of eda.inp."""
    comp = Counter()
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
                m = re.match(r"^\s*([A-Z][a-z]?)(?:\(\d+\))?\s+", line)
                if m:
                    comp[m.group(1)] += 1
    return comp


def smiles_composition(smi: str) -> Counter:
    """Return heavy-atom Counter for a SMILES (no H addition)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return Counter()
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def main():
    labels = pd.read_pickle(LABELS_PKL)
    labels["reaction_number"] = labels["reaction_number"].astype(int)
    n_lab = len(labels)
    assert n_lab == 3504, f"labels_v1 rows={n_lab}, expected 3504"

    rxn_dirs = sorted(REACTIONS.glob("rxn_*"))
    n_dirs = len(rxn_dirs)
    assert n_dirs == 3504, f"reactions dirs={n_dirs}, expected 3504"

    ds3 = pd.read_csv(DS3_CSV)
    print(f"ds3 rows: {len(ds3)}  cols: {list(ds3.columns)}")
    ds3["rxn_id"] = ds3["rxn_id"].astype(int)
    ds3_ids = set(ds3["rxn_id"])
    print(f"ds3 unique rxn_id: {len(ds3_ids)}")

    lab_ids = set(labels["reaction_number"])
    missing_in_ds3 = lab_ids - ds3_ids
    print(f"labels ids not in ds3: {len(missing_in_ds3)}")
    assert not missing_in_ds3, f"GATE-0 FAIL: {len(missing_in_ds3)} labels missing in ds3"

    joined = labels.merge(
        ds3[["rxn_id", "rxn_smiles", "solvent", "temp", "G_act", "G_r"]],
        left_on="reaction_number", right_on="rxn_id", how="left"
    )
    assert joined["rxn_smiles"].notna().all(), "GATE-0 FAIL: join produced NaN rxn_smiles"
    print(f"Joined rows: {len(joined)}")

    print("Verifying atom composition (eda.inp vs ds3 rxn_smiles reactants + H) ...")
    comp_ok = comp_fail = 0
    fail_ids = []
    for _, row in joined.iterrows():
        rxn = int(row["reaction_number"])
        eda_inp = REACTIONS / f"rxn_{rxn:04d}" / "eda.inp"
        eda_comp_heavy = Counter({k: v for k, v in parse_eda_inp_composition(eda_inp).items() if k != "H"})
        # reactants side of >> in rxn_smiles
        reactant_side = row["rxn_smiles"].split(">>")[0]
        mols = [Chem.MolFromSmiles(s) for s in reactant_side.split(".")]
        ds3_comp = Counter()
        for m in mols:
            if m is None:
                continue
            for a in m.GetAtoms():
                ds3_comp[a.GetSymbol()] += 1
        if eda_comp_heavy == ds3_comp:
            comp_ok += 1
        else:
            comp_fail += 1
            fail_ids.append((rxn, dict(eda_comp_heavy), dict(ds3_comp)))

    print(f"Composition match: {comp_ok}/{len(joined)}  fail={comp_fail}")
    if comp_fail:
        for rxn, e, d in fail_ids[:5]:
            print(f"  FAIL rxn_{rxn:04d}: eda={e}  ds3={d}")
    assert comp_fail == 0, f"GATE-0 FAIL: {comp_fail} composition mismatches"

    joined.to_pickle(OUT / "cohort_join.pkl")
    print(f"Saved cohort_join.pkl ({len(joined)} rows)")

    # Channel coverage
    rows = []
    for ch in CHANNELS:
        col = CHANNEL_TO_COL[ch]
        n_present = int(joined[col].notna().sum())
        rows.append({"channel": ch, "column": col, "n_present": n_present,
                     "n_total": len(joined), "coverage": n_present / len(joined)})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "channel_coverage.csv", index=False)
    print("Channel coverage:")
    print(cov.to_string(index=False))
    low = cov[cov["coverage"] < 0.95]
    if len(low):
        print(f"WARN: {len(low)} channels below 95% coverage")

    print("=== GATE-0 PASS ===")


if __name__ == "__main__":
    main()
