#!/usr/bin/env python3
"""Standalone GATE-P — parser round-trip only.

Parses 20 existing eda.out files with the Hartree column (10-digit precision)
× 627.5094740631 and verifies max |Δ| < 1e-6 kcal/mol vs
phase5_dataset_v2.pkl.

Why not m.group(3) (Kcal/mol column)?
  The Kcal/mol column of eda.out is rounded to 2 decimals ("-39.11" for a
  channel whose true value is -39.111677...). The phase5 labels were built
  from the Hartree column × conversion factor, giving 6-decimal precision.
  Using the Kcal/mol column gives ~3-6 mcal/mol reproduction error.
"""
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
EDA_OUT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")

HARTREE_TO_KCAL = 627.5094740631

EDA_TABLE_RE = re.compile(
    r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+",
    re.DOTALL
)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")

LABEL_TO_KEY = {
    "Bond Energy":                "bond",
    "Orbital Energy":             "orb",
    "Electrostatic Energy":       "elst",
    "Pauli Energy":               "pauli",
    "Delta E^0(XC)":              "xc",
    "Delta Dispersion":           "disp",
    "Delta CPCM Dielectric":      "cpcm",
    "Delta SMD CDS correction":   "smd_cds",
}


def parse_eda_hartree(path: Path):
    text = path.read_text(errors="replace")
    if not TERMINATED_RE.search(text):
        return None
    m = EDA_TABLE_RE.search(text)
    if not m:
        return None
    out = {}
    for row in ROW_RE.finditer(m.group(1)):
        label = row.group(1).strip()
        hartree = float(row.group(2))
        if label in LABEL_TO_KEY:
            out[LABEL_TO_KEY[label]] = hartree * HARTREE_TO_KCAL
    return out


def to_phase5(raw):
    return {
        "elst_dft":  raw["elst"],
        "pauli_dft": raw["pauli"] + raw["xc"],
        "oi_dft":    raw["orb"],
        "disp_dft":  raw["disp"],
        "cpcm_dft":  raw["cpcm"],
        "cds_dft":   raw["smd_cds"],
    }


def main():
    p5 = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    p5["reaction_number"] = p5["reaction_number"].astype(int)
    p5 = p5.set_index("reaction_number")

    rxns = sorted(p5.index.tolist())[:20]
    worst = 0.0
    n_ok = 0
    rows = []
    for rid in rxns:
        eda = EDA_OUT / f"rxn_{rid:04d}" / "eda.out"
        if not eda.exists():
            print(f"rxn {rid:4d}: MISSING")
            continue
        raw = parse_eda_hartree(eda)
        if raw is None:
            print(f"rxn {rid:4d}: parse failed")
            continue
        derived = to_phase5(raw)
        max_here = 0.0
        detail = {}
        for ch, dv in derived.items():
            lv = float(p5.at[rid, ch])
            diff = abs(dv - lv)
            detail[ch] = diff
            max_here = max(max_here, diff)
        rows.append({"rxn_id": rid, "max_abs_diff": max_here, **detail})
        if max_here < 1e-6:
            n_ok += 1
        worst = max(worst, max_here)
        print(f"rxn {rid:4d}: max_abs_diff = {max_here:.3e}")

    (BASE / "artifacts").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(BASE / "artifacts" / "parser_roundtrip.csv", index=False)

    status = "PASS" if worst < 1e-6 else "FAIL"
    print(f"\n=== GATE-P {status} ===")
    print(f"n_ok(diff<1e-6): {n_ok}/{len(rxns)}")
    print(f"worst max_abs_diff: {worst:.6e} kcal/mol")

    with open(BASE / "artifacts" / "GATEP_STATUS.txt", "w") as f:
        f.write(f"{status} n_ok={n_ok}/{len(rxns)} worst={worst:.6e}\n")


if __name__ == "__main__":
    main()
