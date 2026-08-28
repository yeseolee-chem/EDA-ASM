#!/usr/bin/env python3
"""SPEC14 Step 4 — parse ORCA EDA outputs using the existing project parser
regexes (from scripts/parse_orca_5channel.py) so the new labels are directly
comparable to the current phase5 labels.

Output column names match phase5_dataset_v2.pkl EDA labels (without _dft
suffix — added in Step 5 for merging).

Only rxns whose eda.out is 'ORCA TERMINATED NORMALLY' are kept; the rest
are reported to GATE3_STATUS.txt for review.
"""
import re
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")

# Regexes lifted from scripts/parse_orca_5channel.py — canonical for this project.
EDA_CHANNEL_RE = {
    "bond":  re.compile(r"Bond Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "orb":   re.compile(r"Orbital Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "elst":  re.compile(r"Electrostatic Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "pauli": re.compile(r"Pauli Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "dxc":   re.compile(r"Delta E\^0\(XC\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "disp":  re.compile(r"Delta Dispersion\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
}
# CPCM + CDS are ORCA solvation contributions — regex names to be verified
# against a real output on first successful calc (see NOTE below).
SOLVATION_RE = {
    "cpcm": re.compile(r"CPCM Dielectric[^\n]*?(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "cds":  re.compile(r"SMD CDS[^\n]*?(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
}
# NOTE: solvation regex patterns must be reconciled with actual eda.out
# text on execution day. If mismatched, parse the exact phrase used by
# the existing 3504-label pipeline and update this block. Store the
# verified regex in artifacts/parser_notes.md.

FSP_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def tail_terminated(p: Path) -> bool:
    if not p.exists():
        return False
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4000))
        return b"ORCA TERMINATED NORMALLY" in f.read()


def parse_channels(eda_out: Path) -> dict:
    if not eda_out.exists():
        return {}
    text = eda_out.read_text(errors="ignore")
    out = {}
    for name, rx in EDA_CHANNEL_RE.items():
        for m in rx.finditer(text):
            out[name] = float(m.group(2))  # column 2 = kcal/mol
    for name, rx in SOLVATION_RE.items():
        for m in rx.finditer(text):
            out[name] = float(m.group(2))
    return out


def main():
    input_dirs = sorted((BASE / "inputs").glob("rxn_*"))
    rows = []
    fails = []
    for d in input_dirs:
        rid = int(d.name.split("_")[1])
        eda_out = d / "eda.out"
        if not tail_terminated(eda_out):
            fails.append(rid)
            continue
        ch = parse_channels(eda_out)
        # Pauli channel used in project convention = ORCA "Pauli Energy" + "Delta E^0(XC)"
        # per parse_orca_5channel.py header.
        if "pauli" in ch and "dxc" in ch:
            ch["pauli_combined"] = ch["pauli"] + ch["dxc"]
        row = {"rxn_id": rid}
        # channel column names match phase5 (without _dft suffix)
        row["elst"] = ch.get("elst")
        row["pauli"] = ch.get("pauli_combined", ch.get("pauli"))
        row["oi"] = ch.get("orb")
        row["disp"] = ch.get("disp")
        row["cpcm"] = ch.get("cpcm")
        row["cds"] = ch.get("cds")
        row["bond"] = ch.get("bond")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(BASE / "artifacts" / "channels_b3lyp.csv", index=False)

    n_total = len(input_dirs)
    n_ok = len(df)
    n_fail = len(fails)
    rate = n_ok / n_total if n_total else 0.0
    print(f"parsed          : {n_ok} / {n_total} ({rate*100:.1f}%)")
    print(f"failed / no-out : {n_fail}")

    status = "PASS" if rate >= 0.95 else "FAIL"
    with open(BASE / "artifacts" / "GATE3_STATUS.txt", "w") as f:
        f.write(f"{status} parsed={n_ok}/{n_total} rate={rate:.3f}\n")
        if fails:
            f.write("failed rxn_ids:\n")
            for r in fails:
                f.write(f"  {r}\n")
    print(f"=== GATE-3 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
