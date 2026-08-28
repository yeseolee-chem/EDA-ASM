#!/usr/bin/env python3
"""Aggregate ORCA EDA-NOCV outputs from all rxn_XXXX/eda.out files into a parquet.

Parses each `data/rxn_XXXX/eda.out`, extracts the 7-component EDA table:
    Pauli, Electrostatic, Orbital, E^0(XC), Dispersion, CPCM_dielectric, SMD_CDS
+ Bond Energy (total interaction, = sum of above)

Also extracts top-5 NOCV pair contributions (DE_k) for downstream analysis.

Idempotent: skips reactions already present in the output parquet.

Usage (inside sbatch):
    python aggregate_channels.py data/ results/tt_eda_5channel.parquet
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


EDA_TABLE_RE = re.compile(
    r"Energy Term\s+Hartree\s+Kcal/mol\s*\n"
    r"-+\s*\n"
    r"(.*?)"
    r"\n\s*-+",
    re.DOTALL
)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
NOCV_TABLE_RE = re.compile(
    r"NOCV analysis.*?\n"
    r"\s+negative eigen\..*?\n"
    r"\s+\(e\).*?\n"
    r"-+\s*\n"
    r"(.*?)"
    # Terminator must be a real separator line: a run of dashes starting at
    # column 0. The old pattern `\n\s*-+` also matched the leading minus of a
    # data row (rows look like "    -0.6923121708   0.69...") and so captured
    # only the FIRST NOCV pair instead of the whole table.
    r"\n-{5,}",
    re.DOTALL
)
NOCV_ROW_RE = re.compile(r"^\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")

# Standard label -> parquet column mapping
LABEL_TO_COL = {
    "Bond Energy": "e_bond_kcal",
    "Orbital Energy": "e_orb_kcal",
    "Electrostatic Energy": "e_elst_kcal",
    "Pauli Energy": "e_pauli_kcal",
    "Delta E^0(XC)": "e_xc_kcal",
    "Delta Dispersion": "e_disp_kcal",
    "Delta CPCM Dielectric": "e_cpcm_kcal",
    "Delta SMD CDS correction": "e_smd_cds_kcal",
}


def parse_eda_out(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    if not TERMINATED_RE.search(text):
        return None

    table_match = EDA_TABLE_RE.search(text)
    if not table_match:
        return None

    rows = {}
    for m in ROW_RE.finditer(table_match.group(1)):
        label = m.group(1).strip()
        hartree = float(m.group(2))
        kcal = float(m.group(3))
        if label in LABEL_TO_COL:
            rows[LABEL_TO_COL[label]] = kcal

    # Top-5 NOCV pair energies
    nocv_top5 = []
    nocv_match = NOCV_TABLE_RE.search(text)
    if nocv_match:
        for i, m in enumerate(NOCV_ROW_RE.finditer(nocv_match.group(1))):
            if i >= 5:
                break
            nocv_top5.append({
                "eigen_neg": float(m.group(1)),
                "eigen_pos": float(m.group(2)),
                "de_k_kcal": float(m.group(3)),
                "dt_k_kcal": float(m.group(4)),
                "dv_k_kcal": float(m.group(5)),
            })
    rows["nocv_top5"] = nocv_top5
    rows["terminated_normally"] = True
    return rows


def main():
    if len(sys.argv) != 3:
        print("usage: aggregate_channels.py <data_root> <output_parquet>", file=sys.stderr)
        sys.exit(1)
    data_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    # Load existing results for resume
    existing = {}
    if out_path.exists():
        df_prev = pd.read_parquet(out_path)
        existing = {r["rxn_id"]: r for r in df_prev.to_dict('records')}
        print(f"resume: {len(existing)} entries already aggregated")

    rows = list(existing.values())
    n_new, n_fail, n_skip = 0, 0, 0

    rxn_dirs = sorted(data_root.glob("rxn_*"))
    print(f"found {len(rxn_dirs)} rxn_XXXX dirs under {data_root}")

    for i, rd in enumerate(rxn_dirs):
        rxn_id = int(rd.name.split("_")[1])
        if rxn_id in existing:
            n_skip += 1
            continue

        eda_out = rd / "eda.out"
        if not eda_out.exists():
            n_fail += 1
            continue

        parsed = parse_eda_out(eda_out)
        if parsed is None:
            rows.append({
                "rxn_id": rxn_id,
                "terminated_normally": False,
                "nocv_top5": [],
                **{c: None for c in LABEL_TO_COL.values()},
            })
            n_fail += 1
        else:
            rows.append({"rxn_id": rxn_id, **parsed})
            n_new += 1

        if (n_new + n_skip + n_fail) % 200 == 0:
            print(f"  progress: {n_new + n_skip + n_fail}/{len(rxn_dirs)} "
                  f"(new={n_new}, skip={n_skip}, fail={n_fail})", flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.rename(out_path)
    print(f"\nwrote {out_path}: {len(df)} rows")
    print(f"  successes: {df['terminated_normally'].sum()}")
    print(f"  failures:  {(~df['terminated_normally'].fillna(False)).sum()}")


if __name__ == "__main__":
    main()
