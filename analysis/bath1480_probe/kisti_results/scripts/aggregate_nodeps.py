#!/usr/bin/env python3
"""Aggregate ORCA EDA-NOCV outputs -- ZERO third-party dependencies.

Nurion's python modules (3.9.5 / 3.12.5) and conda/pytorch_1.0 do NOT provide
pandas or pyarrow, so scripts/aggregate_channels.py cannot run there. This is
the same parsing logic with the same column names, writing CSV + JSON instead
of parquet. Convert to parquet later, anywhere that has pandas.

Idempotent: re-reads every eda.out each time (cheap -- they are ~80 KB), so it
can be run repeatedly while the campaign is still going.

Usage:
    python3 scripts/aggregate_nodeps.py data/ results/tt_eda_5channel.csv
"""
import csv
import json
import re
import sys
from pathlib import Path

EDA_TABLE_RE = re.compile(
    r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL
)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
# Terminator is a dashes run at column 0. `\n\s*-+` would also match the leading
# minus of a data row ("    -0.6923...") and capture only the first NOCV pair.
NOCV_TABLE_RE = re.compile(
    r"NOCV analysis.*?\n\s+negative eigen\..*?\n\s+\(e\).*?\n-+\s*\n(.*?)\n-{5,}",
    re.DOTALL,
)
NOCV_ROW_RE = re.compile(
    r"^\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
    re.MULTILINE,
)
TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")
NBF_RE = re.compile(r"Number of basis functions\s+\.\.\.\s+(\d+)")

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
EH_TO_KCAL = 627.5094740631
COLS = [
    "rxn_id", "terminated_normally", "n_basis_fn",
    *LABEL_TO_COL.values(),
    "channel_sum_kcal", "sum_minus_bond_kcal",
    # Same check computed from the Hartree column, which carries ten significant
    # figures. The kcal/mol column is rounded to 0.01, so summing seven of them
    # accumulates up to 7 x 0.005 = 0.035 of pure print error -- that swamps the
    # real residual, which is ~0.003 kcal/mol (median over 982 reactions).
    # Gate on THIS column, not the kcal one.
    "sum_minus_bond_eh_kcal", "n_nocv",
]


def parse(path):
    text = path.read_text(errors="replace")
    rec = {c: "" for c in COLS}
    rec["terminated_normally"] = bool(TERMINATED_RE.search(text))
    m = NBF_RE.search(text)
    rec["n_basis_fn"] = int(m.group(1)) if m else ""

    tbl = EDA_TABLE_RE.search(text)
    hartree = {}
    if tbl:
        for r in ROW_RE.finditer(tbl.group(1)):
            label = r.group(1).strip()
            if label in LABEL_TO_COL:
                rec[LABEL_TO_COL[label]] = float(r.group(3))
                hartree[LABEL_TO_COL[label]] = float(r.group(2))

    nocv = []
    nm = NOCV_TABLE_RE.search(text)
    if nm:
        for r in NOCV_ROW_RE.finditer(nm.group(1)):
            nocv.append({
                "eigen_neg": float(r.group(1)), "eigen_pos": float(r.group(2)),
                "de_k_kcal": float(r.group(3)), "dt_k_kcal": float(r.group(4)),
                "dv_k_kcal": float(r.group(5)),
            })
    rec["n_nocv"] = len(nocv)

    # Internal consistency: the seven components must reproduce Bond Energy.
    parts = [rec[c] for c in LABEL_TO_COL.values() if c != "e_bond_kcal"]
    if all(isinstance(v, float) for v in parts) and isinstance(rec["e_bond_kcal"], float):
        s = sum(parts)
        rec["channel_sum_kcal"] = round(s, 4)
        rec["sum_minus_bond_kcal"] = round(s - rec["e_bond_kcal"], 4)
    if len(hartree) == len(LABEL_TO_COL):
        sh = sum(v for k, v in hartree.items() if k != "e_bond_kcal")
        rec["sum_minus_bond_eh_kcal"] = round(
            (sh - hartree["e_bond_kcal"]) * EH_TO_KCAL, 5)
    return rec, nocv


def main():
    if len(sys.argv) != 3:
        print("usage: aggregate_nodeps.py <data_root> <output_csv>", file=sys.stderr)
        sys.exit(1)
    data_root, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows, nocv_all = [], {}
    n_ok = n_bad = n_missing = 0
    for rd in sorted(data_root.glob("rxn_*")):
        eda = rd / "eda.out"
        rxn_id = int(rd.name.split("_")[1])
        if not eda.exists():
            n_missing += 1
            continue
        rec, nocv = parse(eda)
        rec["rxn_id"] = rxn_id
        rows.append(rec)
        nocv_all[rxn_id] = nocv
        if rec["terminated_normally"]:
            n_ok += 1
        else:
            n_bad += 1

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    out_json = out_csv.with_suffix(".nocv.json")
    out_json.write_text(json.dumps(nocv_all))

    print(f"wrote {out_csv}   rows={len(rows)}")
    print(f"wrote {out_json}")
    print(f"  terminated normally : {n_ok}")
    print(f"  failed              : {n_bad}")
    print(f"  no eda.out yet      : {n_missing}")

    # Gate on the Hartree-derived residual (see COLS comment).
    vals = [r for r in rows if isinstance(r.get("sum_minus_bond_eh_kcal"), float)]
    for th in (0.02, 0.05):
        bad = [r for r in vals if abs(r["sum_minus_bond_eh_kcal"]) > th]
        print(f"  channel-sum mismatch, Hartree-precision (>{th} kcal): {len(bad)}")
        if th == 0.02:
            for r in sorted(bad, key=lambda x: -abs(x["sum_minus_bond_eh_kcal"]))[:10]:
                print(f"    rxn_{r['rxn_id']:04d}  diff={r['sum_minus_bond_eh_kcal']:+.5f}")
    if vals:
        a = sorted(abs(r["sum_minus_bond_eh_kcal"]) for r in vals)
        print(f"  residual: median {a[len(a)//2]:.2e}  max {a[-1]:.4f} kcal/mol")
    thin = [r for r in rows if r["terminated_normally"] and r["n_nocv"] < 2]
    print(f"  suspiciously few NOCV rows (<2): {len(thin)}")


if __name__ == "__main__":
    main()
