#!/usr/bin/env python3
"""SPEC15 Step 4 — parse EDA channels + strain energies.

Reuses the canonical Hartree-precision parser from spec14 (_gate_p_only.py
approach, not the rounded-Kcal aggregate_channels.py). This gives exact
match to phase5 labels for the reused 218 EDA outputs.

GATE-3: parse success rate ≥ 95%, no sign violations, d1/d2 > 0 always.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
GEOM = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")

# Reuse the vendored aggregate_channels.py path structure but use Hartree column
sys.path.insert(0, str(GEOM))
# The Hartree parser lives inside spec14 as parse_eda_hartree in _gate_p_only.py.
# We inline the same regex here to avoid the leading-underscore import quirk.

HARTREE_TO_KCAL = 627.5094740631

EDA_TABLE_RE = re.compile(
    r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")
FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

LABEL_TO_KEY = {
    "Bond Energy": "bond", "Orbital Energy": "orb",
    "Electrostatic Energy": "elst", "Pauli Energy": "pauli",
    "Delta E^0(XC)": "xc", "Delta Dispersion": "disp",
    "Delta CPCM Dielectric": "cpcm", "Delta SMD CDS correction": "smd_cds",
}


def parse_eda(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    if not TERMINATED_RE.search(text):
        return None
    m = EDA_TABLE_RE.search(text)
    if not m:
        return None
    out = {}
    for row in ROW_RE.finditer(m.group(1)):
        label = row.group(1).strip()
        if label in LABEL_TO_KEY:
            out[LABEL_TO_KEY[label]] = float(row.group(2)) * HARTREE_TO_KCAL
    return out


def last_fspe_hartree(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    if not TERMINATED_RE.search(text):
        return None
    vals = FSPE_RE.findall(text)
    return float(vals[-1]) if vals else None


def to_phase5(raw):
    if raw is None:
        return None
    need = ("elst", "pauli", "xc", "orb", "disp", "cpcm", "smd_cds")
    if not all(k in raw for k in need):
        return None
    return {
        "elst_dft":  raw["elst"],
        "pauli_dft": raw["pauli"] + raw["xc"],
        "oi_dft":    raw["orb"],
        "disp_dft":  raw["disp"],
        "cpcm_dft":  raw["cpcm"],
        "cds_dft":   raw["smd_cds"],
        "e_bond_kcal": raw.get("bond"),
    }


def main():
    meta = pd.read_csv(BASE / "artifacts" / "input_meta.csv")
    rows, bad = [], []
    for _, m in meta.iterrows():
        rid = int(m["rxn_id"])
        d = BASE / "inputs" / f"rxn_{rid:04d}"
        # EDA: try spec14 first, else this dir
        eda_src = GEOM / "results_218" / f"rxn_{rid:04d}" / "eda.out"
        eda_path = eda_src if eda_src.exists() else (d / "eda.out")
        ch = to_phase5(parse_eda(eda_path))
        if ch is None:
            bad.append((rid, "EDA parse failed")); continue

        e = {}
        ok = True
        for k in ("frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel"):
            v = last_fspe_hartree(d / f"{k}.out")
            if v is None:
                bad.append((rid, f"{k} missing/non-terminated"))
                ok = False; break
            e[k] = v
        if not ok:
            continue

        d1 = (e["frag1_dist"] - e["frag1_rel"]) * HARTREE_TO_KCAL
        d2 = (e["frag2_dist"] - e["frag2_rel"]) * HARTREE_TO_KCAL

        row = {"rxn_id": rid, **ch,
               "e_frag1_dist_eh": e["frag1_dist"],
               "e_frag1_rel_eh":  e["frag1_rel"],
               "e_frag2_dist_eh": e["frag2_dist"],
               "e_frag2_rel_eh":  e["frag2_rel"],
               "d1_b3lyp": d1, "d2_b3lyp": d2,
               "sum_dist_b3lyp": d1 + d2}
        rows.append(row)

    R = pd.DataFrame(rows)
    R.to_csv(BASE / "artifacts" / "labels_b3lyp.csv", index=False)
    n_ok, n_tot = len(R), len(meta)
    rate = n_ok / max(n_tot, 1)
    print(f"parsed: {n_ok}/{n_tot} ({rate*100:.1f}%)")
    if bad:
        print(f"failures: {len(bad)}  first 8: {bad[:8]}")

    # Sign/physics checks
    print("\n=== sign / physics checks ===")
    checks = {
        "d1 > 0":     int((R["d1_b3lyp"] > 0).sum()),
        "d2 > 0":     int((R["d2_b3lyp"] > 0).sum()),
        "pauli > 0":  int((R["pauli_dft"] > 0).sum()),
        "elst < 0":   int((R["elst_dft"] < 0).sum()),
        "oi < 0":     int((R["oi_dft"] < 0).sum()),
        "disp < 0":   int((R["disp_dft"] < 0).sum()),
    }
    n_viol = sum(1 for v in checks.values() if v != len(R))
    for lab, v in checks.items():
        tag = "OK" if v == len(R) else "VIOLATION"
        print(f"  {lab:<12} {v}/{len(R)}  {tag}")

    status = "PASS" if (rate >= 0.95 and n_viol == 0) else "FAIL"
    with open(BASE / "artifacts" / "GATE3_STATUS.txt", "w") as f:
        f.write(f"{status} parsed={n_ok}/{n_tot} rate={rate:.3f} sign_viol_channels={n_viol}\n")
        for lab, v in checks.items():
            f.write(f"  {lab}: {v}/{len(R)}\n")
        if bad:
            f.write("failures:\n")
            for r, why in bad:
                f.write(f"  rxn {r}: {why}\n")
    print(f"=== GATE-3 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
