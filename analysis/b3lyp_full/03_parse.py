#!/usr/bin/env python3
"""SPEC16rev Step 3 — parse ORCA outputs. Field↔source 1:1 binding + FSPE count.

Hartree-precision parser (matches phase5 labels to 1e-6, verified in spec14).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
GEOM = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")

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

# ⭐ field name ↔ source file (single source, no ambiguity)
FIELD_SOURCE = {
    "e_ab_eh":         ("eda.out", 1),
    "e_frag1_dist_eh": ("frag1_dist.out", 1),
    "e_frag2_dist_eh": ("frag2_dist.out", 1),
    "e_frag1_rel_eh":  ("frag1_rel.out", 1),
    "e_frag2_rel_eh":  ("frag2_rel.out", 1),
}


def terminated_ok(path):
    p = Path(path)
    if not p.exists():
        return False
    return "ORCA TERMINATED NORMALLY" in p.read_text(errors="replace")


def read_fspe(path, expected_count):
    """Read FSPE with count check. Returns the last value (converged, if opt).

    For single-point files, expected_count=1. For opt files, expected_count=None.
    """
    text = Path(path).read_text(errors="replace")
    vals = FSPE_RE.findall(text)
    if not vals:
        raise ValueError(f"{Path(path).name}: no FSPE found")
    if expected_count is not None and len(vals) != expected_count:
        raise ValueError(f"{Path(path).name}: FSPE count={len(vals)} (expected {expected_count})")
    return float(vals[-1])


def parse_eda_channels(eda_path):
    """Extract 6 EDA channel energies + bond, from the Hartree column."""
    text = Path(eda_path).read_text(errors="replace")
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
    need = ("elst", "pauli", "xc", "orb", "disp", "cpcm", "smd_cds", "bond")
    if not all(k in out for k in need):
        return None
    return {
        "elst_dft":  out["elst"],
        "pauli_dft": out["pauli"] + out["xc"],
        "oi_dft":    out["orb"],
        "disp_dft":  out["disp"],
        "cpcm_dft":  out["cpcm"],
        "cds_dft":   out["smd_cds"],
        "e_bond_kcal": out["bond"],
    }


def main():
    meta = pd.read_csv(BASE / "artifacts" / "input_meta.csv")
    print(f"input_meta: {len(meta)}")

    rows, bad = [], []
    for _, m in meta.iterrows():
        rid = int(m["rxn_id"])
        d = BASE / "inputs" / f"rxn_{rid:04d}"
        try:
            # Field-source 1:1 with FSPE count check
            vals = {}
            for field, (fname, cnt) in FIELD_SOURCE.items():
                p = d / fname
                if not terminated_ok(p):
                    raise ValueError(f"{fname} not terminated")
                vals[field] = read_fspe(p, cnt)

            # EDA channels
            ch = parse_eda_channels(d / "eda.out")
            if ch is None:
                raise ValueError("EDA channel parse failed")

        except Exception as exc:
            bad.append((rid, str(exc)))
            continue

        d1 = (vals["e_frag1_dist_eh"] - vals["e_frag1_rel_eh"]) * HARTREE_TO_KCAL
        d2 = (vals["e_frag2_dist_eh"] - vals["e_frag2_rel_eh"]) * HARTREE_TO_KCAL
        inter = (vals["e_ab_eh"] - vals["e_frag1_dist_eh"] - vals["e_frag2_dist_eh"]) * HARTREE_TO_KCAL
        recon = d1 + d2 + inter
        direct = (vals["e_ab_eh"] - vals["e_frag1_rel_eh"] - vals["e_frag2_rel_eh"]) * HARTREE_TO_KCAL

        rows.append(dict(
            rxn_id=rid, **ch, **vals,
            d1_b3lyp=d1, d2_b3lyp=d2, sum_dist_b3lyp=d1 + d2,
            interaction_own=inter,
            barrier_recon=recon, barrier_direct=direct,
            closure_gap=abs(direct - recon),
            strain_valid=bool(d1 > 0 and d2 > 0),
            geometry_source="coley_r_xyz",
            reoptimized=False,
        ))

    R = pd.DataFrame(rows)
    R.to_csv(BASE / "artifacts" / "labels_b3lyp_full.csv", index=False)
    pd.DataFrame(bad, columns=["rxn_id", "reason"]).to_csv(
        BASE / "artifacts" / "parse_failures.csv", index=False)

    n_ok, n_tot = len(R), len(meta)
    rate = n_ok / max(n_tot, 1)
    print(f"parse success: {n_ok}/{n_tot} ({rate*100:.2f}%)")
    if bad:
        from collections import Counter
        by_reason = Counter(reason.split(":")[0] for _, reason in bad)
        for k, v in by_reason.most_common():
            print(f"  {k}: {v}")

    status = "PASS" if rate >= 0.95 else "FAIL"
    with open(BASE / "artifacts" / "GATE2_STATUS.txt", "w") as f:
        f.write(f"{status} parsed={n_ok}/{n_tot} rate={rate:.4f}\n")
    print(f"=== GATE-2 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
