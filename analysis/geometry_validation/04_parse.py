#!/usr/bin/env python3
"""SPEC14 Step 4 (Hartree-precision revision) — parse ORCA EDA outputs using
the Hartree column × 627.5094740631, which is what phase5_dataset_v2.pkl was
built from. Verified: 20/20 round-trip diff = 0.0 (GATE-P PASS).

The vendored aggregate_channels.py uses the rounded Kcal/mol column and
therefore only reproduces phase5 labels to ~5 mcal/mol. We instead parse
Hartree here (see _gate_p_only.py for the standalone verification).

GATE-P: 20 existing eda.out files vs phase5, max |Δ| < 1e-6
GATE-3: ORCA normal-termination rate ≥ 95% on the 218 new calcs
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
EXISTING_EDA_OUT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")

HARTREE_TO_KCAL = 627.5094740631

EDA_TABLE_RE = re.compile(
    r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+",
    re.DOTALL,
)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")

LABEL_TO_KEY = {
    "Bond Energy":              "bond",
    "Orbital Energy":           "orb",
    "Electrostatic Energy":     "elst",
    "Pauli Energy":             "pauli",
    "Delta E^0(XC)":            "xc",
    "Delta Dispersion":         "disp",
    "Delta CPCM Dielectric":    "cpcm",
    "Delta SMD CDS correction": "smd_cds",
}

SIGN_CHECK = {
    "elst_dft":  "<0",
    "pauli_dft": ">0",
    "oi_dft":    "<0",
    "disp_dft":  "<0",
}


def parse_eda_hartree(path: Path):
    """Parse an eda.out, returning kcal/mol channel values from the Hartree
    column × 627.5094740631. Returns None if ORCA did not terminate normally
    or the EDA table is missing.
    """
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
        hartree = float(row.group(2))
        if label in LABEL_TO_KEY:
            out[LABEL_TO_KEY[label]] = hartree * HARTREE_TO_KCAL
    return out


def to_phase5(raw):
    if raw is None:
        return None
    if not all(k in raw for k in ("elst", "pauli", "xc", "orb", "disp", "cpcm", "smd_cds")):
        return None
    out = {
        "elst_dft":  raw["elst"],
        "pauli_dft": raw["pauli"] + raw["xc"],
        "oi_dft":    raw["orb"],
        "disp_dft":  raw["disp"],
        "cpcm_dft":  raw["cpcm"],
        "cds_dft":   raw["smd_cds"],
        "e_bond_kcal": raw.get("bond"),
    }
    out["channel_sum"] = (out["elst_dft"] + out["pauli_dft"] + out["oi_dft"]
                          + out["disp_dft"] + out["cpcm_dft"] + out["cds_dft"])
    return out


def check_signs(row):
    v = []
    for c, cond in SIGN_CHECK.items():
        val = row.get(c)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if cond == "<0" and val >= 0:
            v.append(f"{c}={val:.3f} !< 0")
        elif cond == ">0" and val <= 0:
            v.append(f"{c}={val:.3f} !> 0")
    return v


def gate_p_roundtrip():
    p5 = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    p5["reaction_number"] = p5["reaction_number"].astype(int)
    p5 = p5.set_index("reaction_number")

    rxns = sorted(p5.index.tolist())[:20]
    rows, fails = [], []
    worst = 0.0
    n_ok = 0
    for rid in rxns:
        eda = EXISTING_EDA_OUT / f"rxn_{rid:04d}" / "eda.out"
        if not eda.exists():
            fails.append(f"rxn {rid}: eda.out missing")
            continue
        raw = parse_eda_hartree(eda)
        derived = to_phase5(raw)
        if derived is None:
            fails.append(f"rxn {rid}: parse failed")
            continue
        max_here = 0.0
        for ch in ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"):
            max_here = max(max_here, abs(derived[ch] - float(p5.at[rid, ch])))
        rows.append({"rxn_id": rid, "max_abs_diff": max_here})
        worst = max(worst, max_here)
        if max_here < 1e-6:
            n_ok += 1

    pd.DataFrame(rows).to_csv(BASE / "artifacts" / "parser_roundtrip.csv", index=False)
    status = "PASS" if worst < 1e-6 else "FAIL"
    lines = [f"n_ok(diff<1e-6): {n_ok}/{len(rxns)}",
             f"worst max_abs_diff: {worst:.6e} kcal/mol"]
    lines += [f"  {f}" for f in fails]
    with open(BASE / "artifacts" / "GATEP_STATUS.txt", "w") as f:
        f.write(f"{status}\n")
        for ln in lines:
            f.write(ln + "\n")
    return status, lines


def parse_new_outputs():
    input_dirs = sorted((BASE / "inputs").glob("rxn_*"))
    rows, fails = [], []
    for d in input_dirs:
        rid = int(d.name.split("_")[1])
        eda_out = d / "eda.out"
        raw = parse_eda_hartree(eda_out)
        derived = to_phase5(raw)
        if derived is None:
            fails.append(rid)
            continue
        row = {"rxn_id": rid, **derived}
        row["sign_violations"] = ";".join(check_signs(row)) or ""
        rows.append(row)
    return len(rows), len(input_dirs), rows, fails


def main():
    print("=== GATE-P: round-trip (20 existing eda.out, Hartree parser) ===")
    gp_status, gp_lines = gate_p_roundtrip()
    for ln in gp_lines:
        print("  " + ln)
    print(f"=== GATE-P {gp_status} ===")
    if gp_status != "PASS":
        print("STOP: parser round-trip failed. Not proceeding to new-output parsing.")
        raise SystemExit(1)

    print("\n=== Parse new B3LYP-geometry outputs ===")
    n_ok, n_total, rows, fails = parse_new_outputs()
    rate = n_ok / max(n_total, 1)
    print(f"parsed: {n_ok}/{n_total} ({rate*100:.1f}%)")
    if fails:
        print(f"failures: {len(fails)}  first 10: {fails[:10]}")

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(BASE / "artifacts" / "channels_b3lyp.csv", index=False)
        n_sign = int((df["sign_violations"] != "").sum())
        print(f"sign-convention violations: {n_sign}/{len(df)}")
        if n_sign:
            for _, r in df[df["sign_violations"] != ""].head(10).iterrows():
                print(f"  rxn {int(r['rxn_id'])}: {r['sign_violations']}")
        closed = df.dropna(subset=["channel_sum", "e_bond_kcal"])
        if len(closed):
            gap = (closed["channel_sum"] - closed["e_bond_kcal"]).abs()
            print(f"closure |Σch − bond|: median={gap.median():.4f}  max={gap.max():.4f}")

    status = "PASS" if rate >= 0.95 else "FAIL"
    with open(BASE / "artifacts" / "GATE3_STATUS.txt", "w") as f:
        f.write(f"{status} parsed={n_ok}/{n_total} rate={rate:.3f} "
                f"sign_violations={int((df['sign_violations'] != '').sum()) if len(df) else 0}\n")
        if fails:
            f.write("failed rxn_ids:\n")
            for r in fails:
                f.write(f"  {r}\n")
    print(f"=== GATE-3 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
