#!/usr/bin/env python3
"""SPEC14 Step 4 (revised per precheck §1) — parse ORCA EDA outputs using the
CANONICAL parser that produced the 3504 phase5 labels.

Two behavioural requirements from the precheck:

  (A) Reuse the same parse code, not just the same regex — vendored as
      _vendor_aggregate_channels.py (byte-copy of the kisti_results parser).
  (B) Round-trip verify: parse 20 existing eda.out files from gpfs scratch
      and confirm the derived channels match phase5_dataset_v2.pkl to 1e-6.
      This is GATE-P; without it, downstream comparisons are meaningless.

Channel mapping (raw kisti → phase5, verified on rxn 0):
    e_elst_kcal                 → elst_dft
    e_pauli_kcal + e_xc_kcal    → pauli_dft   (Pauli combined with Delta E^0(XC))
    e_orb_kcal                  → oi_dft
    e_disp_kcal                 → disp_dft
    e_cpcm_kcal                 → cpcm_dft
    e_smd_cds_kcal              → cds_dft

GATE-3: ORCA normal-termination rate ≥ 95% on the 218 new calcs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the vendored canonical parser (byte-copy of kisti_results/scripts/aggregate_channels.py)
sys.path.insert(0, str(Path(__file__).parent))
from _vendor_aggregate_channels import parse_eda_out  # noqa: E402

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
EXISTING_EDA_OUT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")

# Sign convention (verified: 3504/3504 rows in phase5 obey these)
SIGN_CHECK = {
    "elst_dft": "<0",
    "pauli_dft": ">0",
    "oi_dft": "<0",
    "disp_dft": "<0",
}


def kisti_to_phase5(raw: dict) -> dict:
    """Map raw kisti channel dict to phase5-style keys."""
    if raw is None:
        return None
    out = {
        "elst_dft": raw.get("e_elst_kcal"),
        "pauli_dft": (raw["e_pauli_kcal"] + raw["e_xc_kcal"])
                       if "e_pauli_kcal" in raw and "e_xc_kcal" in raw else None,
        "oi_dft": raw.get("e_orb_kcal"),
        "disp_dft": raw.get("e_disp_kcal"),
        "cpcm_dft": raw.get("e_cpcm_kcal"),
        "cds_dft": raw.get("e_smd_cds_kcal"),
        # useful extras for closure checks
        "e_bond_kcal": raw.get("e_bond_kcal"),
        "channel_sum": None,
    }
    if all(out[c] is not None for c in ("elst_dft", "pauli_dft", "oi_dft",
                                         "disp_dft", "cpcm_dft", "cds_dft")):
        out["channel_sum"] = (out["elst_dft"] + out["pauli_dft"] + out["oi_dft"]
                              + out["disp_dft"] + out["cpcm_dft"] + out["cds_dft"])
    return out


def check_signs(row: dict) -> list:
    """Return list of violation messages for phase5-style row."""
    v = []
    for c, cond in SIGN_CHECK.items():
        val = row.get(c)
        if val is None or np.isnan(val):
            continue
        if cond == "<0" and val >= 0:
            v.append(f"{c}={val:.3f} !< 0")
        elif cond == ">0" and val <= 0:
            v.append(f"{c}={val:.3f} !> 0")
    return v


def gate_p_roundtrip() -> tuple[str, list]:
    """Parse 20 existing eda.out files, verify against phase5 labels to 1e-6.

    Returns: (status, list of diagnostic lines).
    """
    p5 = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    p5["reaction_number"] = p5["reaction_number"].astype(int)
    p5 = p5.set_index("reaction_number")

    check_rxns = sorted(p5.index.tolist())[:20]
    rows = []
    fails = []
    for rid in check_rxns:
        eda_out = EXISTING_EDA_OUT / f"rxn_{rid:04d}" / "eda.out"
        if not eda_out.exists():
            fails.append(f"rxn {rid}: eda.out missing at {eda_out}")
            continue
        raw = parse_eda_out(eda_out)
        if raw is None:
            fails.append(f"rxn {rid}: parse_eda_out returned None")
            continue
        derived = kisti_to_phase5(raw)
        row = {"rxn_id": rid}
        max_abs_diff = 0.0
        for ch in ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"):
            derived_v = derived[ch]
            label_v = float(p5.at[rid, ch])
            diff = abs(derived_v - label_v)
            row[f"{ch}_derived"] = derived_v
            row[f"{ch}_label"] = label_v
            row[f"{ch}_diff"] = diff
            max_abs_diff = max(max_abs_diff, diff)
        row["max_abs_diff"] = max_abs_diff
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(BASE / "artifacts" / "parser_roundtrip.csv", index=False)

    lines = []
    if not len(df):
        status = "FAIL"
        lines.append("no eda.out files parseable")
    else:
        worst = df["max_abs_diff"].max()
        n_ok = int((df["max_abs_diff"] < 1e-6).sum())
        lines.append(f"n_ok(diff<1e-6): {n_ok}/{len(df)}")
        lines.append(f"worst max_abs_diff: {worst:.6e} kcal/mol")
        if worst < 1e-6:
            status = "PASS"
        else:
            status = "FAIL"
            lines.append("bad rows:")
            bad = df[df["max_abs_diff"] >= 1e-6].head(5)
            for _, r in bad.iterrows():
                lines.append(f"  rxn {int(r['rxn_id'])}: max_diff={r['max_abs_diff']:.3e}")
    for f in fails:
        lines.append(f"  {f}")

    with open(BASE / "artifacts" / "GATEP_STATUS.txt", "w") as f:
        f.write(f"{status}\n")
        for ln in lines:
            f.write(ln + "\n")
    return status, lines


def parse_new_outputs() -> tuple[int, int, list]:
    """Parse the 218 new B3LYP-geometry EDA outputs. Returns (n_ok, n_total, rows)."""
    input_dirs = sorted((BASE / "inputs").glob("rxn_*"))
    rows = []
    fails = []
    for d in input_dirs:
        rid = int(d.name.split("_")[1])
        eda_out = d / "eda.out"
        raw = parse_eda_out(eda_out) if eda_out.exists() else None
        derived = kisti_to_phase5(raw)
        if derived is None:
            fails.append(rid)
            continue
        row = {"rxn_id": rid, **derived}
        row["sign_violations"] = ";".join(check_signs(row)) or ""
        rows.append(row)
    return len(rows), len(input_dirs), rows, fails


def main():
    print("=== GATE-P: round-trip parser verification (20 existing eda.out) ===")
    gp_status, gp_lines = gate_p_roundtrip()
    for ln in gp_lines:
        print("  " + ln)
    print(f"=== GATE-P {gp_status} ===")
    if gp_status != "PASS":
        print("STOP: parser round-trip failed. Do not proceed to new-output parsing.")
        raise SystemExit(1)

    print("\n=== Parse new B3LYP-geometry outputs ===")
    n_ok, n_total, rows, fails = parse_new_outputs()
    rate = n_ok / max(n_total, 1)
    print(f"parsed: {n_ok}/{n_total} ({rate*100:.1f}%)")
    if fails:
        print(f"failures: {len(fails)}   first 10 rxn_ids: {fails[:10]}")

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(BASE / "artifacts" / "channels_b3lyp.csv", index=False)
        n_sign = int((df["sign_violations"] != "").sum())
        print(f"sign-convention violations: {n_sign}/{len(df)}")
        if n_sign:
            sub = df[df["sign_violations"] != ""].head(10)
            for _, r in sub.iterrows():
                print(f"  rxn {int(r['rxn_id'])}: {r['sign_violations']}")
        # 6-channel-sum closure vs bond
        closed = df.dropna(subset=["channel_sum", "e_bond_kcal"])
        if len(closed):
            gap = (closed["channel_sum"] - closed["e_bond_kcal"]).abs()
            print(f"closure |Σch − bond| (kcal/mol): "
                  f"median={gap.median():.4f} max={gap.max():.4f} p95={gap.quantile(0.95):.4f}")

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
