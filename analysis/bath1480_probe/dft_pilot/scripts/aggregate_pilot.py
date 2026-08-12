#!/usr/bin/env python3
"""Aggregate pilot channels_dft.json + compare against Espley reference and
our earlier AM1-geometry EDA labels.

Writes:
  results/pilot_summary.parquet  — one row per rxn, 7 DFT-geom channels
                                    + Espley reference + AM1-geom EDA
  results/pilot_summary.md       — human-readable comparison table
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot")
WORK = BASE / "work"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

# Existing AM1-geometry EDA labels (Arm B pkl already has these merged in)
AM1_PKL = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/manual_tt_7ch_xtb.pkl")

CHANNELS = ["elst_dft", "pauli_dft", "oi_dft", "disp_dft",
            "cpcm_dft", "cds_dft", "eint_dft"]
ESPLEY_COLS = ["interaction_energies_dft", "e_barrier_dft", "q_barrier_dft",
                "sum_distortion_energies_dft", "distortion_energy_1_dft",
                "distortion_energy_2_dft"]


def load_pilot():
    rows = []
    for jf in sorted(WORK.glob("rxn_*/channels_dft.json")):
        d = json.loads(jf.read_text())
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["reaction_number"] = df["rxn_id"].astype(int)
    return df


def load_am1_labels():
    d = pd.read_pickle(AM1_PKL)
    keep = ["reaction_number"] + [c for c in CHANNELS if c in d.columns] + [c for c in ESPLEY_COLS if c in d.columns]
    return d[keep]


def main():
    dft = load_pilot()
    print(f"pilot DFT-geom rxns found: {len(dft)}")
    if dft.empty:
        print("nothing to aggregate yet")
        return

    am1 = load_am1_labels()
    # Rename AM1-geom EDA channels with _am1 suffix (so DFT vs AM1 side-by-side)
    am1_rename = {c: f"{c.replace('_dft','')}_am1_geom_dft" for c in CHANNELS if c in am1.columns}
    am1 = am1.rename(columns=am1_rename)

    m = dft.merge(am1, on="reaction_number", how="left")
    m.to_parquet(OUT_DIR / "pilot_summary.parquet", index=False)

    # Human-readable comparison
    lines = ["# Pilot summary — DFT-geometry EDA vs AM1-geometry EDA vs Espley reference",
             "", f"N pilot rxns: **{len(m)}**", "",
             "## Per-reaction interaction energies (kcal/mol)",
             "",
             "| rxn_id | n_atoms | eint_dft (DFT geom) | eint_am1_geom | Espley interaction_dft | Δ(DFT − Espley) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for _, r in m.iterrows():
        n_at = int(r.get("n_atoms_frag1", 0)) + int(r.get("n_atoms_frag2", 0))
        eint_dft_geom = r.get("eint_dft", np.nan)
        eint_am1_geom = r.get("eint_am1_geom_dft", np.nan)
        espley = r.get("interaction_energies_dft", np.nan)
        # Sign convention: Espley uses positive magnitude (dist_energy - barrier)
        # → compare to -eint_dft (our attractive convention)
        our_signed = -eint_dft_geom if pd.notna(eint_dft_geom) else np.nan
        gap = our_signed - espley if pd.notna(espley) and pd.notna(our_signed) else np.nan
        lines.append(f"| {int(r['reaction_number'])} | {n_at} | "
                     f"{eint_dft_geom:.2f} | {eint_am1_geom:.2f} | "
                     f"{espley:.2f} | {gap:+.2f} |")

    lines.extend(["", "## Per-channel comparison (DFT geom vs AM1 geom)", "",
                  "| rxn_id | channel | DFT geom | AM1 geom | Δ |",
                  "|---:|---|---:|---:|---:|"])
    for _, r in m.iterrows():
        rid = int(r["reaction_number"])
        for ch in CHANNELS:
            am1_col = f"{ch.replace('_dft','')}_am1_geom_dft"
            v_dft = r.get(ch, np.nan)
            v_am1 = r.get(am1_col, np.nan)
            delta = v_dft - v_am1 if pd.notna(v_dft) and pd.notna(v_am1) else np.nan
            lines.append(f"| {rid} | {ch} | {v_dft:.2f} | {v_am1:.2f} | {delta:+.2f} |")

    lines.extend(["", "## Sum-consistency + health flags", "",
                  "| rxn_id | ok | sum_residual (kcal) | warnings |",
                  "|---:|---|---:|---|"])
    for _, r in m.iterrows():
        rid = int(r["reaction_number"])
        ok = r.get("ok", None)
        sr = r.get("sum_residual_kcal", np.nan)
        warns = r.get("warnings", []) or []
        wtxt = "; ".join(warns) if isinstance(warns, (list, tuple, np.ndarray)) else str(warns)
        lines.append(f"| {rid} | {ok} | {sr:.4f} | {wtxt} |")

    (OUT_DIR / "pilot_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote:\n  {OUT_DIR/'pilot_summary.parquet'}\n  {OUT_DIR/'pilot_summary.md'}")


if __name__ == "__main__":
    main()
