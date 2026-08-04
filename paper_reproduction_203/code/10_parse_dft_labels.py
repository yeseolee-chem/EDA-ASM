#!/usr/bin/env python3
"""
Parse ORCA outputs (spec23_paper_setup) into paper-compliant DIAS labels.

Uses Espley 2024 (Digital Discovery 3:2479) DIAS formula VERBATIM:
    strain_1        = E(spe_d1) − E(spe_R_1)
    strain_2        = E(spe_d2) − E(spe_R_2)
    distortion_dft  = strain_1 + strain_2
    barrier_dft     = E(TS)     − E(spe_R_1) − E(spe_R_2)
    interaction_dft = barrier_dft − distortion_dft            (simple subtraction)

All 5 SPEs in CPCM(water) at wB97X-D3BJ/def2-TZVP (ORCA 6.1.1).

E(TS) is read from eda.out (final SCF energy of the TS complex), NOT from ORCA
EDA-NOCV "Bond Energy" which is a fundamentally different quantity and gave
non-physical negative barriers in prior label runs (see D2 diagnostic).

The 5-channel EDA-NOCV decomposition (Pauli/elstat/orb/xc/disp/CPCM) is
extracted for audit only, into labels_5channel_audit_paper.parquet. It is NOT
used as a target — paper's interaction is a scalar subtraction, not a sum of
EDA channels.

Fragment 1 (=ORCA eda.inp FRAG1) vs Fragment 2 identity is resolved via
meta.json's orca_filename_swap flag written by 20_prepare_geometries.py.

Outputs:
  labels/labels_2ch_paper.parquet          reaction_number, reaction_id,
                                           distortion_dft, interaction_dft, e_barrier_dft
  labels/labels_2ch_paper.csv              same, CSV
  labels/labels_5channel_audit_paper.parquet  5 EDA channels for audit ONLY
  labels/label_schema.json                 protocol + ranges + provenance
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HARTREE_TO_KCAL = 627.5094740631

RE_FINAL_SPE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
RE_EDA_LINE = re.compile(
    r"^\s*(Bond Energy|Orbital Energy|Electrostatic Energy|Pauli Energy|"
    r"Delta E\^0\(XC\)|Delta Dispersion|Delta CPCM Dielectric)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)
EDA_KEYS = {
    "Bond Energy": "bond_kcal",
    "Orbital Energy": "orb_kcal",
    "Electrostatic Energy": "elstat_kcal",
    "Pauli Energy": "pauli_kcal",
    "Delta E^0(XC)": "xc_kcal",
    "Delta Dispersion": "disp_kcal",
    "Delta CPCM Dielectric": "cpcm_kcal",
}


def parse_final_spe(out_path: Path) -> float:
    text = out_path.read_text()
    matches = RE_FINAL_SPE.findall(text)
    if not matches:
        raise ValueError(f"no FINAL SINGLE POINT ENERGY in {out_path}")
    return float(matches[-1])


def parse_eda_channels(eda_out_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in eda_out_path.read_text().splitlines():
        m = RE_EDA_LINE.match(line)
        if m:
            out[EDA_KEYS[m.group(1)]] = float(m.group(3))
    missing = set(EDA_KEYS.values()) - set(out.keys())
    if missing:
        raise ValueError(f"missing EDA channels in {eda_out_path}: {missing}")
    return out


def parse_one_rxn(rid: str, workdir: Path, geom_root: Path) -> dict:
    rxn_dir = workdir / rid
    meta_path = geom_root / rid / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing — run 20_prepare_geometries.py first")
    meta = json.loads(meta_path.read_text())

    swap = meta["orca_filename_swap"]
    if swap == "no_swap":
        R_1_out = rxn_dir / "spe_R_A" / "spe_R_A.out"
        R_2_out = rxn_dir / "spe_R_B" / "spe_R_B.out"
    else:
        R_1_out = rxn_dir / "spe_R_B" / "spe_R_B.out"
        R_2_out = rxn_dir / "spe_R_A" / "spe_R_A.out"

    d_1_out = rxn_dir / "cpcm_extra" / "spe_dA.out"
    d_2_out = rxn_dir / "cpcm_extra" / "spe_dB.out"
    eda_out = rxn_dir / "eda" / "eda.out"

    # Energies (Hartree → kcal/mol)
    e_R_1 = parse_final_spe(R_1_out) * HARTREE_TO_KCAL
    e_R_2 = parse_final_spe(R_2_out) * HARTREE_TO_KCAL
    e_d_1 = parse_final_spe(d_1_out) * HARTREE_TO_KCAL
    e_d_2 = parse_final_spe(d_2_out) * HARTREE_TO_KCAL
    e_TS  = parse_final_spe(eda_out) * HARTREE_TO_KCAL

    # Paper DIAS formula (verbatim)
    strain_1        = e_d_1 - e_R_1
    strain_2        = e_d_2 - e_R_2
    distortion_dft  = strain_1 + strain_2
    e_barrier_dft   = e_TS - e_R_1 - e_R_2
    interaction_dft = e_barrier_dft - distortion_dft

    # Per-type distortion (dipole vs dipolarophile)
    r1_type = meta["reactant_1_type"]
    r2_type = meta["reactant_2_type"]
    if r1_type == "dipole":
        distortion_dipole, distortion_dipolarophile = strain_1, strain_2
    elif r1_type == "dipolarophile":
        distortion_dipole, distortion_dipolarophile = strain_2, strain_1
    else:
        raise ValueError(f"{rid}: unexpected reactant_1_type {r1_type}")

    # 5-channel EDA decomposition — AUDIT ONLY, not a target
    eda_ch = parse_eda_channels(eda_out)
    audit = {
        "reaction_id": rid,
        "reaction_number": meta["reaction_number"],
        "bond_kcal_orca":    eda_ch["bond_kcal"],
        "pauli_kcal_orca":   eda_ch["pauli_kcal"],
        "elstat_kcal_orca":  eda_ch["elstat_kcal"],
        "orb_kcal_orca":     eda_ch["orb_kcal"],
        "xc_kcal_orca":      eda_ch["xc_kcal"],
        "disp_kcal_orca":    eda_ch["disp_kcal"],
        "cpcm_kcal_orca":    eda_ch["cpcm_kcal"],
        "diff_bond_vs_interaction_dias": eda_ch["bond_kcal"] - interaction_dft,
    }

    label = {
        "reaction_id": rid,
        "reaction_number": meta["reaction_number"],
        "distortion_dft":  distortion_dft,
        "interaction_dft": interaction_dft,
        "e_barrier_dft":   e_barrier_dft,
        "distortion_dipole": distortion_dipole,
        "distortion_dipolarophile": distortion_dipolarophile,
        "strain_1_kcal": strain_1,
        "strain_2_kcal": strain_2,
        "reactant_1_type": r1_type,
        "reactant_2_type": r2_type,
        "e_R_1_hartree": e_R_1 / HARTREE_TO_KCAL,
        "e_R_2_hartree": e_R_2 / HARTREE_TO_KCAL,
        "e_d_1_hartree": e_d_1 / HARTREE_TO_KCAL,
        "e_d_2_hartree": e_d_2 / HARTREE_TO_KCAL,
        "e_TS_hartree":  e_TS  / HARTREE_TO_KCAL,
        "orca_filename_swap": swap,
    }
    return label, audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path,
                    default=Path("/gpfs/tmp_cpu2/yeseo1ee/spec23_paper_setup"))
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent.parent / "MANIFEST.txt")
    ap.add_argument("--geom-root", type=Path,
                    default=Path(__file__).parent.parent / "geometries")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "labels")
    args = ap.parse_args()

    rxns = []
    for line in args.manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rxns.append(parts[1])
    print(f"parsing DFT labels via paper DIAS formula for {len(rxns)} rxns")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels, audits = [], []
    fails = []
    for rid in rxns:
        try:
            label, audit = parse_one_rxn(rid, args.workdir, args.geom_root)
            labels.append(label)
            audits.append(audit)
        except Exception as e:
            fails.append((rid, str(e)))
            print(f"  FAIL {rid}: {e}")

    if fails:
        print(f"\n{len(fails)} FAIL(s), aborting")
        return 1

    df = pd.DataFrame(labels)
    df_audit = pd.DataFrame(audits)

    cols = [
        "reaction_number", "reaction_id",
        "distortion_dft", "interaction_dft", "e_barrier_dft",
        "distortion_dipole", "distortion_dipolarophile",
        "strain_1_kcal", "strain_2_kcal", "reactant_1_type", "reactant_2_type",
        "e_R_1_hartree", "e_R_2_hartree", "e_d_1_hartree", "e_d_2_hartree", "e_TS_hartree",
        "orca_filename_swap",
    ]
    df = df[cols].sort_values("reaction_number").reset_index(drop=True)

    # Sanity check
    n = len(df)
    n_pos = (df["e_barrier_dft"] > 0).sum()
    print(f"\n=== DIAS barrier statistics (n={n}) ===")
    print(f"  positive barrier:  {n_pos}/{n} ({100*n_pos/n:.1f}%)")
    print(f"  barrier mean:      {df['e_barrier_dft'].mean():+.2f}  range [{df['e_barrier_dft'].min():+.2f}, {df['e_barrier_dft'].max():+.2f}]")
    print(f"  distortion mean:   {df['distortion_dft'].mean():+.2f}")
    print(f"  interaction mean:  {df['interaction_dft'].mean():+.2f}")

    df_2ch = df[["reaction_number", "reaction_id",
                 "distortion_dft", "interaction_dft", "e_barrier_dft"]]
    df_2ch.to_parquet(args.out_dir / "labels_2ch_paper.parquet", index=False)
    df_2ch.to_csv(args.out_dir / "labels_2ch_paper.csv", index=False)
    df.to_parquet(args.out_dir / "labels_5channel_paper.parquet", index=False)
    df_audit.to_parquet(args.out_dir / "labels_5channel_audit_paper.parquet", index=False)

    meta = {
        "count": int(n),
        "formula": "paper DIAS (Espley 2024): interaction = barrier - distortion (simple subtraction)",
        "protocol": "wB97X-D3BJ/def2-TZVP + CPCM(water) SPE (ORCA 6.1.1)",
        "positive_barrier_frac": float(n_pos) / n,
        "ranges_kcal": {
            "distortion_dft":           [float(df["distortion_dft"].min()),  float(df["distortion_dft"].max())],
            "interaction_dft":          [float(df["interaction_dft"].min()), float(df["interaction_dft"].max())],
            "e_barrier_dft":            [float(df["e_barrier_dft"].min()),   float(df["e_barrier_dft"].max())],
            "distortion_dipole":        [float(df["distortion_dipole"].min()),        float(df["distortion_dipole"].max())],
            "distortion_dipolarophile": [float(df["distortion_dipolarophile"].min()), float(df["distortion_dipolarophile"].max())],
        },
        "note_on_eda_bond_energy": "ORCA EDA-NOCV Bond Energy (previously used as 'interaction') gave non-physical barriers due to double-counting / CPCM handling ambiguity. Paper's DIAS interaction = barrier - distortion (subtraction) is used instead. 5-channel EDA output preserved in labels_5channel_audit_paper.parquet for audit only.",
    }
    (args.out_dir / "label_schema.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote:")
    print(f"  {args.out_dir}/labels_2ch_paper.parquet  ({len(df_2ch)} rows)")
    print(f"  {args.out_dir}/labels_5channel_paper.parquet  ({n} rows, full audit)")
    print(f"  {args.out_dir}/labels_5channel_audit_paper.parquet  ({n} rows, EDA 5-ch)")
    print(f"  {args.out_dir}/label_schema.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
