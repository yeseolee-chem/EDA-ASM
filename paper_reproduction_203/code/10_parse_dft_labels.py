#!/usr/bin/env python3
"""
Parse ORCA outputs from spec23_paper_setup/ into 2-channel DFT labels.

Uses geometries/dipolar_XXXX/meta.json (produced by 20_prepare_geometries.py) to
correctly identify which ORCA .out file corresponds to reactant_1 (FRAG1) vs
reactant_2 (FRAG2), because for ~50 rxns the ORCA filename naming (spe_R_A vs
spe_R_B) does not match the actual fragment content.

Reads for each rxn:
  spe_R_A/spe_R_A.out    → FINAL SPE  (may hold FRAG1 or FRAG2 depending on swap)
  spe_R_B/spe_R_B.out    → FINAL SPE
  cpcm_extra/spe_dA.out  → FINAL SPE  (always FRAG1 distorted, per eda.inp tags)
  cpcm_extra/spe_dB.out  → FINAL SPE  (always FRAG2 distorted)
  eda/eda.out            → 5-channel EDA-NOCV (Pauli, Orb, Elstat, XC, Disp, CPCM) + Bond Energy

Outputs (all in kcal/mol):
  labels/labels_2ch_paper.parquet          Grayson 2-channel targets
    distortion_dft, interaction_dft, e_barrier_dft
  labels/labels_2ch_paper.csv              same (CSV)
  labels/labels_5channel_paper.parquet     full audit
    distortion_dipole, distortion_dipolarophile,
    pauli_kcal, elstat_kcal, orb_kcal, xc_kcal, disp_kcal, cpcm_kcal,
    strain_1_kcal, strain_2_kcal (matched to correct fragment)
    e_R_1_hartree, e_R_2_hartree, e_d_1_hartree, e_d_2_hartree
    bond_vs_sum_residual_kcal

Definitions (paper Espley 2024, ds3 reproduction):
  distortion_dft            = strain_1 + strain_2               = total distortion
  distortion_dipole         = strain of the fragment that is dipole
  distortion_dipolarophile  = strain of the fragment that is dipolarophile
  interaction_dft           = pauli + elstat + orb + xc + disp + cpcm (kcal/mol)
                              = "Bond Energy" as reported by ORCA EDA-NOCV
  e_barrier_dft             = distortion_dft + interaction_dft
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
            key = EDA_KEYS[m.group(1)]
            out[key] = float(m.group(3))
    missing = set(EDA_KEYS.values()) - set(out.keys())
    if missing:
        raise ValueError(f"missing EDA channels in {eda_out_path}: {missing}")
    return out


def parse_one_rxn(rid: str, workdir: Path, geom_root: Path) -> dict:
    rxn_dir = workdir / rid

    # Load meta.json produced by 20_prepare_geometries.py — tells us which ORCA
    # file has FRAG1 relaxed vs FRAG2 relaxed (may be swapped from filename)
    meta_path = geom_root / rid / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing — run 20_prepare_geometries.py first"
        )
    meta = json.loads(meta_path.read_text())

    swap = meta["orca_filename_swap"]  # 'no_swap' or 'swap'
    if swap == "no_swap":
        R_1_out = rxn_dir / "spe_R_A" / "spe_R_A.out"
        R_2_out = rxn_dir / "spe_R_B" / "spe_R_B.out"
    else:
        R_1_out = rxn_dir / "spe_R_B" / "spe_R_B.out"  # actual FRAG1 relaxed
        R_2_out = rxn_dir / "spe_R_A" / "spe_R_A.out"

    # spe_dA is always FRAG1 distorted, spe_dB is always FRAG2 distorted
    d_1_out = rxn_dir / "cpcm_extra" / "spe_dA.out"
    d_2_out = rxn_dir / "cpcm_extra" / "spe_dB.out"

    e_R_1_h = parse_final_spe(R_1_out)
    e_R_2_h = parse_final_spe(R_2_out)
    e_d_1_h = parse_final_spe(d_1_out)
    e_d_2_h = parse_final_spe(d_2_out)
    eda = parse_eda_channels(rxn_dir / "eda" / "eda.out")

    strain_1_kcal = (e_d_1_h - e_R_1_h) * HARTREE_TO_KCAL
    strain_2_kcal = (e_d_2_h - e_R_2_h) * HARTREE_TO_KCAL
    distortion_total_kcal = strain_1_kcal + strain_2_kcal

    # Assign strain by type (dipole vs dipolarophile)
    r1_type = meta["reactant_1_type"]  # 'dipole' or 'dipolarophile'
    r2_type = meta["reactant_2_type"]
    if r1_type == "dipole":
        distortion_dipole = strain_1_kcal
        distortion_dipolarophile = strain_2_kcal
    elif r1_type == "dipolarophile":
        distortion_dipole = strain_2_kcal
        distortion_dipolarophile = strain_1_kcal
    else:
        raise ValueError(f"{rid}: unexpected reactant_1_type {r1_type}")

    interaction_kcal = eda["bond_kcal"]
    interaction_sum = (
        eda["orb_kcal"] + eda["elstat_kcal"] + eda["pauli_kcal"]
        + eda["xc_kcal"] + eda["disp_kcal"] + eda["cpcm_kcal"]
    )
    bond_residual = abs(interaction_sum - interaction_kcal)

    barrier_kcal = distortion_total_kcal + interaction_kcal

    return {
        "reaction_id": rid,
        "reaction_number": meta["reaction_number"],
        # PAPER 2-CHANNEL TARGETS
        "distortion_dft": distortion_total_kcal,
        "interaction_dft": interaction_kcal,
        "e_barrier_dft": barrier_kcal,
        # PAPER 3-target audit (per-fragment)
        "distortion_dipole": distortion_dipole,
        "distortion_dipolarophile": distortion_dipolarophile,
        # 5-channel audit
        "pauli_kcal": eda["pauli_kcal"],
        "elstat_kcal": eda["elstat_kcal"],
        "orb_kcal": eda["orb_kcal"],
        "xc_kcal": eda["xc_kcal"],
        "disp_kcal": eda["disp_kcal"],
        "cpcm_kcal": eda["cpcm_kcal"],
        # per-fragment strain (correctly paired via meta.json)
        "strain_1_kcal": strain_1_kcal,
        "strain_2_kcal": strain_2_kcal,
        "reactant_1_type": r1_type,
        "reactant_2_type": r2_type,
        # raw energies
        "e_R_1_hartree": e_R_1_h,
        "e_R_2_hartree": e_R_2_h,
        "e_d_1_hartree": e_d_1_h,
        "e_d_2_hartree": e_d_2_h,
        # consistency
        "bond_vs_sum_residual_kcal": bond_residual,
        "orca_filename_swap": swap,
    }


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

    rxns: list[str] = []
    for line in args.manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rxns.append(parts[1])
    print(f"loaded {len(rxns)} rxns from manifest")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, fails = [], []
    for rid in rxns:
        try:
            rows.append(parse_one_rxn(rid, args.workdir, args.geom_root))
        except Exception as e:
            fails.append((rid, str(e)))
            print(f"  FAIL {rid}: {e}")

    if fails:
        print(f"\n{len(fails)} FAIL(s), aborting")
        return 1

    df = pd.DataFrame(rows)
    cols = [
        "reaction_number", "reaction_id",
        "distortion_dft", "interaction_dft", "e_barrier_dft",
        "distortion_dipole", "distortion_dipolarophile",
        "pauli_kcal", "elstat_kcal", "orb_kcal", "xc_kcal", "disp_kcal", "cpcm_kcal",
        "strain_1_kcal", "strain_2_kcal", "reactant_1_type", "reactant_2_type",
        "e_R_1_hartree", "e_R_2_hartree", "e_d_1_hartree", "e_d_2_hartree",
        "bond_vs_sum_residual_kcal", "orca_filename_swap",
    ]
    df = df[cols].sort_values("reaction_number").reset_index(drop=True)

    max_bond_resid = df["bond_vs_sum_residual_kcal"].max()
    print(f"\nmax |Bond Energy - sum(channels)| residual: {max_bond_resid:.4f} kcal/mol")
    if max_bond_resid > 0.5:
        print(f"WARN: residual > 0.5 kcal/mol — check ORCA EDA parsing")

    print(f"\ndistortion_dft (total):      min={df['distortion_dft'].min():.2f}  max={df['distortion_dft'].max():.2f}  mean={df['distortion_dft'].mean():.2f}  n={len(df)}")
    print(f"distortion_dipole:           min={df['distortion_dipole'].min():.2f}  max={df['distortion_dipole'].max():.2f}  mean={df['distortion_dipole'].mean():.2f}")
    print(f"distortion_dipolarophile:    min={df['distortion_dipolarophile'].min():.2f}  max={df['distortion_dipolarophile'].max():.2f}  mean={df['distortion_dipolarophile'].mean():.2f}")
    print(f"interaction_dft:             min={df['interaction_dft'].min():.2f}  max={df['interaction_dft'].max():.2f}  mean={df['interaction_dft'].mean():.2f}")
    print(f"e_barrier_dft:               min={df['e_barrier_dft'].min():.2f}  max={df['e_barrier_dft'].max():.2f}  mean={df['e_barrier_dft'].mean():.2f}")
    n_swap = (df["orca_filename_swap"] == "swap").sum()
    print(f"\nORCA filename swap resolved: {n_swap} / {len(df)}")

    df_2ch = df[["reaction_number", "reaction_id",
                 "distortion_dft", "interaction_dft", "e_barrier_dft"]]

    out_parquet = args.out_dir / "labels_2ch_paper.parquet"
    out_csv = args.out_dir / "labels_2ch_paper.csv"
    out_full = args.out_dir / "labels_5channel_paper.parquet"

    df_2ch.to_parquet(out_parquet, index=False)
    df_2ch.to_csv(out_csv, index=False)
    df.to_parquet(out_full, index=False)

    print(f"\nwrote:")
    print(f"  {out_parquet}  ({len(df_2ch)} rows)")
    print(f"  {out_csv}")
    print(f"  {out_full}  ({len(df)} rows, {len(df.columns)} cols)")

    meta = {
        "count": int(len(df)),
        "protocol": "wB97X-D3BJ/def2-TZVP + CPCM(water) SPE + EDA-NOCV (ORCA 6.1.1)",
        "max_bond_residual_kcal": float(max_bond_resid),
        "orca_filename_swap_count": int(n_swap),
        "ranges_kcal": {
            "distortion_dft":           [float(df["distortion_dft"].min()),           float(df["distortion_dft"].max())],
            "distortion_dipole":        [float(df["distortion_dipole"].min()),        float(df["distortion_dipole"].max())],
            "distortion_dipolarophile": [float(df["distortion_dipolarophile"].min()), float(df["distortion_dipolarophile"].max())],
            "interaction_dft":          [float(df["interaction_dft"].min()),          float(df["interaction_dft"].max())],
            "e_barrier_dft":            [float(df["e_barrier_dft"].min()),            float(df["e_barrier_dft"].max())],
        },
    }
    (args.out_dir / "label_schema.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
