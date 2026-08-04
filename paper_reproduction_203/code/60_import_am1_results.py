#!/usr/bin/env python3
"""
Import AM1 Gaussian .log results into Grayson-compatible barrier pkl files.

Parses all 1015 AM1 .log files (ts, gs, dist_gs) into per-rxn energies, then
computes distortion + interaction using the same formulas as the DFT labels:

  strain_1_am1     = E_AM1(d_1)   - E_AM1(R_1)         (kcal/mol)
  strain_2_am1     = E_AM1(d_2)   - E_AM1(R_2)
  distortion_am1   = strain_1 + strain_2
  barrier_am1      = E_AM1(TS)    - E_AM1(R_1) - E_AM1(R_2)
  interaction_am1  = barrier_am1  - distortion_am1

Assignment of di/dp per fragment (for per-type distortion columns) uses the
same user-manual reactant_1_type / reactant_2_type from meta.json.

Outputs:
  barriers/am1_barriers.pkl       DataFrame keyed by reaction_number
    id, e_barrier_am1, distortion_am1, interaction_am1,
    distortion_dipole_am1, distortion_dipolarophile_am1,
    strain_1_am1, strain_2_am1,
    e_ts_am1, e_R_1_am1, e_R_2_am1, e_d_1_am1, e_d_2_am1  (all kcal/mol)
  barriers/dft_barriers.pkl       DataFrame keyed by reaction_number
    id, e_barrier_dft, distortion_dft, interaction_dft,
    distortion_dipole_dft, distortion_dipolarophile_dft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cclib
import pandas as pd

HARTREE_TO_KCAL = 627.5094740631


def scf_energy_hartree(log_path: Path) -> float:
    """Return the LAST SCF energy from a Gaussian .log file (in Hartree).

    cclib returns scfenergies in eV; convert to Hartree.
    """
    d = cclib.io.ccread(str(log_path))
    e_eV = float(d.scfenergies[-1])
    return e_eV / 27.211386245988  # eV → Hartree


def process_rxn(rn: int, rid: str, logs_root: Path, geom_dir: Path) -> dict:
    meta = json.loads((geom_dir / "meta.json").read_text())

    # locate log files
    ts_log = logs_root / "ts" / f"ts_{rn}.log"
    gs_1_log = logs_root / "gs" / f"gs_{rn}_reactant_1.log"
    gs_2_log = logs_root / "gs" / f"gs_{rn}_reactant_2.log"
    d_1_log = logs_root / "dist_gs" / f"dist_{rn}_reactant_1.log"
    d_2_log = logs_root / "dist_gs" / f"dist_{rn}_reactant_2.log"

    e_ts_h = scf_energy_hartree(ts_log)
    e_R_1_h = scf_energy_hartree(gs_1_log)
    e_R_2_h = scf_energy_hartree(gs_2_log)
    e_d_1_h = scf_energy_hartree(d_1_log)
    e_d_2_h = scf_energy_hartree(d_2_log)

    # convert to kcal/mol
    e_ts = e_ts_h * HARTREE_TO_KCAL
    e_R_1 = e_R_1_h * HARTREE_TO_KCAL
    e_R_2 = e_R_2_h * HARTREE_TO_KCAL
    e_d_1 = e_d_1_h * HARTREE_TO_KCAL
    e_d_2 = e_d_2_h * HARTREE_TO_KCAL

    strain_1 = e_d_1 - e_R_1
    strain_2 = e_d_2 - e_R_2
    distortion = strain_1 + strain_2
    barrier = e_ts - e_R_1 - e_R_2
    interaction = barrier - distortion

    # per-type distortion (dipole vs dipolarophile)
    if meta["reactant_1_type"] == "dipole":
        d_dipole, d_dipolar = strain_1, strain_2
    else:
        d_dipole, d_dipolar = strain_2, strain_1

    # Note: Grayson's collate step add_suffix('_am1') and add_suffix('_dft') to
    # incoming barrier columns; therefore this function returns UNSUFFIXED
    # column names. String columns (reaction_id) are excluded — sklearn's
    # standardizer would choke on them downstream.
    return {
        "id": rn,
        "e_barrier": barrier,
        "distortion": distortion,
        "interaction": interaction,
        "distortion_dipole": d_dipole,
        "distortion_dipolarophile": d_dipolar,
        "strain_1": strain_1,
        "strain_2": strain_2,
        "e_ts": e_ts,
        "e_R_1": e_R_1,
        "e_R_2": e_R_2,
        "e_d_1": e_d_1,
        "e_d_2": e_d_2,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent.parent)
    args = ap.parse_args()

    logs_root = args.root / "gaussian_inputs" / "am1_logs"
    if not logs_root.exists():
        print(f"ERROR: {logs_root} missing — run Gaussian AM1 first", file=sys.stderr)
        return 1

    # Parse manifest
    rxns: list[tuple[int, str]] = []
    for line in (args.root / "MANIFEST.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            rxns.append((int(parts[0]), parts[1]))
    print(f"processing {len(rxns)} rxns")

    rows = []
    fails = []
    for rn, rid in rxns:
        try:
            row = process_rxn(rn, rid, logs_root, args.root / "geometries" / rid)
            rows.append(row)
        except Exception as e:
            fails.append((rid, str(e)))
            print(f"  FAIL {rid}: {e}")

    if fails:
        print(f"\n{len(fails)} FAIL(s)")
        return 1

    df_am1 = pd.DataFrame(rows)
    print(f"\nAM1 barrier ranges (kcal/mol):")
    print(f"  distortion:  min={df_am1['distortion'].min():.2f}  max={df_am1['distortion'].max():.2f}  mean={df_am1['distortion'].mean():.2f}")
    print(f"  interaction: min={df_am1['interaction'].min():.2f}  max={df_am1['interaction'].max():.2f}  mean={df_am1['interaction'].mean():.2f}")
    print(f"  barrier:     min={df_am1['e_barrier'].min():.2f}  max={df_am1['e_barrier'].max():.2f}  mean={df_am1['e_barrier'].mean():.2f}")

    # DFT barriers — Grayson's collate adds `_dft` suffix, so use unsuffixed names here
    # (and drop string columns like reaction_id that would break sklearn)
    dft_full = pd.read_parquet(args.root / "labels" / "labels_5channel_paper.parquet")
    df_dft = pd.DataFrame({
        "id":                        dft_full["reaction_number"].astype(int),
        "e_barrier":                 dft_full["e_barrier_dft"],
        "distortion":                dft_full["distortion_dft"],
        "interaction":               dft_full["interaction_dft"],
        "distortion_dipole":         dft_full["distortion_dipole"],
        "distortion_dipolarophile":  dft_full["distortion_dipolarophile"],
    })

    out_dir = args.root / "barriers"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_am1.to_pickle(out_dir / "am1_barriers.pkl")
    df_dft.to_pickle(out_dir / "dft_barriers.pkl")
    df_am1.to_csv(out_dir / "am1_barriers.csv", index=False)
    df_dft.to_csv(out_dir / "dft_barriers.csv", index=False)

    # Pre-ML MAE check (AM1 vs DFT); merge on id
    merged = df_am1.merge(df_dft, on="id", suffixes=("_am1", "_dft"))
    for col in ["e_barrier", "distortion", "interaction",
                "distortion_dipole", "distortion_dipolarophile"]:
        mae = (merged[f"{col}_am1"] - merged[f"{col}_dft"]).abs().mean()
        print(f"  pre-ML MAE {col}: {mae:.2f} kcal/mol")

    print(f"\nwrote: {out_dir}/am1_barriers.pkl ({len(df_am1)} rows)")
    print(f"wrote: {out_dir}/dft_barriers.pkl ({len(df_dft)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
