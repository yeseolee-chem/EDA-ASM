#!/usr/bin/env python3
"""Clean physics_24 — excludes disp_xtb / dispd4_xtb (oracle features).

xtb shard parquets contain these; we filter them out before feature selection.
Result: physics has <24 columns (drop d1..d24 constraint), used by Phase 4c training.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
COHORT_PATH = OUT / "cohort_subset.parquet"
XTB_SHARDS = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/xtb_channels"
MANIFEST = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data/manifest.parquet"

ORACLE_FEATURES = {"disp_xtb", "dispd4_xtb", "eint_total_xtb"}
# strain frag swap fix: xTB frag numbering is inverted vs DFT labels (audit 2026-08-21)
SWAP_PAIRS = [("strain_1_xtb", "strain_2_xtb")]


def main():
    cohort = pd.read_parquet(COHORT_PATH)
    manifest = pd.read_parquet(MANIFEST)
    shards = sorted(Path(XTB_SHARDS).glob("channels_shard*.parquet"))
    xtb = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)

    # Drop oracle columns from xtb
    for c in ORACLE_FEATURES:
        if c in xtb.columns:
            xtb = xtb.drop(columns=[c])
            print(f"  excluded oracle: {c}")
    # Fix strain frag swap
    for a, b in SWAP_PAIRS:
        if a in xtb.columns and b in xtb.columns:
            xtb = xtb.rename(columns={a: f"__TMP__{a}"})
            xtb = xtb.rename(columns={f"__TMP__{a}": b, b: a})
            print(f"  swapped: {a} ↔ {b}")
    print(f"xTB shards: {len(shards)}, rows: {len(xtb)}, cols: {len(xtb.columns)}")

    df = cohort.merge(manifest[["rxn_id", "imag_freq_cm1"]],
                       left_on="reaction_number", right_on="rxn_id", how="inner")
    df = df.drop(columns=["rxn_id"])
    df = df.merge(xtb, left_on="reaction_number", right_on="reaction_number",
                   how="left", suffixes=("", "_xtb"))

    feature_cols = ["n_atoms", "imag_freq_cm1"]
    for c in xtb.columns:
        if c == "reaction_number": continue
        if c in ORACLE_FEATURES: continue
        if df[c].dtype.kind in "biufc":
            feature_cols.append(c)

    physics = df[["reaction_number"] + feature_cols].copy()
    for c in feature_cols:
        physics[c] = physics[c].fillna(physics[c].median())

    out_path = OUT / "physics_v2.pkl"  # versioned (not overwriting old)
    physics.to_pickle(out_path)
    print(f"physics (clean): {physics.shape} → {out_path}")
    print(f"Columns: {list(physics.columns)}")


if __name__ == "__main__":
    main()
