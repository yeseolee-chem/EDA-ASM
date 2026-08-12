#!/usr/bin/env python3
"""Phase 4 preparation: compute d1..d24 physics descriptors (m3 spec).

Reuses existing xTB channel outputs where possible. Writes to physics_24.pkl
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
COHORT_PATH = OUT / "cohort_subset.parquet"
XTB_SHARDS = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/xtb_channels"
MANIFEST = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data/manifest.parquet"


def main():
    cohort = pd.read_parquet(COHORT_PATH)
    manifest = pd.read_parquet(MANIFEST)

    # Load xTB shard concatenation
    shards = sorted(Path(XTB_SHARDS).glob("channels_shard*.parquet"))
    xtb = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    print(f"xTB shards: {len(shards)}, rows: {len(xtb)}")
    print(f"xTB cols: {list(xtb.columns)[:20]}")

    # Build physics_24: (this is a placeholder — actual d1..d24 mapping requires
    # tracking m3 spec exact definitions. For initial run, we use a proxy set:
    #   d1..d6:  n_atoms, mean bond distance (from manifest), imag_freq_cm1
    #   d7..d15: xTB energies, dipoles, HOMO/LUMO for complex + fragments
    #   d16..d24: xTB derived quantities from strain outputs
    # cohort already has n_atoms; merge only imag_freq_cm1 from manifest
    df = cohort.merge(manifest[["rxn_id", "imag_freq_cm1"]],
                       left_on="reaction_number", right_on="rxn_id", how="inner")
    df = df.drop(columns=["rxn_id"])
    df = df.merge(xtb, left_on="reaction_number", right_on="reaction_number", how="left", suffixes=("", "_xtb"))

    # Compact 24-d feature block (fallback: reuse xTB columns wherever present)
    feature_cols = ["n_atoms", "imag_freq_cm1"]
    # xTB numeric columns
    for c in xtb.columns:
        if c == "reaction_number":
            continue
        if df[c].dtype.kind in "biufc":
            feature_cols.append(c)
        if len(feature_cols) >= 24:
            break

    if len(feature_cols) < 24:
        print(f"WARNING: only {len(feature_cols)} physics columns available (want 24)")
    else:
        feature_cols = feature_cols[:24]

    physics = df[["reaction_number"] + feature_cols].copy()
    # fillna with column median
    for c in feature_cols:
        physics[c] = physics[c].fillna(physics[c].median())

    out_path = OUT / "physics_24.pkl"
    physics.to_pickle(out_path)
    print(f"physics_24: {physics.shape} → {out_path}")
    print(f"Columns: {list(physics.columns)}")


if __name__ == "__main__":
    main()
