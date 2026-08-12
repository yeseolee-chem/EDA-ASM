#!/usr/bin/env python3
"""Select 300 reactions for DFT re-labeling, stratified on three axes.

Axes (each split into equal-count quantile bins):
  1. n_atoms (compute cost proxy)
  2. reacting_distance_diff_ts_ts (TS asynchronicity — the axis that dominates
     the AM1-vs-DFT geometry gap per user diagnosis)
  3. eint_dft magnitude (Espley interaction energy — cover full range so ML has signal)

Design: 3 × 3 × 3 = 27 cells, ~11-12 rxns per cell. Total 300 (rounded).
Random within each cell (seed=42). Drop cells that are empty; distribute
shortfall proportionally.

Output: pilot 5 (already running) + 295 new rxn_ids → JSON list.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

MANIFEST = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data/manifest.parquet"
PKL = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/manual_tt_7ch_xtb.pkl"
PILOT_IDS = [3, 7, 56, 489, 1217]
OUT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot/selection_300.json")
SEED = 42
TARGET_N = 300
N_BINS = 3


def main():
    m = pd.read_parquet(MANIFEST).rename(columns={"rxn_id": "reaction_number"})
    d = pd.read_pickle(PKL)[["reaction_number", "reacting_distance_diff_ts_ts",
                              "eint_dft"]]
    df = m.merge(d, on="reaction_number", how="inner")
    df = df[df["partition_matches_espley"] == True].copy()
    df = df.dropna(subset=["reacting_distance_diff_ts_ts", "eint_dft"])

    rng = np.random.default_rng(SEED)

    # Quantile bin on each axis
    for col, tag in [("n_atoms", "size"),
                     ("reacting_distance_diff_ts_ts", "asym"),
                     ("eint_dft", "eint")]:
        df[f"bin_{tag}"] = pd.qcut(df[col].rank(method="first"),
                                    q=N_BINS, labels=False)

    df["cell"] = (df["bin_size"].astype(str) + "_"
                  + df["bin_asym"].astype(str) + "_"
                  + df["bin_eint"].astype(str))

    # Force-include pilot IDs so we can measure them within the selection
    pilot_mask = df["reaction_number"].isin(PILOT_IDS)
    remaining_target = TARGET_N - pilot_mask.sum()

    # Sample the rest uniformly across cells (excluding already-picked pilot rows)
    pool = df[~pilot_mask].copy()
    cells = pool["cell"].unique()
    per_cell = remaining_target // len(cells)
    print(f"cells: {len(cells)}, per_cell target: {per_cell}, "
          f"pilot force-included: {pilot_mask.sum()}")

    picks = []
    for c in sorted(cells):
        sub = pool[pool["cell"] == c]
        take = min(per_cell, len(sub))
        if take > 0:
            idx = rng.choice(sub.index, size=take, replace=False)
            picks.extend(sub.loc[idx, "reaction_number"].tolist())

    # Fill shortfall with random remainder
    remaining_pool = pool[~pool["reaction_number"].isin(picks)]
    short = remaining_target - len(picks)
    if short > 0 and len(remaining_pool) > 0:
        extra_idx = rng.choice(remaining_pool.index,
                                size=min(short, len(remaining_pool)),
                                replace=False)
        picks.extend(remaining_pool.loc[extra_idx, "reaction_number"].tolist())

    final = sorted(set(PILOT_IDS + picks))
    print(f"final selection: {len(final)} rxns")
    print(f"  n_atoms  range: {df[df['reaction_number'].isin(final)]['n_atoms'].min()}"
          f" .. {df[df['reaction_number'].isin(final)]['n_atoms'].max()}")
    print(f"  n_atoms  median: {df[df['reaction_number'].isin(final)]['n_atoms'].median():.0f}")

    OUT.write_text(json.dumps({
        "seed": SEED,
        "target_n": TARGET_N,
        "n_bins_per_axis": N_BINS,
        "pilot_ids": PILOT_IDS,
        "rxn_ids": final,
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
