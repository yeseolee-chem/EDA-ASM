"""Sample 4500 NEW Halo8 trajectories (excluding existing 500), stratified by
source × Ea tertile to roughly match the existing distribution.

Writes:
  outputs/phase1/selected_reactions_4500.csv     — new 4500 only
  outputs/phase1/selected_reactions_5000.csv     — combined 500 + 4500 (full pool)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
INDEX = REPO / "data/halo8_index/index.parquet"
EXISTING_500 = REPO / "outputs/phase1/selected_reactions.csv"
OUT_4500 = REPO / "outputs/phase1/selected_reactions_4500.csv"
OUT_5000 = REPO / "outputs/phase1/selected_reactions_5000.csv"

SEED = 20260516
N_TARGET = 4500


def main():
    df = pd.read_parquet(INDEX)
    existing = set(pd.read_csv(EXISTING_500)["reaction_id"])
    pool = df[~df["reaction_id"].isin(existing)].copy()
    # Only interior_ts trajectories
    pool = pool[pool["interior_ts"]]
    print(f"halo8_index: {len(df):,}")
    print(f"existing 500 excluded → pool: {len(pool):,}")

    # Stratify by source × ea_tertile (computed on the pool itself for balance)
    pool["ea_tertile"] = pd.qcut(pool["activation_energy"], 3,
                                  labels=["low", "mid", "high"])
    counts = pool.groupby(["source", "ea_tertile"], observed=True).size().unstack()
    print(f"\n=== pool counts by source × ea_tertile ===")
    print(counts)

    rng = np.random.default_rng(SEED)
    # Allocate 4500 proportionally to source × tertile
    total = len(pool)
    sampled = []
    for src in pool["source"].unique():
        for tert in ["low", "mid", "high"]:
            cell = pool[(pool["source"] == src) & (pool["ea_tertile"] == tert)]
            if len(cell) == 0:
                continue
            quota = round(N_TARGET * len(cell) / total)
            take = min(quota, len(cell))
            picked = cell.sample(n=take, random_state=rng.integers(2**31))
            sampled.append(picked)
            print(f"  {src:10s} × {tert:5s}: pool={len(cell):5d}  quota={quota:5d}  picked={take}")

    df_sampled = pd.concat(sampled, ignore_index=True)
    # If short of N_TARGET due to rounding, top up randomly from remaining
    if len(df_sampled) < N_TARGET:
        remaining = pool[~pool["reaction_id"].isin(df_sampled["reaction_id"])]
        topup = remaining.sample(n=N_TARGET - len(df_sampled),
                                  random_state=rng.integers(2**31))
        df_sampled = pd.concat([df_sampled, topup], ignore_index=True)
    elif len(df_sampled) > N_TARGET:
        df_sampled = df_sampled.sample(n=N_TARGET, random_state=rng.integers(2**31))

    df_sampled["seed"] = SEED
    df_sampled["cohort"] = "phase2_4500"

    print(f"\nfinal sample: {len(df_sampled)}")
    print(f"by source:")
    print(df_sampled["source"].value_counts())

    df_sampled.to_csv(OUT_4500, index=False)
    print(f"\n[OK] → {OUT_4500}")

    # Combined: existing 500 + new 4500
    ex = pd.read_csv(EXISTING_500)
    # Make sure ea_tertile column exists in both (existing may have it)
    combo = pd.concat([ex, df_sampled], ignore_index=True)
    combo.to_csv(OUT_5000, index=False)
    print(f"[OK] combined 5000 → {OUT_5000}  rows={len(combo)}")


if __name__ == "__main__":
    main()
