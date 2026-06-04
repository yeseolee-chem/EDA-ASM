"""Build a CSV of the remaining halo8 reactions (those NOT in our 5000 set),
so Stage 5a can be run on them to identify n_fragments==1 cases.
"""
from pathlib import Path
import csv
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")

def main():
    idx = pd.read_parquet(REPO / "data/halo8_index/index.parquet")
    # Apply same eligibility filter Phase 1 likely used
    eligible = idx[idx["interior_ts"]].copy()
    print(f"halo8_index: {len(idx)}")
    print(f"  interior_ts=True: {len(eligible)}")

    existing = set(pd.read_csv(REPO / "outputs/phase1/selected_reactions.csv")["reaction_id"])
    existing |= set(pd.read_csv(REPO / "outputs/phase1/selected_reactions_4500.csv")["reaction_id"])
    print(f"  existing (500+4500): {len(existing)}")
    remaining = eligible[~eligible["reaction_id"].isin(existing)].copy()
    print(f"  remaining pool: {len(remaining)}")

    out = REPO / "outputs/phase1/remaining_pool.csv"
    remaining.to_csv(out, index=False)
    print(f"[OK] → {out}")


if __name__ == "__main__":
    main()
