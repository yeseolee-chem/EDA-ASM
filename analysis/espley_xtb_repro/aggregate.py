#!/usr/bin/env python3
"""aggregate.py — merge partial slice parquets into xtb_features.parquet."""
import sys
from pathlib import Path

import pandas as pd

SLICES = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/slices")
OUT = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features.parquet")


def main():
    paths = sorted(SLICES.glob("slice_*.parquet"))
    if not paths:
        sys.exit("no slice parquets found")
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True).sort_values("rxn_id").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"aggregated {len(df)} rows from {len(paths)} slices -> {OUT}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    print(f"xtb_status distribution: {dict(df['xtb_status'].value_counts())}")


if __name__ == "__main__":
    main()
