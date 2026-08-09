#!/usr/bin/env python3
"""
merge_xtb_into_pkl.py — combine 10 shard parquets into unified xtb channels DF,
then merge as FEATURES into manual_tt_7ch.pkl (add 8 xtb columns).

Output columns added (all end in `_xtb`, so they're not caught by `_dft` target regex):
    elst_xtb, pauli_xtb, oi_xtb, disp_xtb, cpcm_xtb, cds_xtb,
    strain_1_xtb, strain_2_xtb
Bookkeeping added (don't end in `_dft` or `_xtb`; excluded from ML):
    eint_total_xtb, dispd4_xtb, gshift_xtb, telescope_residual_kcal

Inner join on reaction_number. Rows without xtb data are dropped.
"""
import sys
from pathlib import Path
import pandas as pd

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results")
SHARD_DIR = BASE / "xtb_channels"
IN_PKL = BASE / "manual_tt_7ch.pkl"
OUT_PKL = BASE / "manual_tt_7ch_xtb.pkl"

# gather all shard parquets
shard_files = sorted(SHARD_DIR.glob("channels_shard*.parquet"))
if not shard_files:
    sys.exit(f"[fatal] no shard parquets in {SHARD_DIR}")
print(f"[merge] found {len(shard_files)} shard parquets")

xtb_df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
print(f"[merge] combined xtb: {len(xtb_df)} rows, {len(xtb_df.columns)} cols")
print(f"        columns: {list(xtb_df.columns)}")

# Check for duplicates
dup = xtb_df['reaction_number'].duplicated().sum()
if dup:
    print(f"[warn] {dup} duplicate reaction_numbers in xtb data; keeping first")
    xtb_df = xtb_df.drop_duplicates(subset='reaction_number', keep='first')

# Load base pkl
base = pd.read_pickle(IN_PKL)
print(f"[merge] base pkl: {base.shape}")

merged = base.merge(xtb_df, on='reaction_number', how='inner', validate='1:1')
print(f"[merge] merged: {merged.shape}  ({len(base) - len(merged)} base rows dropped, no xtb)")

merged.to_pickle(OUT_PKL)
print(f"[merge] wrote {OUT_PKL}")

# Report new columns
new_cols = [c for c in merged.columns if c not in base.columns]
print(f"[merge] added columns ({len(new_cols)}): {new_cols}")

# Sanity: channel comparison DFT vs xTB for elst (should correlate)
if 'elst_dft' in merged.columns and 'elst_xtb' in merged.columns:
    import numpy as np
    r = np.corrcoef(merged['elst_dft'], merged['elst_xtb'])[0,1]
    print(f"[sanity] r(elst_dft, elst_xtb) = {r:.4f}")
