#!/usr/bin/env python3
"""
Feature selection: correlation + variance filter (Grayson-style substitute for
their feature_selection/f_select.py which is interactive/Windows-only).

Input: tt_features.pkl (from f_extract.py -c collate)
Output: manual_dipolar.pkl (Grayson-compatible name for hyp_tuning.py input)

Filter rules (matches prior REPORT.md pipeline):
  1. Drop columns with |corr| > 0.99 (near-collinear)
  2. Drop columns with variance < 0.05 (low variance)

Preserves:
  - reaction_number (metadata)
  - Any column containing '_dft' (targets — matched by hyp_tuning via _dft suffix)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--corr-threshold", type=float, default=0.99)
    ap.add_argument("--variance-threshold", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_pickle(args.input)
    print(f"loaded {args.input.name}: {df.shape}")

    # Preserved columns
    preserve = {"reaction_number"}
    target_cols = [c for c in df.columns if "_dft" in c]
    preserve.update(target_cols)

    # Candidate feature columns (numeric only)
    feature_cols = [
        c for c in df.columns
        if c not in preserve and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  targets ({len(target_cols)}): {target_cols}")
    print(f"  candidate features: {len(feature_cols)}")
    print(f"  non-numeric/dropped from filter: {set(df.columns) - preserve - set(feature_cols)}")

    # Variance filter
    variances = df[feature_cols].var()
    low_var = variances[variances < args.variance_threshold].index.tolist()
    print(f"  dropped for low variance (<{args.variance_threshold}): {len(low_var)}")
    for c in low_var:
        print(f"    {c}: var={variances[c]:.4f}")
    feature_cols = [c for c in feature_cols if c not in low_var]

    # Correlation filter (drop later of highly-correlated pair)
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > args.corr_threshold)]
    print(f"  dropped for high correlation (>{args.corr_threshold}): {len(high_corr)}")
    for c in high_corr:
        # find its partner
        partners = upper.index[upper[c] > args.corr_threshold].tolist()
        print(f"    {c} (partner: {partners[0]}, corr={upper.loc[partners[0], c]:.3f})")
    feature_cols = [c for c in feature_cols if c not in high_corr]

    # Assemble output
    keep_cols = ["reaction_number"] + feature_cols + target_cols
    out = df[keep_cols].copy()
    print(f"\nkept {len(feature_cols)} features, {len(target_cols)} targets, {len(out)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(args.output)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
