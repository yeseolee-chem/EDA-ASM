#!/usr/bin/env python3
"""aggregate.py — merge slice parquets into xtb_features.parquet with completeness gates."""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
SLICES, OUT = ROOT / "slices", ROOT / "xtb_features.parquet"
N_SLICES = int(os.environ.get("ESPLEY_NSLICES", "18"))
N_EXPECTED = 5260          # 5,265 labels − 5 excluded (3090, 3766, 4252, 3400, 5783)


def main():
    paths = sorted(SLICES.glob("slice_*.parquet"))
    if len(paths) != N_SLICES:
        sys.exit(f"GATE: expected {N_SLICES} slice files, found {len(paths)}: {[p.name for p in paths]}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values("rxn_id").reset_index(drop=True)
    if len(df) != N_EXPECTED or df.rxn_id.nunique() != N_EXPECTED:
        sys.exit(f"GATE: expected {N_EXPECTED} unique rxns, got {len(df)} rows / {df.rxn_id.nunique()} unique")

    df["dft_barrier_eda"] = df["dft_d1_kcal"] + df["dft_d2_kcal"] + df["dft_e_bond_kcal"]
    df["dft_bsse_gap"] = df["dft_barrier_kcal"] - df["dft_barrier_eda"]
    df["is_charged"] = (df["charge2"].fillna(0).astype(int) != 0).astype(int)

    ok_mask = df["xtb_status"] == "ok"
    for col in ("dft_barrier_eda", "dft_bsse_gap"):
        nan_ok = int(df.loc[ok_mask, col].isna().sum())
        if nan_ok:
            sys.exit(f"GATE: derived column {col} has {nan_ok} NaN in ok rows")
    gap_mean = float(df.loc[ok_mask, "dft_bsse_gap"].mean())
    gap_sd = float(df.loc[ok_mask, "dft_bsse_gap"].std())
    print(f"derived: dft_barrier_eda / dft_bsse_gap (mean {gap_mean:+.3f}, sd {gap_sd:.3f}) / is_charged  ({int(df.is_charged.sum())} rxns)")

    df.to_parquet(OUT, index=False)
    st = dict(df["xtb_status"].value_counts())
    print(f"aggregated {len(df)} rows from {len(paths)} slices -> {OUT}")
    print(f"xtb_status: {st}")
    ok = df[df.xtb_status == "ok"]
    print(f"ok={len(ok)}  solvation tags: {dict(ok.xtb_solvation.value_counts())}")
    print(f"ok rows with NaN in any xtb_*/q_* column: {int(ok.filter(regex='^(xtb_|q_)').isna().any(axis=1).sum())}")
    if len(ok) < 0.99 * N_EXPECTED:
        sys.exit(f"GATE: xTB success rate {len(ok)/N_EXPECTED:.3%} < 99% — inspect failures before ML")


if __name__ == "__main__":
    main()
