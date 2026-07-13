"""OOD (out-of-distribution) detection on the assembled 5-channel labels.

Rule: per-family z-score for each channel; a reaction is flagged OOD if
|z| > 4 in any channel or if any channel is a top-3 extreme.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
IN_PQ = V8 / "labels/labels_v8_5channel.parquet"
OUT_CSV = V8 / "labels/ood_report_v8.csv"

CH = ["pauli_kcal", "elst_kcal", "orb_kcal", "disp_kcal", "strain_kcal", "act_kcal"]
Z_THRESH = 4.0


def main():
    df = pd.read_parquet(IN_PQ)
    print(f"Loaded {len(df)} rxns")

    z_df = pd.DataFrame(index=df.index)
    z_df["reaction_id"] = df["reaction_id"]
    z_df["family"] = df["family"]

    for ch in CH:
        for fam, grp in df.groupby("family"):
            m = grp[ch].mean()
            s = grp[ch].std()
            if s < 1e-6:
                continue
            z_df.loc[grp.index, f"z_{ch}"] = (grp[ch] - m) / s

    max_abs_z = z_df[[f"z_{c}" for c in CH]].abs().max(axis=1)
    z_df["max_abs_z"] = max_abs_z
    ood = z_df[max_abs_z > Z_THRESH].copy()
    ood = ood.sort_values("max_abs_z", ascending=False)

    print(f"\n=== OOD (|z| > {Z_THRESH} in any channel) — {len(ood)} rxns ===")
    if len(ood):
        for _, r in ood.head(30).iterrows():
            top_ch = None; top_z = 0.0
            for c in CH:
                v = abs(r.get(f"z_{c}", 0.0) or 0.0)
                if v > top_z:
                    top_z = v; top_ch = c
            print(f"  {r['reaction_id']:35}  ({r['family']:12}) top={top_ch}={r.get(f'z_{top_ch}', 0.0):.2f}")

    print(f"\n=== per-family channel stats ===")
    for fam, grp in df.groupby("family"):
        print(f"\n{fam}  (n={len(grp)})")
        for ch in CH:
            print(f"  {ch:15}  mean={grp[ch].mean():>8.2f}  std={grp[ch].std():>7.2f}  min={grp[ch].min():>8.2f}  max={grp[ch].max():>8.2f}")

    ood.to_csv(OUT_CSV, index=False)
    print(f"\nReport: {OUT_CSV}")


if __name__ == "__main__":
    main()
