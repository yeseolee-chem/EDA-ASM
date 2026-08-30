#!/usr/bin/env python3
"""SPEC15 Step 5 — compare new B3LYP-TS labels vs old AM1-TS labels.

Key metric: δ_prop = std × √2 (correct for zero-mean noise assumption on
δ pairs where systematic bias cancels). Reported alongside MAE and mean-signed
so the paper can quote whichever is appropriate.

GATE-4: 6 channels' delta_prop within ±20% of spec14's 218-rxn measurement.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
SPEC14 = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")

CH = ["elst", "pauli", "oi", "disp", "cpcm", "cds"]


def main():
    new = pd.read_csv(BASE / "artifacts" / "labels_b3lyp.csv").set_index("rxn_id")
    old = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    old["reaction_number"] = old["reaction_number"].astype(int)
    old = old.set_index("reaction_number")

    idx = new.index.intersection(old.index)
    print(f"comparable rxns: {len(idx)}")

    rows = []
    # 6 EDA channels
    for c in CH:
        a = old.loc[idx, f"{c}_dft"].values.astype(float)
        b = new.loc[idx, f"{c}_dft"].values.astype(float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        d = b - a
        rows.append(dict(
            channel=c, n=int(mask.sum()),
            mean_signed=float(d.mean()), std=float(d.std()),
            mae=float(np.abs(d).mean()),
            median_abs=float(np.median(np.abs(d))),
            p95=float(np.percentile(np.abs(d), 95)),
            delta_prop=float(d.std() * np.sqrt(2)),
            pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else np.nan,
        ))
    # d1, d2 (own labels)
    for c, ocol in [("d1", "d1_own_dft"), ("d2", "d2_own_dft")]:
        a = old.loc[idx, ocol].values.astype(float)
        b = new.loc[idx, f"{c}_b3lyp"].values.astype(float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        d = b - a
        rows.append(dict(
            channel=c, n=int(mask.sum()),
            mean_signed=float(d.mean()), std=float(d.std()),
            mae=float(np.abs(d).mean()),
            median_abs=float(np.median(np.abs(d))),
            p95=float(np.percentile(np.abs(d), 95)),
            delta_prop=float(d.std() * np.sqrt(2)),
            pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else np.nan,
        ))
    R = pd.DataFrame(rows)
    R.to_csv(BASE / "artifacts" / "label_shift.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== label shift (B3LYP TS − AM1 TS, kcal/mol) ===")
    print(R.round(3).to_string(index=False))

    # Cross-check vs spec14
    prev_path = SPEC14 / "artifacts" / "channel_comparison.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        print("\n=== spec14 (n=218) vs this run — δ_prop consistency (±20%) ===")
        violations = []
        for c in CH:
            r_now = R[R["channel"] == c].iloc[0]
            r_prev = prev[prev["channel"] == c].iloc[0]
            dp_now = r_now["delta_prop"]
            dp_prev = r_prev["std"] * np.sqrt(2) if "std" in prev.columns else np.nan
            frac = abs(dp_now - dp_prev) / max(abs(dp_prev), 1e-9)
            tag = "OK" if frac <= 0.20 else "OUT"
            if tag == "OUT":
                violations.append(c)
            print(f"  {c:>6}: this {dp_now:6.2f}   spec14 {dp_prev:6.2f}   |Δ|/prev {frac*100:5.1f}%  {tag}")

        status = "PASS" if not violations else "FAIL"
        with open(BASE / "artifacts" / "GATE4_STATUS.txt", "w") as f:
            f.write(f"{status} violations={violations}\n")
        print(f"=== GATE-4 {status} ===")


if __name__ == "__main__":
    main()
