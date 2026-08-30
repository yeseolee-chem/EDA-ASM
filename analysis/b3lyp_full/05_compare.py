#!/usr/bin/env python3
"""SPEC16rev Step 5 — compare new B3LYP-TS labels vs old AM1-TS labels.

⚠️ ONLY script that reads phase5_dataset_v2.pkl (read-only, for label-shift
comparison in the paper). New label generation path never touches it.

GATE-4: 6 EDA channels + d1, d2 δ_prop within ±20% of spec15's 218-rxn measurement.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
SPEC15 = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")

CH = ["elst", "pauli", "oi", "disp", "cpcm", "cds"]

# spec15 reference δ_prop values (218-rxn observation)
SPEC15_REFS = {
    "elst":  13.77, "pauli": 26.69, "oi": 17.75,
    "disp":   2.54, "cpcm":   5.50, "cds":  0.67,
    "d1":     8.92, "d2":     7.02,
}


def main():
    new = pd.read_csv(BASE / "artifacts" / "labels_b3lyp_full.csv").set_index("rxn_id")
    old = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    old["reaction_number"] = old["reaction_number"].astype(int)
    old = old.set_index("reaction_number")

    idx = new.index.intersection(old.index)
    print(f"comparable rxns: {len(idx)}")

    rows = []
    for c in CH:
        a = old.loc[idx, f"{c}_dft"].values.astype(float)
        b = new.loc[idx, f"{c}_dft"].values.astype(float)
        m = ~(np.isnan(a) | np.isnan(b))
        a, b = a[m], b[m]
        d = b - a
        rows.append(dict(
            channel=c, n=int(m.sum()),
            mean_signed=float(d.mean()), std=float(d.std()),
            mae=float(np.abs(d).mean()),
            p95=float(np.percentile(np.abs(d), 95)),
            delta_prop=float(d.std() * np.sqrt(2)),
            pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else np.nan,
        ))
    for c, ocol in [("d1", "d1_own_dft"), ("d2", "d2_own_dft")]:
        a = old.loc[idx, ocol].values.astype(float)
        b = new.loc[idx, f"{c}_b3lyp"].values.astype(float)
        m = ~(np.isnan(a) | np.isnan(b))
        a, b = a[m], b[m]
        d = b - a
        rows.append(dict(
            channel=c, n=int(m.sum()),
            mean_signed=float(d.mean()), std=float(d.std()),
            mae=float(np.abs(d).mean()),
            p95=float(np.percentile(np.abs(d), 95)),
            delta_prop=float(d.std() * np.sqrt(2)),
            pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else np.nan,
        ))
    R = pd.DataFrame(rows)
    R.to_csv(BASE / "artifacts" / "label_shift.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== label shift (B3LYP TS − AM1 TS, kcal/mol, on 3504 overlap) ===")
    print(R.round(3).to_string(index=False))

    # Consistency vs spec15
    print("\n=== spec15 (n=218) vs full (n=3504) — δ_prop consistency (±20%) ===")
    violations = []
    for _, r in R.iterrows():
        ref = SPEC15_REFS.get(r["channel"])
        if ref is None:
            continue
        frac = abs(r["delta_prop"] - ref) / max(abs(ref), 1e-9)
        tag = "OK" if frac <= 0.20 else "OUT"
        if tag == "OUT":
            violations.append(r["channel"])
        print(f"  {r['channel']:>6}: this {r['delta_prop']:6.2f}  spec15 {ref:6.2f}  "
              f"|Δ|/ref {frac*100:5.1f}%  {tag}")

    status = "PASS" if not violations else "FAIL"
    with open(BASE / "artifacts" / "GATE4_STATUS.txt", "w") as f:
        f.write(f"{status} violations={violations}\n")
    print(f"\n=== GATE-4 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
