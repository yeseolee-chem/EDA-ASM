#!/usr/bin/env python3
"""Step 5: equivalent ρ - back-compute what pair-error correlation Arm-0 (subtract)
would need to reach the same MAE_δ that pairwise arms achieve.

ρ_equiv = 1 - (MAE_δ_arm / (sqrt(2) * abs_MAE_arm0))^2

Larger ρ_equiv → this arm gives Arm-0 an equivalent 'ρ boost' of this much.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta")
OUT = BASE / "artifacts"
DMB = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts")

LEARNED = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
           "oi_dft", "disp_dft", "cpcm_dft"]


def main():
    agg = pd.read_csv(OUT / "delta_mae_by_arm.csv")
    ref = pd.read_csv(DMB / "delta_mae_table.csv")
    ref_comp = ref[ref["scheme"] == "groupkfold_component"].set_index("channel")

    rows = []
    for _, r in agg.iterrows():
        ch = r["channel"]
        if ch not in LEARNED:
            continue
        abs_mae0 = ref_comp.loc[ch, "abs_mae"]
        rho_equiv = 1 - (r["mae_delta"] / (np.sqrt(2) * abs_mae0)) ** 2
        rows.append({
            "arm": r["arm"], "split": r["split"], "channel": ch,
            "abs_mae_arm0": abs_mae0,
            "mae_delta": r["mae_delta"],
            "arm0_mae_delta": r["arm0_mae_delta"],
            "rho_equiv": rho_equiv,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rho_equiv.csv", index=False)
    print("ρ_equiv per (arm, split, channel):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
