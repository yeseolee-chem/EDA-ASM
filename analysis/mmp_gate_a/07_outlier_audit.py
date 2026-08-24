#!/usr/bin/env python3
"""Step 7: outlier audit - GATE-6.

Extract pairs where |delta_G_act| > 15 kcal/mol. Manually flag likely spurious.
"""
from pathlib import Path
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"
OUTLIER_THRESH = 15.0


def main():
    pairs = pd.read_pickle(OUT / "mmp_pairs_labeled.pkl")
    cohort = pd.read_pickle(OUT / "cohort_join.pkl").set_index("reaction_number")

    outliers = pairs[pairs["delta_G_act"].abs() > OUTLIER_THRESH].copy()
    print(f"Outliers (|delta_G_act|>{OUTLIER_THRESH}): {len(outliers)} of {len(pairs)} "
          f"({len(outliers)/len(pairs)*100:.1f}%)")

    if len(outliers):
        outliers["smi_A"] = outliers["rxn_id_A"].map(cohort["rxn_smiles"])
        outliers["smi_B"] = outliers["rxn_id_B"].map(cohort["rxn_smiles"])
        outliers = outliers.sort_values("delta_G_act", key=lambda s: s.abs(), ascending=False)
        cols = ["rxn_id_A", "rxn_id_B", "sub_A", "sub_B", "G_act_A", "G_act_B", "delta_G_act",
                "key", "smi_A", "smi_B"]
        outliers[cols].to_csv(OUT / "outliers.csv", index=False)
        print(f"Top 10 outliers:")
        print(outliers[["rxn_id_A", "rxn_id_B", "sub_A", "sub_B", "delta_G_act"]].head(10).to_string(index=False))
    else:
        outliers[["rxn_id_A", "rxn_id_B"]].to_csv(OUT / "outliers.csv", index=False)

    frac = len(outliers) / len(pairs) if len(pairs) else 0
    with open(OUT / "GATE6_STATUS.txt", "w") as f:
        if frac > 0.05:
            f.write(f"REVIEW frac_outliers={frac:.3f}\n")
        else:
            f.write(f"PASS frac_outliers={frac:.3f}\n")


if __name__ == "__main__":
    main()
