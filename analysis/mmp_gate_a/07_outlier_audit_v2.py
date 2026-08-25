#!/usr/bin/env python3
"""Step 7 (v2): outlier audit + strict filter-integrity check (§6 of rev SPEC).

For pairs with |δG_act| > 15 kcal/mol, count how many still fail the new filters.
Post-filter, (a) core_signature mismatch and (b) sub_heavy>8 must be 0.
"""
from pathlib import Path
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"
OUTLIER_THRESH = 15.0
SUB_MAX_HEAVY = 8


def main():
    pairs = pd.read_pickle(OUT / "mmp_pairs_labeled_v2.pkl")
    cohort = pd.read_pickle(OUT / "cohort_join.pkl").set_index("reaction_number")
    core_df = pd.read_pickle(OUT / "core_v2.pkl").set_index("rxn_id")

    outliers = pairs[pairs["delta_G_act"].abs() > OUTLIER_THRESH].copy()
    print(f"Outliers (|δG_act|>{OUTLIER_THRESH}): {len(outliers)}/{len(pairs)} "
          f"({len(outliers)/len(pairs)*100:.1f}%)")

    if len(outliers):
        # Reason (a): core mismatch — get cores of both reactions, compare signatures
        outliers["core_sig_A"] = outliers["rxn_id_A"].map(core_df["core_signature"])
        outliers["core_sig_B"] = outliers["rxn_id_B"].map(core_df["core_signature"])
        outliers["reason_a_core_mismatch"] = outliers["core_sig_A"] != outliers["core_sig_B"]
        # Reason (b): sub too big
        outliers["reason_b_sub_too_big"] = (
            (outliers["sub_n_heavy_A"] > SUB_MAX_HEAVY) |
            (outliers["sub_n_heavy_B"] > SUB_MAX_HEAVY)
        )

        n_a = int(outliers["reason_a_core_mismatch"].sum())
        n_b = int(outliers["reason_b_sub_too_big"].sum())
        print(f"Reason (a) core mismatch in outliers: {n_a}")
        print(f"Reason (b) sub_heavy>{SUB_MAX_HEAVY} in outliers: {n_b}")

        outliers["smi_A"] = outliers["rxn_id_A"].map(cohort["rxn_smiles"])
        outliers["smi_B"] = outliers["rxn_id_B"].map(cohort["rxn_smiles"])
        outliers = outliers.sort_values("delta_G_act", key=lambda s: s.abs(), ascending=False)
        cols = ["rxn_id_A", "rxn_id_B", "sub_A", "sub_B", "sub_n_heavy_A", "sub_n_heavy_B",
                "core_sig_A", "core_sig_B", "reason_a_core_mismatch", "reason_b_sub_too_big",
                "G_act_A", "G_act_B", "delta_G_act", "smi_A", "smi_B", "key"]
        outliers[cols].to_csv(OUT / "outliers_v2.csv", index=False)
        print(f"Top 10 outliers (v2):")
        print(outliers[["rxn_id_A", "rxn_id_B", "sub_A", "sub_B", "delta_G_act",
                        "reason_a_core_mismatch", "reason_b_sub_too_big"]].head(10).to_string(index=False))
    else:
        outliers[["rxn_id_A", "rxn_id_B"]].to_csv(OUT / "outliers_v2.csv", index=False)
        n_a = n_b = 0

    frac = len(outliers) / len(pairs) if len(pairs) else 0.0

    # STOP conditions per §6
    stop = (n_a > 0) or (n_b > 0)
    with open(OUT / "GATE6_STATUS_v2.txt", "w") as f:
        if stop:
            f.write(f"FAIL filter_leaked reason_a={n_a} reason_b={n_b} frac_outliers={frac:.3f}\n")
        elif frac > 0.05:
            f.write(f"REVIEW frac_outliers={frac:.3f} (all filters pass, but rate>5%)\n")
        else:
            f.write(f"PASS frac_outliers={frac:.3f}\n")
    if stop:
        print(f"\n=== GATE-6-rev FAIL: filter leak (a={n_a}, b={n_b}) ===")
    else:
        print(f"\n=== GATE-6-rev: frac={frac:.3f} (filter integrity OK: a=0 b=0) ===")


if __name__ == "__main__":
    main()
