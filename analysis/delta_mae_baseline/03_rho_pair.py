#!/usr/bin/env python3
"""Step 3: ρ_pair — intra-pair error correlation (§3.3 strategy decision).

For each pair × channel:
  e_A = y_pred(A) - y_true(A)
  e_B = y_pred(B) - y_true(B)
ρ_pair = Pearson corr over ALL pairs, symmetrized by including (e_B, e_A).

Required ρ for MAE_δ < 1.0:  1 - (1.0 / (sqrt(2) * abs_mae))^2
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline")
OUT = BASE / "artifacts"

TARGETS = [
    "d1_own_dft", "d2_own_dft",
    "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft",
    "interaction_own_dft",
]
TARGET_MAE = 1.0


def main():
    oof = pd.read_pickle(OUT / "oof_predictions.pkl")
    pairs = pd.read_pickle(OUT / "pairs_dedup.pkl")

    rows = []
    for scheme in oof["scheme"].unique():
        for seed in sorted(oof["seed"].unique()):
            sub = oof[(oof["scheme"] == scheme) & (oof["seed"] == seed)].set_index("reaction_number")
            for tgt in TARGETS:
                eA = (sub[f"{tgt}_pred"] - sub[f"{tgt}_true"])
                eB = (sub[f"{tgt}_pred"] - sub[f"{tgt}_true"])
                # for each pair: get e at rxn_A and rxn_B
                pair_eA = pairs["rxn_id_A"].map(eA).values
                pair_eB = pairs["rxn_id_B"].map(eB).values
                # symmetrize by concatenating (eA,eB) with (eB,eA)
                x = np.concatenate([pair_eA, pair_eB])
                y = np.concatenate([pair_eB, pair_eA])
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    rho = np.nan
                else:
                    rho, _ = pearsonr(x, y)
                abs_mae_pair = float(np.mean(np.abs(np.concatenate([pair_eA, pair_eB]))))
                # sqrt(2*(1-rho)) * abs_mae = predicted mae_delta
                predicted_mae_delta = float(np.sqrt(2 * (1 - rho)) * abs_mae_pair) if not np.isnan(rho) else np.nan
                # required rho for mae_delta < TARGET_MAE
                if abs_mae_pair > 0:
                    val = 1 - (TARGET_MAE / (np.sqrt(2) * abs_mae_pair)) ** 2
                    required_rho = float(val)
                else:
                    required_rho = np.nan
                rows.append({
                    "scheme": scheme, "seed": seed, "channel": tgt,
                    "abs_mae_pair": abs_mae_pair,
                    "rho_pair_measured": float(rho),
                    "predicted_mae_delta": predicted_mae_delta,
                    "required_rho_for_1kcal": required_rho,
                    "gap_actual_minus_required": float(rho - required_rho) if not np.isnan(rho) else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rho_pair_per_seed.csv", index=False)

    agg = df.groupby(["scheme", "channel"]).agg(
        abs_mae_pair=("abs_mae_pair", "mean"),
        rho_pair_measured=("rho_pair_measured", "mean"),
        rho_pair_std=("rho_pair_measured", "std"),
        predicted_mae_delta=("predicted_mae_delta", "mean"),
        required_rho_for_1kcal=("required_rho_for_1kcal", "mean"),
        gap_actual_minus_required=("gap_actual_minus_required", "mean"),
    ).reset_index()

    def tier(row):
        if np.isnan(row["rho_pair_measured"]):
            return "?"
        gap = row["gap_actual_minus_required"]
        if gap >= 0:
            return "A_SUFFICIENT"
        if gap >= -0.05:
            return "B_NEAR_REACHABLE"
        if row["rho_pair_measured"] >= 0.7:
            return "C_HARD"
        if row["rho_pair_measured"] < 0.3:
            return "D_NO_HOPE"
        return "C_HARD"

    agg["strategy_tier"] = agg.apply(tier, axis=1)
    agg.to_csv(OUT / "rho_pair.csv", index=False)
    print("ρ_pair per (scheme, channel) — seed averaged:")
    print(agg.to_string(index=False))

    with open(OUT / "GATE3_STATUS.txt", "w") as f:
        # informational
        for scheme in agg["scheme"].unique():
            sub = agg[agg["scheme"] == scheme]
            f.write(f"{scheme}:\n")
            for _, r in sub.iterrows():
                f.write(f"  {r['channel']}: rho={r['rho_pair_measured']:.3f} "
                        f"req={r['required_rho_for_1kcal']:.3f} tier={r['strategy_tier']}\n")


if __name__ == "__main__":
    main()
