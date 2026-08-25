#!/usr/bin/env python3
"""Step 5: barrier δ — direct vs channel-sum reconstruction.

(a) δ(interaction_own_dft) directly predicted
(b) δ(Σ channels)  reconstructed from per-channel predictions
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline")
OUT = BASE / "artifacts"

CH_SUM = ["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]  # EDA channels sum ≈ interaction
BARRIER_LEARNED = "interaction_own_dft"  # direct interaction predictor


def main():
    oof = pd.read_pickle(OUT / "oof_predictions.pkl")
    pairs = pd.read_pickle(OUT / "pairs_dedup.pkl")

    rows = []
    for scheme in oof["scheme"].unique():
        for seed in sorted(oof["seed"].unique()):
            sub = oof[(oof["scheme"] == scheme) & (oof["seed"] == seed)].set_index("reaction_number")

            # (a) direct
            dA_t = pairs["rxn_id_A"].map(sub[f"{BARRIER_LEARNED}_true"]).values
            dB_t = pairs["rxn_id_B"].map(sub[f"{BARRIER_LEARNED}_true"]).values
            dA_p = pairs["rxn_id_A"].map(sub[f"{BARRIER_LEARNED}_pred"]).values
            dB_p = pairs["rxn_id_B"].map(sub[f"{BARRIER_LEARNED}_pred"]).values
            direct_true = dB_t - dA_t
            direct_pred = dB_p - dA_p
            mae_direct = float(np.mean(np.abs(direct_pred - direct_true)))

            # (b) sum
            sum_true = np.zeros(len(pairs))
            sum_pred = np.zeros(len(pairs))
            for ch in CH_SUM:
                a_t = pairs["rxn_id_A"].map(sub[f"{ch}_true"]).values
                b_t = pairs["rxn_id_B"].map(sub[f"{ch}_true"]).values
                a_p = pairs["rxn_id_A"].map(sub[f"{ch}_pred"]).values
                b_p = pairs["rxn_id_B"].map(sub[f"{ch}_pred"]).values
                sum_true += (b_t - a_t)
                sum_pred += (b_p - a_p)
            mae_sum = float(np.mean(np.abs(sum_pred - sum_true)))

            # baseline δ=0 for each
            base_direct = float(np.mean(np.abs(direct_true)))
            base_sum = float(np.mean(np.abs(sum_true)))

            rows.append({
                "scheme": scheme, "seed": seed,
                "direct_mae_delta": mae_direct,
                "direct_baseline": base_direct,
                "sum_mae_delta": mae_sum,
                "sum_baseline": base_sum,
                "direct_beats_sum": mae_direct < mae_sum,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "barrier_delta_per_seed.csv", index=False)

    agg = df.groupby("scheme").agg(
        direct_mae_delta=("direct_mae_delta", "mean"),
        direct_baseline=("direct_baseline", "mean"),
        sum_mae_delta=("sum_mae_delta", "mean"),
        sum_baseline=("sum_baseline", "mean"),
        n_wins_direct=("direct_beats_sum", "sum"),
    ).reset_index()
    agg.to_csv(OUT / "barrier_delta.csv", index=False)
    print("Barrier δ — direct vs channel-sum (seed averaged):")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
