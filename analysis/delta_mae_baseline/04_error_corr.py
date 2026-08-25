#!/usr/bin/env python3
"""Step 4: cross-channel δ-error correlation (8x8).

DIFFERENT from mmp_gate_a §5.4 (that was correlation of TRUE δ values, physics).
This is correlation of MODEL PREDICTION ERRORS (model limitation).
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline")
OUT = BASE / "artifacts"

CHANNELS = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft", "oi_dft",
            "disp_dft", "cpcm_dft", "cds_dft"]


def main():
    oof = pd.read_pickle(OUT / "oof_predictions.pkl")
    pairs = pd.read_pickle(OUT / "pairs_dedup.pkl")

    for scheme in oof["scheme"].unique():
        # Aggregate over all seeds: compute err_delta per (pair, seed, channel), stack
        stacks = {ch: [] for ch in CHANNELS}
        cancel_num = 0.0
        cancel_den = 0.0
        for seed in sorted(oof["seed"].unique()):
            sub = oof[(oof["scheme"] == scheme) & (oof["seed"] == seed)].set_index("reaction_number")
            errs_this_seed = {}
            for ch in CHANNELS:
                dA_t = pairs["rxn_id_A"].map(sub[f"{ch}_true"])
                dB_t = pairs["rxn_id_B"].map(sub[f"{ch}_true"])
                dA_p = pairs["rxn_id_A"].map(sub[f"{ch}_pred"])
                dB_p = pairs["rxn_id_B"].map(sub[f"{ch}_pred"])
                err_d = ((dB_p - dA_p) - (dB_t - dA_t)).values
                stacks[ch].append(err_d)
                errs_this_seed[ch] = err_d
            # cancellation: for each pair, sum of channel errors vs sum of abs errors
            E = np.array([errs_this_seed[ch] for ch in CHANNELS])   # (nch, npairs)
            cancel_num += np.sum(np.abs(E.sum(axis=0)))
            cancel_den += np.sum(np.abs(E))

        # Stack across seeds
        M = np.column_stack([np.concatenate(stacks[ch]) for ch in CHANNELS])
        dfE = pd.DataFrame(M, columns=CHANNELS)
        pearson = dfE.corr(method="pearson")
        spearman = dfE.corr(method="spearman")
        pearson.to_csv(OUT / f"error_corr_pearson_{scheme}.csv")
        spearman.to_csv(OUT / f"error_corr_spearman_{scheme}.csv")
        cancel = cancel_num / cancel_den if cancel_den > 0 else np.nan
        print(f"\nscheme={scheme}")
        print(f"δ-error Pearson (stacked over {len(oof['seed'].unique())} seeds):")
        print(pearson.round(3).to_string())
        print(f"Error cancellation ratio (|Σe| / Σ|e|) = {cancel:.4f}")
        if "oi_dft" in dfE.columns and "elst_dft" in dfE.columns:
            r_elst_oi = pearson.loc["elst_dft", "oi_dft"]
            print(f"  (elst-oi δ-error corr = {r_elst_oi:.3f}; v9 observed +0.777)")
        with open(OUT / f"cancellation_{scheme}.txt", "w") as f:
            f.write(f"cancel_ratio={cancel:.4f} for scheme={scheme}\n")


if __name__ == "__main__":
    main()
