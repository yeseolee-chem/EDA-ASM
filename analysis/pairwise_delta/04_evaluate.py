#!/usr/bin/env python3
"""Step 4: evaluate arms - main deliverable + GATE-4 + GATE-5.

Uses predictions_all.pkl from Step 2.
Baseline (Arm-0) MAE_δ read from delta_mae_baseline/artifacts/delta_mae_table.csv.

GATE-4: per channel verdict
GATE-5: swaptype_holdout / component ratio (generalization test)
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta")
OUT = BASE / "artifacts"
DMB = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts")

LEARNED = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
           "oi_dft", "disp_dft", "cpcm_dft"]
N_BOOT = 1000
RNG_SEED = 42
BINS = [0, 1, 2, 3, 5, 10, np.inf]


def bootstrap_mean(x, n_boot=N_BOOT, seed=RNG_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    stats = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(stats, (1 - ci) / 2 * 100)), float(np.percentile(stats, (1 + ci) / 2 * 100))


def main():
    preds = pd.read_pickle(OUT / "predictions_all.pkl")
    print(f"predictions_all: {len(preds)} rows")

    # Arm-0 reference (subtract baseline from delta_mae_baseline)
    ref = pd.read_csv(DMB / "delta_mae_table.csv")
    ref_comp = ref[ref["scheme"] == "groupkfold_component"].set_index("channel")

    # Per (arm, split, seed, channel): MAE_δ + metrics
    preds["err"] = np.abs(preds["delta_pred"] - preds["delta_true"])
    per = preds.groupby(["arm", "split", "seed", "channel"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "mae_delta": g["err"].mean(),
            "median_err": g["err"].median(),
            "sign_acc": float((np.sign(g["delta_pred"]) == np.sign(g["delta_true"])).mean()),
            "spearman": float(spearmanr(g["delta_pred"], g["delta_true"])[0]),
            "slope": float(np.polyfit(g["delta_true"], g["delta_pred"], 1)[0]),
        }), include_groups=False
    ).reset_index()
    per.to_csv(OUT / "per_seed_metrics.csv", index=False)

    # Seed-averaged
    agg = per.groupby(["arm", "split", "channel"]).agg(
        n=("n", "first"),
        mae_delta=("mae_delta", "mean"),
        mae_delta_std=("mae_delta", "std"),
        median_err=("median_err", "mean"),
        sign_acc=("sign_acc", "mean"),
        spearman=("spearman", "mean"),
        slope=("slope", "mean"),
    ).reset_index()

    # Attach Arm-0 baseline (only for split==component; for swaptype we still show it as ref)
    agg["arm0_mae_delta"] = agg["channel"].map(ref_comp["mae_delta"])
    agg["arm0_baseline_delta_zero"] = agg["channel"].map(ref_comp["baseline_mae_delta"])
    agg["improve_vs_arm0"] = 1.0 - agg["mae_delta"] / agg["arm0_mae_delta"]
    agg["improve_vs_zero"] = 1.0 - agg["mae_delta"] / agg["arm0_baseline_delta_zero"]

    def verdict(row):
        if row["mae_delta"] < 1.0:
            return "A_GOAL"
        if row["mae_delta"] < row["arm0_mae_delta"]:
            return "B_IMPROVED"
        return "C_WORSE"

    agg["verdict"] = agg.apply(verdict, axis=1)
    agg.to_csv(OUT / "delta_mae_by_arm.csv", index=False)
    print("\nMAE_δ by (arm, split, channel) — seed averaged:")
    show_cols = ["arm", "split", "channel", "mae_delta", "arm0_mae_delta",
                 "improve_vs_arm0", "sign_acc", "spearman", "slope", "verdict"]
    print(agg[show_cols].to_string(index=False))

    # Resolution table — sign accuracy binned by |δ_true|
    preds["abs_true"] = preds["delta_true"].abs()
    preds["bin"] = pd.cut(preds["abs_true"], bins=BINS, right=False,
                          labels=[f"[{BINS[i]},{BINS[i+1]})" for i in range(len(BINS) - 1)])
    preds["sign_hit"] = (np.sign(preds["delta_pred"]) == np.sign(preds["delta_true"])).astype(int)
    resol = preds.groupby(["arm", "split", "channel", "bin"], observed=False).agg(
        n=("sign_hit", "size"),
        sign_acc=("sign_hit", "mean"),
    ).reset_index()
    resol.to_csv(OUT / "resolution_table.csv", index=False)

    # Per-swap-type performance (aggregated over seeds & folds), for split=swaptype_holdout
    pairs_meta = pickle.load(open(OUT / "pair_dataset.pkl", "rb"))["pairs"]
    preds = preds.merge(pairs_meta[["swap_type"]].reset_index().rename(columns={"index": "pair_idx"}),
                        on="pair_idx", how="left")
    sw_perf = preds[preds["split"] == "swaptype_holdout"].groupby(
        ["arm", "channel", "swap_type"]).agg(
        n=("err", "size"),
        mae_delta=("err", "mean"),
    ).reset_index()
    sw_perf.to_csv(OUT / "swap_type_performance.csv", index=False)

    # GATE-5: swaptype_holdout / component MAE ratio
    piv = agg.pivot_table(index=["arm", "channel"], columns="split", values="mae_delta").reset_index()
    piv["ratio_swap_over_comp"] = piv["swaptype_holdout"] / piv["component"]
    piv.to_csv(OUT / "swaptype_generalization.csv", index=False)
    print("\nGeneralization ratio (swap_holdout / component):")
    print(piv.to_string(index=False))

    # GATE-4 assessment: per channel per arm - count of A_GOAL and B_IMPROVED under Arm-C component
    comp_C = agg[(agg["arm"] == "C_symmetric") & (agg["split"] == "component")]
    goal_hits = (comp_C["mae_delta"] < 1.0).sum()
    improved = (comp_C["mae_delta"] < comp_C["arm0_mae_delta"]).sum()
    with open(OUT / "GATE4_STATUS.txt", "w") as f:
        f.write(f"C_symmetric/component: goal_hits(A)={goal_hits}/7 improved_vs_arm0(B+)={improved}/7\n")
    print(f"\n=== GATE-4: Arm-C goal_hits={goal_hits}/7 improved={improved}/7 ===")

    # GATE-5 (Arm-C, per channel)
    piv_C = piv[piv["arm"] == "C_symmetric"]
    bad = piv_C[piv_C["ratio_swap_over_comp"] > 2.0]
    with open(OUT / "GATE5_STATUS.txt", "w") as f:
        if len(bad) == 0:
            f.write("PASS all channels ratio<=2 (Arm-C)\n")
        else:
            f.write(f"WARN {len(bad)} channels ratio>2 for Arm-C: {list(bad['channel'])}\n")


if __name__ == "__main__":
    main()
