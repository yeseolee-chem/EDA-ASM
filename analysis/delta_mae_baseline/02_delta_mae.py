#!/usr/bin/env python3
"""Step 2: δ-MAE per channel — the Q answer (§3.2 of SPEC).

For each MMP pair (deduped, N=1936), each learned channel:
  δ_true = y_true(B) - y_true(A)
  δ_pred = y_pred(B) - y_pred(A)
  err_δ  = |δ_pred - δ_true|

Report seed-averaged MAE_δ per (scheme, channel) with bootstrap CI + verdict tier.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline")
OUT = BASE / "artifacts"

TARGETS = [
    "d1_own_dft", "d2_own_dft",
    "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft",
    "interaction_own_dft",
]
LEARNED = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft", "oi_dft",
           "disp_dft", "cpcm_dft"]  # cds excluded from learning
TARGET_MAE = 1.0
N_BOOT = 1000
RNG_SEED = 42


def verdict(mae_d, baseline):
    if mae_d < 1.0:
        return "A_GOAL"
    if mae_d < 2.0:
        return "B_NEAR"
    if mae_d < baseline:
        return "C_LIMITED"
    return "D_WORSE_THAN_ZERO"


def bootstrap_ci(x, fn, n_boot=N_BOOT, seed=RNG_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    stats = np.array([fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(stats, (1 - ci) / 2 * 100)), float(np.percentile(stats, (1 + ci) / 2 * 100))


def main():
    oof = pd.read_pickle(OUT / "oof_predictions.pkl")
    pairs = pd.read_pickle(OUT / "pairs_dedup.pkl")
    print(f"OOF rows: {len(oof)}  pairs: {len(pairs)}")

    # Index oof by (scheme, seed, reaction_number) for fast lookup
    rows = []
    for scheme in oof["scheme"].unique():
        for seed in sorted(oof["seed"].unique()):
            sub = oof[(oof["scheme"] == scheme) & (oof["seed"] == seed)].set_index("reaction_number")
            for tgt in TARGETS:
                yA_t = pairs["rxn_id_A"].map(sub[f"{tgt}_true"])
                yB_t = pairs["rxn_id_B"].map(sub[f"{tgt}_true"])
                yA_p = pairs["rxn_id_A"].map(sub[f"{tgt}_pred"])
                yB_p = pairs["rxn_id_B"].map(sub[f"{tgt}_pred"])
                d_true = (yB_t - yA_t).values
                d_pred = (yB_p - yA_p).values
                err_d = np.abs(d_pred - d_true)
                mae_d = float(err_d.mean())
                median_err_d = float(np.median(err_d))
                mae_d_lo, mae_d_hi = bootstrap_ci(err_d, np.mean)
                baseline = float(np.abs(d_true).mean())
                improvement = 1.0 - mae_d / baseline if baseline > 0 else np.nan
                sign_acc = float((np.sign(d_pred) == np.sign(d_true)).mean())
                sp_rho, _ = spearmanr(d_pred, d_true)
                rows.append({
                    "scheme": scheme, "seed": seed, "channel": tgt,
                    "n_pairs": len(err_d),
                    "abs_mae": float(np.mean(np.abs(sub[f"{tgt}_pred"] - sub[f"{tgt}_true"]))),
                    "baseline_mae_delta": baseline,
                    "mae_delta": mae_d,
                    "mae_delta_ci_lo": mae_d_lo,
                    "mae_delta_ci_hi": mae_d_hi,
                    "median_err_delta": median_err_d,
                    "improvement_over_zero": improvement,
                    "sign_accuracy": sign_acc,
                    "spearman_delta": float(sp_rho),
                    "target_vs_1kcal": mae_d / TARGET_MAE,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "delta_mae_per_seed.csv", index=False)

    # Seed-averaged summary
    agg = df.groupby(["scheme", "channel"]).agg(
        n_pairs=("n_pairs", "first"),
        abs_mae=("abs_mae", "mean"),
        baseline_mae_delta=("baseline_mae_delta", "mean"),
        mae_delta=("mae_delta", "mean"),
        mae_delta_std=("mae_delta", "std"),
        median_err_delta=("median_err_delta", "mean"),
        improvement_over_zero=("improvement_over_zero", "mean"),
        sign_accuracy=("sign_accuracy", "mean"),
        spearman_delta=("spearman_delta", "mean"),
        target_vs_1kcal=("target_vs_1kcal", "mean"),
    ).reset_index()
    agg["verdict"] = agg.apply(lambda r: verdict(r["mae_delta"], r["baseline_mae_delta"]), axis=1)
    agg["excluded_from_learning"] = ~agg["channel"].isin(LEARNED)
    agg.to_csv(OUT / "delta_mae_table.csv", index=False)
    print("\nδ-MAE per (scheme, channel) — seed averaged:")
    show = agg[["scheme", "channel", "abs_mae", "baseline_mae_delta", "mae_delta",
                "improvement_over_zero", "sign_accuracy", "verdict", "excluded_from_learning"]]
    print(show.to_string(index=False))

    # GATE-2: H4 — mae_delta < sqrt(2) * abs_mae for all channels
    sqrt2 = np.sqrt(2)
    violations = agg[agg["mae_delta"] >= sqrt2 * agg["abs_mae"]]
    with open(OUT / "GATE2_STATUS.txt", "w") as f:
        if len(violations) == 0:
            f.write("PASS mae_delta < sqrt(2)*abs_mae for all (scheme,channel)\n")
        else:
            f.write("FAIL violations:\n")
            for _, v in violations.iterrows():
                f.write(f"  {v['scheme']} {v['channel']}: mae_d={v['mae_delta']:.3f} "
                        f">= sqrt2*abs_mae={sqrt2*v['abs_mae']:.3f}\n")
    print(f"\n=== GATE-2: {'PASS' if len(violations)==0 else 'FAIL'} "
          f"({len(violations)} violations) ===")


if __name__ == "__main__":
    main()
