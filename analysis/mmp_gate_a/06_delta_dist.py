#!/usr/bin/env python3
"""Step 6: channel-wise delta distributions - Q2 answer + GATE-5.

For each pair × each channel: delta_c = y_c(B) - y_c(A).
Baseline MAE(c) = mean|delta_c| (this is what a "predict δ=0" model achieves).
Cross-channel correlation on the all-8-channel subset only.
Sum-rule check: residual = delta_G_act - Σ delta_c (weak — G_act includes solvent etc).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds"]
RANDOM_SEED = 42
N_BOOT = 1000


def bootstrap_ci(values, stat_fn, n_boot=N_BOOT, seed=RANDOM_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        stats[i] = stat_fn(sample)
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return (lo, hi)


def main():
    pairs = pd.read_pickle(OUT / "mmp_pairs_labeled.pkl")
    print(f"MMP pairs: {len(pairs)}")

    # Compute deltas for each channel
    for ch in CHANNELS:
        pairs[f"delta_{ch}"] = pairs[f"{ch}_B"] - pairs[f"{ch}_A"]

    # Per-channel stats
    rows = []
    for ch in CHANNELS:
        d = pairs[pairs[f"has_{ch}"]][f"delta_{ch}"].values
        if len(d) == 0:
            rows.append({"channel": ch, "n": 0})
            continue
        ad = np.abs(d)
        med_ci = bootstrap_ci(ad, np.median)
        rows.append({
            "channel": ch,
            "n": len(d),
            "mean_abs": float(ad.mean()),
            "median_abs": float(np.median(ad)),
            "median_abs_ci_lo": med_ci[0],
            "median_abs_ci_hi": med_ci[1],
            "std_signed": float(d.std(ddof=1)),
            "iqr_signed": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "p10_signed": float(np.percentile(d, 10)),
            "p25_signed": float(np.percentile(d, 25)),
            "p75_signed": float(np.percentile(d, 75)),
            "p90_signed": float(np.percentile(d, 90)),
            "max_abs": float(ad.max()),
            "frac_lt_0.5": float((ad < 0.5).mean()),
            "frac_lt_1.0": float((ad < 1.0).mean()),
            "frac_lt_2.0": float((ad < 2.0).mean()),
            "baseline_MAE": float(ad.mean()),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "delta_channel_stats.csv", index=False)
    print("\nPer-channel delta stats (Q2 answer):")
    print(stats.to_string(index=False))

    # GATE-5 assessment
    with open(OUT / "GATE5_STATUS.txt", "w") as f:
        med = stats["median_abs"].values
        if (med > 2.0).all():
            status = "PASS"
        elif (med < 1.0).all():
            status = "FAIL"
        else:
            status = "WARN"
        f.write(f"{status}\n")
        f.write(stats[["channel", "median_abs", "baseline_MAE"]].to_string(index=False))
        f.write("\n")
    print(f"\n=== GATE-5: {status} ===")

    # Cross-channel correlation on all-8-complete subset
    mask_all8 = pairs[[f"has_{ch}" for ch in CHANNELS]].all(axis=1)
    sub = pairs[mask_all8]
    print(f"\nAll-8-channel complete pairs for correlation: {len(sub)}")
    if len(sub) >= 30:
        X = sub[[f"delta_{ch}" for ch in CHANNELS]]
        pearson = X.corr(method="pearson")
        spearman = X.corr(method="spearman")
        pearson.to_csv(OUT / "delta_corr_pearson.csv")
        spearman.to_csv(OUT / "delta_corr_spearman.csv")
        print("\nPearson correlation matrix (delta, all-8 subset):")
        print(pearson.round(3).to_string())
    else:
        print("Not enough all-8 pairs for correlation matrix")

    # Sum rule
    if len(sub) > 0:
        sum_deltas = sum(sub[f"delta_{ch}"] for ch in CHANNELS)
        residual = sub["delta_G_act"] - sum_deltas
        print(f"\nSum-rule residual (delta_G_act - Σdelta_c):")
        print(f"  n={len(sub)}  mean={residual.mean():.3f}  median={residual.median():.3f}  "
              f"std={residual.std():.3f}  |median|={residual.abs().median():.3f}")
        print("  (non-zero expected: G_act is Gibbs w/ solvation-inclusive kinetics)")

    # Figures
    if len(sub) >= 30:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ch in enumerate(CHANNELS):
            ax = axes[i // 4, i % 4]
            d = pairs[pairs[f"has_{ch}"]][f"delta_{ch}"].values
            if len(d) == 0:
                ax.set_title(f"{ch}: n=0")
                continue
            ax.hist(d, bins=60, color="#4c9fd8", edgecolor="black", linewidth=0.3)
            ax.axvline(0, color="k", linewidth=1, linestyle="-")
            ax.axvline(1, color="r", linewidth=1, linestyle="--", label="±1 kcal/mol")
            ax.axvline(-1, color="r", linewidth=1, linestyle="--")
            ax.set_title(f"{ch}: n={len(d)}  MAE={np.abs(d).mean():.2f}  med|δ|={np.median(np.abs(d)):.2f}")
            ax.set_xlabel("δ (kcal/mol)")
            ax.set_ylabel("count")
            if i == 0:
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "delta_distributions.png", dpi=120)
        plt.close(fig)
        print("Saved figures/delta_distributions.png")

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(pearson.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CHANNELS))); ax.set_xticklabels(CHANNELS, rotation=45)
        ax.set_yticks(range(len(CHANNELS))); ax.set_yticklabels(CHANNELS)
        for i in range(len(CHANNELS)):
            for j in range(len(CHANNELS)):
                ax.text(j, i, f"{pearson.values[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(pearson.values[i,j]) > 0.5 else "black", fontsize=8)
        ax.set_title(f"Pairwise δ Pearson correlation (n={len(sub)} all-8 pairs)")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIG / "delta_corr_heatmap.png", dpi=120)
        plt.close(fig)
        print("Saved figures/delta_corr_heatmap.png")


if __name__ == "__main__":
    main()
