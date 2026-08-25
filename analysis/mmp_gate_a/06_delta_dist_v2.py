#!/usr/bin/env python3
"""Step 6 (v2): channel-δ + GATE-5-rev with cds excluded from learning."""
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
LEARNED = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm"]
EXCLUDED = ["cds"]
RANDOM_SEED = 42
N_BOOT = 1000


def bootstrap_ci(values, stat_fn, n_boot=N_BOOT, seed=RANDOM_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = stat_fn(rng.choice(values, size=n, replace=True))
    return (np.percentile(stats, (1 - ci) / 2 * 100), np.percentile(stats, (1 + ci) / 2 * 100))


def main():
    pairs = pd.read_pickle(OUT / "mmp_pairs_labeled_v2.pkl")
    print(f"MMP pairs (v2): {len(pairs)}")

    for ch in CHANNELS:
        pairs[f"delta_{ch}"] = pairs[f"{ch}_B"] - pairs[f"{ch}_A"]
    pairs["delta_G_act"] = pairs["G_act_B"] - pairs["G_act_A"]

    rows = []
    for ch in CHANNELS + ["G_act"]:
        col = f"delta_{ch}"
        d = pairs[pairs[f"has_{ch}"]][col].values if ch in CHANNELS else pairs[col].values
        if len(d) == 0:
            rows.append({"channel": ch, "n": 0, "excluded_from_learning": ch in EXCLUDED})
            continue
        ad = np.abs(d)
        med_lo, med_hi = bootstrap_ci(ad, np.median)
        rows.append({
            "channel": ch,
            "excluded_from_learning": ch in EXCLUDED,
            "n": len(d),
            "mean_abs": float(ad.mean()),
            "median_abs": float(np.median(ad)),
            "median_abs_ci_lo": med_lo,
            "median_abs_ci_hi": med_hi,
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
    stats.to_csv(OUT / "delta_channel_stats_v2.csv", index=False)
    print("\nPer-channel δ stats (v2, incl. G_act):")
    print(stats[["channel", "excluded_from_learning", "n", "mean_abs", "median_abs",
                 "frac_lt_1.0", "baseline_MAE"]].to_string(index=False))

    # GATE-5-rev: assess only LEARNED channels
    learned_stats = stats[stats["channel"].isin(LEARNED)]
    med = learned_stats["median_abs"].values
    if (med > 2.0).all():
        gate5 = "PASS"
    elif (med >= 1.0).all():
        gate5 = "CONDITIONAL"
    else:
        gate5 = "FAIL"
    with open(OUT / "GATE5_STATUS_v2.txt", "w") as f:
        f.write(f"{gate5} learned_channels={LEARNED} excluded={EXCLUDED}\n")
        f.write(learned_stats[["channel", "median_abs", "baseline_MAE"]].to_string(index=False))
        f.write("\n")
    print(f"\n=== GATE-5-rev: {gate5} (cds excluded; learned={LEARNED}) ===")

    # Cross-channel correlation on all-8-complete subset
    mask8 = pairs[[f"has_{ch}" for ch in CHANNELS]].all(axis=1)
    sub = pairs[mask8]
    print(f"\nAll-8 subset for correlation: {len(sub)}")
    if len(sub) >= 30:
        X = sub[[f"delta_{ch}" for ch in CHANNELS]].copy()
        X.columns = CHANNELS
        pearson = X.corr(method="pearson")
        spearman = X.corr(method="spearman")
        pearson.to_csv(OUT / "delta_corr_pearson_v2.csv")
        spearman.to_csv(OUT / "delta_corr_spearman_v2.csv")
        print("\nPearson (v2):")
        print(pearson.round(3).to_string())
    else:
        pearson = None

    if len(sub) > 0:
        sum_deltas = sum(sub[f"delta_{ch}"] for ch in CHANNELS)
        residual = sub["delta_G_act"] - sum_deltas
        print(f"\nSum-rule residual: n={len(sub)} mean={residual.mean():.3f} "
              f"median={residual.median():.3f} std={residual.std():.3f}")

    # Figures
    if len(sub) >= 30:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ch in enumerate(CHANNELS):
            ax = axes[i // 4, i % 4]
            d = pairs[pairs[f"has_{ch}"]][f"delta_{ch}"].values
            if len(d) == 0:
                continue
            ax.hist(d, bins=60, color="#4c9fd8" if ch in LEARNED else "#c0c0c0",
                    edgecolor="black", linewidth=0.3)
            ax.axvline(0, color="k", linewidth=1)
            ax.axvline(1, color="r", linewidth=1, linestyle="--")
            ax.axvline(-1, color="r", linewidth=1, linestyle="--")
            excl_tag = " [EXCLUDED]" if ch in EXCLUDED else ""
            ax.set_title(f"{ch}{excl_tag}: n={len(d)} MAE={np.abs(d).mean():.2f} med={np.median(np.abs(d)):.2f}")
            ax.set_xlabel("δ (kcal/mol)")
        fig.tight_layout()
        fig.savefig(FIG / "delta_distributions_v2.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(pearson.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CHANNELS))); ax.set_xticklabels(CHANNELS, rotation=45)
        ax.set_yticks(range(len(CHANNELS))); ax.set_yticklabels(CHANNELS)
        for i in range(len(CHANNELS)):
            for j in range(len(CHANNELS)):
                ax.text(j, i, f"{pearson.values[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(pearson.values[i,j]) > 0.5 else "black", fontsize=8)
        ax.set_title(f"Pearson δ correlation (v2, n={len(sub)} all-8 pairs)")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIG / "delta_corr_heatmap_v2.png", dpi=120)
        plt.close(fig)
        print("Figures saved (v2)")


if __name__ == "__main__":
    main()
