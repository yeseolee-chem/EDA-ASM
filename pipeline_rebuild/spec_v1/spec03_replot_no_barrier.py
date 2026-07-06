"""Replot SPEC_03 baseline_bars without the barrier_sum / barrier_direct columns.

Reads results/spec03_bmax/baseline_leaderboard.csv (already committed) and
writes results/spec03_bmax/baseline_bars_no_barrier.png. The original
baseline_bars.png (with barrier columns) is not touched.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
OUT = REPO / "results" / "spec03_bmax"

CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
METHODS = ["ridge", "lasso", "enet", "xgb"]
METHOD_COLORS = {"ridge": "#1f4e79", "lasso": "#2b8a89",
                 "enet": "#d6a13b", "xgb": "#c25a5a"}


def main():
    df = pd.read_csv(OUT / "baseline_leaderboard.csv")
    channels_bar = list(CH)  # 5 EDA channels only; barrier columns dropped
    x = np.arange(len(channels_bar))
    width = 0.8 / len(METHODS)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(METHODS):
        color = METHOD_COLORS[name]
        means = [df[(df.method == name) & (df.channel == ch)].NMAE.mean() for ch in channels_bar]
        stds = [df[(df.method == name) & (df.channel == ch)].NMAE.std() for ch in channels_bar]
        ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, means, width, yerr=stds,
               label=name, color=color, capsize=2, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_ylabel("NMAE")
    ax.set_xticks(x)
    ax.set_xticklabels(channels_bar)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out_path = OUT / "baseline_bars_no_barrier.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
