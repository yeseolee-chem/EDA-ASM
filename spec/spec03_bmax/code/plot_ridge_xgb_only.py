"""Replot SPEC_03 baseline_bars with only ridge and xgb (drop lasso/enet).

Reads spec/spec03_bmax/results/baseline_leaderboard.csv and writes
figures/baseline_bars_ridge_xgb.png. Purely a plotting pass — no refitting.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RES = REPO / "spec/spec03_bmax/results/baseline_leaderboard.csv"
FIG = REPO / "spec/spec03_bmax/figures/baseline_bars_ridge_xgb.png"

CHANNELS_PLOT = ["strain", "Pauli", "elst", "oi", "disp", "barrier_sum"]
METHODS = [("ridge", "#1f77b4"), ("xgb", "#d62728")]


def main():
    lb = pd.read_csv(RES)
    x = np.arange(len(CHANNELS_PLOT))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (m, color) in enumerate(METHODS):
        vals = [lb[(lb.model == m) & (lb.channel == ch)].NMAE.iloc[0]
                for ch in CHANNELS_PLOT]
        ax.bar(x + (i - 0.5) * width, vals, width, label=m,
               color=color, edgecolor="white", lw=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS_PLOT)
    ax.set_ylabel("NMAE (5-fold CV)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
