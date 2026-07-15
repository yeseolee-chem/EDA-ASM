"""SPEC_03 RMSE bar plots (4-model + 2-model variants).

Reads spec/spec03_bmax/results/baseline_leaderboard.csv (no refitting).
Writes to spec/spec03_bmax/figures/:
  - baseline_bars_rmse.png              (ridge, lasso, enet, xgb)
  - baseline_bars_rmse_ridge_xgb.png    (ridge, xgb only)

disp is on its own y-axis panel because its RMSE (~1.6-2.0 kcal/mol) is
~30x smaller than Pauli (~36-54 kcal/mol) and would otherwise be
invisible. Barrier_sum also gets its own panel to preserve channel order.
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
FIG_DIR = REPO / "spec/spec03_bmax/figures"

# Match spec03_bmax.py default color cycle for the 4-model variant.
METHODS_4 = [
    ("ridge", "#1f77b4"),
    ("lasso", "#ff7f0e"),
    ("enet",  "#2ca02c"),
    ("xgb",   "#d62728"),
]
METHODS_2 = [
    ("ridge", "#1f77b4"),
    ("xgb",   "#d62728"),
]

# Preserve natural channel order; disp gets its own y-scale.
GROUPS = [
    ["strain", "Pauli", "elst", "oi"],
    ["disp"],
    ["barrier_sum"],
]


def bar_plot_split_disp(lb, methods, out_path):
    fig, axes = plt.subplots(
        1, len(GROUPS), figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [len(g) for g in GROUPS], "wspace": 0.28},
    )
    width = 0.8 / len(methods)  # total ~0.8 of x-slot width regardless of n_methods
    offset0 = -(len(methods) - 1) / 2.0
    for ax, chans in zip(axes, GROUPS):
        x = np.arange(len(chans))
        for i, (m, color) in enumerate(methods):
            vals = [lb[(lb.model == m) & (lb.channel == ch)].RMSE.iloc[0]
                    for ch in chans]
            ax.bar(x + (offset0 + i) * width, vals, width, label=m,
                   color=color, edgecolor="white", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(chans)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("RMSE (kcal/mol, 5-fold CV)")
    axes[1].set_title("(independent y-scale)", fontsize=9, color="gray")
    axes[0].legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    lb = pd.read_csv(RES)
    bar_plot_split_disp(lb, METHODS_4, FIG_DIR / "baseline_bars_rmse.png")
    bar_plot_split_disp(lb, METHODS_2, FIG_DIR / "baseline_bars_rmse_ridge_xgb.png")


if __name__ == "__main__":
    main()
