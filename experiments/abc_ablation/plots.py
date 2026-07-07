"""Plots for the A/B/C ablation.

Three deliverables:
  1. nmae_bars.png      — 3 arms × (5 channels + barrier) grouped bar + 95% CI.
  2. parity_grid.png    — arm(row) × channel(col) parity; NMAE/R²/slope in-panel.
  3. rmse_bars.png      — same layout as (1) but RMSE.

Style: navy/white minimal, matches existing figures/*.png style.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
ARMS = ["A", "B", "C"]
ARM_LABEL = {"A": "A · xgb_direct", "B": "B · ridge+δ", "C": "C · xgb+δ"}
ARM_COLOR = {"A": "#c25a5a", "B": "#1f4e79", "C": "#2b8a89"}


def _load_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "abc_metrics.csv")


def _load_oof(arm: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"oof_pred_{arm}.parquet")


def plot_bars(metric: str, out_name: str, ylabel: str) -> None:
    df = _load_metrics()
    channels = CH + ["barrier"]
    x = np.arange(len(channels))
    n_arm = len(ARMS)
    width = 0.8 / n_arm

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, arm in enumerate(ARMS):
        sub = df[df.arm == arm].set_index("channel")
        if sub.empty:
            continue
        vals = np.array([sub.loc[c, metric] for c in channels])
        lo = np.array([sub.loc[c, f"{metric}_lo"] for c in channels])
        hi = np.array([sub.loc[c, f"{metric}_hi"] for c in channels])
        yerr = np.stack([vals - lo, hi - vals], axis=0)
        ax.bar(x + (i - (n_arm - 1) / 2) * width, vals, width,
               yerr=yerr, capsize=2, color=ARM_COLOR[arm],
               edgecolor="white", linewidth=0.4, label=ARM_LABEL[arm])
    if metric == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    out = RESULTS / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote → {out}", flush=True)


def plot_parity_grid() -> None:
    df_metrics = _load_metrics()
    fig, axes = plt.subplots(len(ARMS), len(CH), figsize=(3.1 * len(CH), 3.1 * len(ARMS)),
                             sharex=False, sharey=False)
    for r, arm in enumerate(ARMS):
        try:
            oof = _load_oof(arm)
        except FileNotFoundError:
            continue
        for c, ch in enumerate(CH):
            ax = axes[r, c]
            y = oof[f"y_{ch}"].values
            p = oof[f"yhat_{ch}"].values
            ax.scatter(y, p, s=6, alpha=0.35, color=ARM_COLOR[arm], edgecolor="none")
            lo = float(min(y.min(), p.min()))
            hi = float(max(y.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], "-", color="black", lw=0.6, alpha=0.6)
            row = df_metrics[(df_metrics.arm == arm) & (df_metrics.channel == ch)].iloc[0]
            ax.text(0.03, 0.97,
                    f"NMAE {row['NMAE']:.2f}\nR² {row['R2']:.2f}\nslope {row['slope']:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(ch, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{ARM_LABEL[arm]}\n$\\hat{{y}}$", fontsize=9)
            if r == len(ARMS) - 1:
                ax.set_xlabel("y (kcal/mol)")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    out = RESULTS / "parity_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote → {out}", flush=True)


def main() -> None:
    plot_bars("NMAE", "nmae_bars.png", "NMAE (lower is better)")
    plot_bars("RMSE", "rmse_bars.png", "RMSE (kcal/mol)")
    plot_parity_grid()


if __name__ == "__main__":
    main()
