"""SPEC_07 — figures for the λ-contribution sweep.

Uses results/lambda_curve.csv + results/pooled_oof.parquet. Renders three PNGs
under spec/spec07_lambda_contribution/figures/:

  lambda_nmae.png         — NMAE(λ) per channel + barrier, CI bands, λ*, spec06 ref line
  lambda_rmse.png         — RMSE(λ) per channel + barrier, CI bands
  parity_at_lamstar.png   — 5-channel + barrier parity grid at barrier's λ*

Render on compute node (per CLAUDE.md).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec07_lambda_contribution"
OUT_RES = SPEC / "results"
OUT_FIG = SPEC / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
CH_COLORS = {
    "strain":  "#4b779a",
    "Pauli":   "#a83232",
    "elst":    "#3e8548",
    "oi":      "#c05e2b",
    "disp":    "#7d3d7a",
    "barrier": "#111111",
}


def plot_metric_curve(curve_df, metric, ylabel, path):
    lambdas = sorted(curve_df["lambda"].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ch in CHANNELS + ["barrier"]:
        sub = curve_df[(curve_df.channel == ch) & (curve_df.metric == metric)]
        sub = sub.set_index("lambda").loc[lambdas]
        pts = sub["point"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        lw = 2.4 if ch == "barrier" else 1.4
        alpha_band = 0.10 if ch == "barrier" else 0.06
        ax.plot(lambdas, pts, "-o", color=CH_COLORS[ch], lw=lw, ms=5,
                label=ch)
        ax.fill_between(lambdas, lo, hi, color=CH_COLORS[ch], alpha=alpha_band,
                        linewidth=0)

    ax.set_xlabel("λ  (0 = pure δ, 1 = pure b)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parity_at_lamstar(pooled_df, path):
    star_json = OUT_RES / "lambda_star.json"
    if not star_json.exists():
        print("[parity] no lambda_star.json; skipping"); return
    st = json.loads(star_json.read_text())
    lam = st["barrier"]["lambda"]
    df = pooled_df[np.isclose(pooled_df["lam"], lam)]
    if not len(df):
        print(f"[parity] no rows at λ={lam}; skipping"); return

    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()

    channels_plot = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(1, len(channels_plot),
                             figsize=(3.4 * len(channels_plot), 3.6))
    for ci, ch in enumerate(channels_plot):
        ax = axes[ci]
        if ch == "barrier":
            a = yt.sum(1); b = yp.sum(1)
        else:
            i_ = CHANNELS.index(ch); a = yt[:, i_]; b = yp[:, i_]
        color = CH_COLORS[ch]
        ax.scatter(a, b, s=6, c=color, alpha=0.55, edgecolor="none")
        lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.6)
        mad = float(np.mean(np.abs(a - a.mean())))
        nm = float(np.mean(np.abs(a - b)) / (mad + 1e-12))
        r_ss = np.sum((a - b) ** 2); tot = np.sum((a - a.mean()) ** 2)
        r2v = float(1 - r_ss / (tot + 1e-12))
        ax.text(0.03, 0.97, f"NMAE={nm:.2f}\nR²={r2v:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_title(ch, fontsize=10)
        if ci == 0:
            ax.set_ylabel(f"ŷ  at λ*={lam:.2f}", fontsize=9)
        ax.set_xlabel("y_true", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    curve_df = pd.read_csv(OUT_RES / "lambda_curve.csv")
    pooled = pd.read_parquet(OUT_RES / "pooled_oof.parquet")
    plot_metric_curve(curve_df, "NMAE", "NMAE", OUT_FIG / "lambda_nmae.png")
    plot_metric_curve(curve_df, "RMSE", "RMSE", OUT_FIG / "lambda_rmse.png")
    parity_at_lamstar(pooled, OUT_FIG / "parity_at_lamstar.png")
    print(f"wrote figures under {OUT_FIG}")


if __name__ == "__main__":
    main()
