"""V1 Claisen 15-substrate Hammett + per-channel EDA regression.

Reads V1/outputs/v1_claisen_asr.parquet (15 rows: id, sigma_p, F, R_res,
dE_barrier_wb97x3c, dE_strain, dV_elst, dE_Pauli, dE_oi, dE_disp) and
produces:

  figures/hammett_barrier.png            Ea vs sigma_p, linear fit
  figures/hammett_per_channel_grid.png   6-panel Δ(channel) vs sigma_p
  figures/swain_lupton_barrier.png       Ea vs (F, R_res) dual-parameter
  figures/channel_correlation.png        pairwise |r| heatmap
  results/regression_summary.csv         slope, intercept, R², r, p per channel
  results/regression_summary.md          human-readable table

Interpretation:
- The slope of Ea vs σ_p is the Hammett reaction constant ρ (in kcal/mol
  per σ unit). ρ > 0 ⇒ TS destabilized by electron-donors (rate faster
  for acceptors); ρ < 0 ⇒ opposite.
- Regressing each EDA channel separately tells us *which channel* is
  most correlated with σ_p — i.e. which physical interaction carries the
  substituent effect.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
DATA = V1 / "outputs" / "v1_claisen_asr.parquet"
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Channel column names (physical sign preserved from the parquet)
CHANNELS = [
    ("dE_barrier_wb97x3c", "ΔE‡ (wB97X-3c)", "kcal/mol"),
    ("dE_strain",          "ΔE_strain",       "kcal/mol"),
    ("dV_elst",            "ΔV_elst",         "kcal/mol"),
    ("dE_Pauli",           "ΔE_Pauli",        "kcal/mol"),
    ("dE_oi",              "ΔE_oi",           "kcal/mol"),
    ("dE_disp",            "ΔE_disp",         "kcal/mol"),
]


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    keep = df["status"] == "OK"
    df = df[keep].sort_values("sigma_p").reset_index(drop=True)
    return df


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS y = a x + b, return slope/intercept/R²/Pearson r/p-value."""
    res = stats.linregress(x, y)
    return {
        "slope":     float(res.slope),
        "intercept": float(res.intercept),
        "r":         float(res.rvalue),
        "r2":        float(res.rvalue ** 2),
        "p":         float(res.pvalue),
        "stderr":    float(res.stderr),
    }


def plot_scatter(ax, x, y, labels, fit, title, ylabel):
    ax.scatter(x, y, s=40, c="#1E2761", alpha=0.85, edgecolor="white", zorder=3)
    xs = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    ys = fit["slope"] * xs + fit["intercept"]
    ax.plot(xs, ys, "-", color="#C45A4D", lw=1.6, zorder=2,
            label=f"ρ = {fit['slope']:+.2f}\nR² = {fit['r2']:.3f}")
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(lbl, (xi, yi), fontsize=7, color="#333",
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.5, alpha=0.5)
    ax.axvline(0, color="#999", lw=0.5, alpha=0.5)
    ax.set_xlabel("Hammett σₚ")
    ax.set_ylabel(f"{ylabel} (kcal/mol)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)


def figure_hammett_barrier(df: pd.DataFrame) -> dict:
    fit = fit_linear(df["sigma_p"].values, df["dE_barrier_wb97x3c"].values)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plot_scatter(
        ax,
        df["sigma_p"].values,
        df["dE_barrier_wb97x3c"].values,
        df["id"].values,
        fit,
        "V1 Claisen — Hammett plot (activation barrier)",
        "ΔE‡ (wB97X-3c)",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hammett_barrier.png", dpi=160)
    plt.close(fig)
    return fit


def figure_per_channel_grid(df: pd.DataFrame) -> dict:
    fits: dict[str, dict] = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, sym, unit) in zip(axes.flat, CHANNELS):
        fit = fit_linear(df["sigma_p"].values, df[col].values)
        fits[col] = fit
        plot_scatter(
            ax,
            df["sigma_p"].values,
            df[col].values,
            df["id"].values,
            fit,
            f"{sym} vs σₚ",
            sym,
        )
    fig.suptitle("V1 Claisen — per-channel EDA Hammett regression",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hammett_per_channel_grid.png", dpi=160)
    plt.close(fig)
    return fits


def figure_swain_lupton(df: pd.DataFrame) -> dict:
    """Dual-parameter fit: ΔE‡ = a F + b R_res + c."""
    X = np.column_stack([df["F"].values, df["R_res"].values, np.ones(len(df))])
    y = df["dE_barrier_wb97x3c"].values
    (a, b, c), _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ np.array([a, b, c])
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    fit = {"a_F": float(a), "b_R": float(b), "intercept": float(c), "r2": r2}

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y, y_pred, s=40, c="#1C7293", alpha=0.85, edgecolor="white",
               zorder=3)
    lo, hi = min(y.min(), y_pred.min()) - 1, max(y.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "--", color="#999", lw=0.8)
    for yi, yp, lbl in zip(y, y_pred, df["id"].values):
        ax.annotate(lbl, (yi, yp), fontsize=7, color="#333",
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("ΔE‡ observed (kcal/mol)")
    ax.set_ylabel("ΔE‡ predicted by Swain–Lupton (kcal/mol)")
    ax.set_title("V1 Claisen — Swain–Lupton dual-parameter fit\n"
                 f"ΔE‡ ≈ {a:+.2f}·F {b:+.2f}·R + {c:+.2f}   (R² = {r2:.3f})",
                 fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "swain_lupton_barrier.png", dpi=160)
    plt.close(fig)
    return fit


def figure_channel_correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sigma_p", "F", "R_res"] + [c[0] for c in CHANNELS]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:+.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title("V1 Claisen — pairwise Pearson correlations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "channel_correlation.png", dpi=160)
    plt.close(fig)
    return corr


def write_summary(hammett_fits: dict, dual_fit: dict, corr: pd.DataFrame):
    rows = []
    for col, sym, unit in CHANNELS:
        f = hammett_fits[col]
        rows.append({
            "channel":         col,
            "symbol":          sym,
            "slope_rho":       f["slope"],
            "intercept":       f["intercept"],
            "r":               f["r"],
            "r2":              f["r2"],
            "p_value":         f["p"],
            "stderr_slope":    f["stderr"],
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES_DIR / "regression_summary.csv", index=False)

    lines = ["# V1 Claisen — Hammett & EDA regression summary",
             "",
             "OLS fit of each channel against σₚ across the 15 para-substituents.",
             "Slope is the effective Hammett constant ρ (kcal · mol⁻¹ · σ⁻¹).",
             "",
             "| channel | ρ (kcal/mol per σ) | R² | Pearson r | p-value |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['symbol']} | {r['slope_rho']:+.3f} ± {r['stderr_slope']:.3f} "
            f"| {r['r2']:.3f} | {r['r']:+.3f} | {r['p_value']:.2e} |"
        )
    lines += [
        "",
        "## Swain–Lupton dual-parameter fit of ΔE‡",
        f"ΔE‡ ≈ {dual_fit['a_F']:+.3f} · F  {dual_fit['b_R']:+.3f} · R  + {dual_fit['intercept']:+.3f}",
        f"R² = {dual_fit['r2']:.3f}",
        "",
        "## Correlation matrix",
        "See `figures/channel_correlation.png`. Full matrix stored in",
        "`results/correlation_matrix.csv`.",
    ]
    (RES_DIR / "regression_summary.md").write_text("\n".join(lines))
    corr.to_csv(RES_DIR / "correlation_matrix.csv")


def main():
    df = load()
    print(f"Loaded {len(df)} rows from {DATA.name}")
    barrier_fit = figure_hammett_barrier(df)
    channel_fits = figure_per_channel_grid(df)
    dual_fit = figure_swain_lupton(df)
    corr = figure_channel_correlation(df)
    hammett_fits = {"dE_barrier_wb97x3c": barrier_fit, **channel_fits}
    write_summary(hammett_fits, dual_fit, corr)

    print(f"\nHammett fit (barrier):  ρ = {barrier_fit['slope']:+.3f}  "
          f"R² = {barrier_fit['r2']:.3f}  p = {barrier_fit['p']:.2e}")
    print(f"Swain–Lupton:  a_F = {dual_fit['a_F']:+.3f}, "
          f"b_R = {dual_fit['b_R']:+.3f}, R² = {dual_fit['r2']:.3f}")
    print("\nPer-channel Hammett constants (sorted by |ρ|):")
    ranked = sorted(channel_fits.items(),
                    key=lambda kv: abs(kv[1]["slope"]), reverse=True)
    for col, f in ranked:
        print(f"  {col:<20s} ρ = {f['slope']:+.3f}  R² = {f['r2']:.3f}  "
              f"r = {f['r']:+.3f}")
    print(f"\nFigures  → {FIG_DIR}")
    print(f"Results  → {RES_DIR}")


if __name__ == "__main__":
    main()
