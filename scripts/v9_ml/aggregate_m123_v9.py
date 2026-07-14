"""Aggregate m1/m2/m3 v9 (783-rxn) cells into 3 consolidated figures.

Reads:
  m1/code/trackB_lowlr_v9_geom6/m1_delta/fold{0..4}/member{0..4}.json
  m2/code/trackB_lowlr_v9_xtb_geom6/m2_delta/fold{0..4}/member{0..4}.json
  m3/code/trackB_lowlr_v9_xtb_geom6_plus_v2/m3_delta/fold{0..4}/member{0..4}.json

Writes (no top titles per request):
  comparison/v9/figures/nmae_bar.png     - 3 models side-by-side, 6 channels
  comparison/v9/figures/rmse_bar.png     - same
  comparison/v9/figures/parity_grid.png  - 3 rows (m1/m2/m3) x 6 cols
  comparison/v9/results/per_cell_metrics.csv
  comparison/v9/results/summary_per_model.csv
  comparison/v9/README.md
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
OUT = REPO / "comparison" / "v9"
FIG = OUT / "figures"
RES = OUT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("m1", "m1/code/trackB_lowlr_v9_geom6/m1_delta",              "#1f2b6b"),
    ("m2", "m2/code/trackB_lowlr_v9_xtb_geom6/m2_delta",          "#227b8f"),
    ("m3", "m3/code/trackB_lowlr_v9_xtb_geom6_plus_v2/m3_delta",  "#c26f6b"),
]
CHANNELS = ["strain", "Pauli", "Velst", "oi", "disp"]
CHANNELS_BAR = CHANNELS + ["barrier"]
N = 783


def load_cells(root):
    cells = []
    for f in range(5):
        for m in range(5):
            p = REPO / root / f"fold{f}" / f"member{m}.json"
            if p.exists():
                cells.append(json.load(open(p)))
    return cells


def channel_metrics(yt, yp):
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ybar = float(np.mean(yt))
    denom = float(np.mean(np.abs(yt - ybar)))
    nmae = mae / denom if denom > 0 else float("nan")
    ss_tot = float(np.sum((yt - ybar) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "NMAE": nmae, "R2": r2}


def parity_slope(yt, yp):
    x = yt - yt.mean()
    y = yp - yp.mean()
    d = float(np.sum(x ** 2))
    return float(np.sum(x * y) / d) if d > 0 else float("nan")


def bar_plot(per_cell_df, metric, ylabel, path, ymean_hline=None):
    x = np.arange(len(CHANNELS_BAR))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, _, color) in enumerate(MODELS):
        means, stds = [], []
        for ch in CHANNELS_BAR:
            sub = per_cell_df[(per_cell_df.model == name) & (per_cell_df.channel == ch)]
            means.append(sub[metric].mean() if len(sub) else np.nan)
            stds.append(sub[metric].std() if len(sub) else 0)
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=name,
               color=color, capsize=3, edgecolor="white", linewidth=0.4)
    if ymean_hline is not None:
        ax.axhline(ymean_hline, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS_BAR)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[fig] {path}")


def parity_grid(all_cells, path):
    fig, axes = plt.subplots(len(MODELS), len(CHANNELS_BAR),
                             figsize=(3.4 * len(CHANNELS_BAR), 3.2 * len(MODELS)))
    for r, (name, _, color) in enumerate(MODELS):
        cells = all_cells[name]
        m0 = [c for c in cells if c["member"] == 0] or cells
        yt_all = np.concatenate([np.array(c["y_true"]) for c in m0])
        yp_all = np.concatenate([np.array(c["y_pred"]) for c in m0])
        for i, ch in enumerate(CHANNELS_BAR):
            ax = axes[r, i]
            if ch == "barrier":
                yt = yt_all.sum(1); yp = yp_all.sum(1)
            else:
                yt = yt_all[:, CHANNELS.index(ch)]
                yp = yp_all[:, CHANNELS.index(ch)]
            m = channel_metrics(yt, yp)
            slope = parity_slope(yt, yp)
            ax.scatter(yt, yp, s=6, c=color, alpha=0.55, edgecolor="none")
            lo = float(min(yt.min(), yp.min()))
            hi = float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=0.6)
            xx = np.array([lo, hi])
            b0 = float(yp.mean() - slope * yt.mean())
            ax.plot(xx, slope * xx + b0, "-", color="orange", lw=1.1)
            ax.text(0.03, 0.97,
                    f"MAE={m['MAE']:.2f}\nNMAE={m['NMAE']:.2f}\n"
                    f"R^2={m['R2']:.2f}\nslope={slope:.2f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(ch, fontsize=10)
            if i == 0:
                ax.set_ylabel(f"{name}\ny_pred", fontsize=9)
            if r == len(MODELS) - 1:
                ax.set_xlabel("y_true (kcal/mol)", fontsize=8)
            ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def write_readme(summary, counts):
    lines = [
        f"# m1 / m2 / m3 consolidated (v9, {N}-rxn cohort)",
        "",
        "3-way comparison of delta-learners over MACE-OFF23_medium features.",
        "",
        "| model | baseline | dim | cells |",
        "|-------|----------|-----|-------|",
        f"| m1    | geom6              |  6  | {counts['m1']}/25 |",
        f"| m2    | xtb_geom6          | 21  | {counts['m2']}/25 |",
        f"| m3    | xtb_geom6_plus_v2  | 24  | {counts['m3']}/25 |",
        "",
        "## Aggregate NMAE (mean +/- std across 25 cells)",
        "",
        "| channel | m1 | m2 | m3 |",
        "|---------|----|----|----|",
    ]
    for ch in CHANNELS_BAR:
        row = [f"| {ch} |"]
        for name, _, _ in MODELS:
            r = summary.loc[(name, ch)]
            row.append(f" {r['NMAE']['mean']:.3f} +/- {r['NMAE']['std']:.3f} |")
        lines.append("".join(row))
    lines += [
        "",
        "## Aggregate RMSE (kcal/mol, mean +/- std)",
        "",
        "| channel | m1 | m2 | m3 |",
        "|---------|----|----|----|",
    ]
    for ch in CHANNELS_BAR:
        row = [f"| {ch} |"]
        for name, _, _ in MODELS:
            r = summary.loc[(name, ch)]
            row.append(f" {r['RMSE']['mean']:.2f} +/- {r['RMSE']['std']:.2f} |")
        lines.append("".join(row))
    lines += [
        "",
        "## Figures",
        "",
        "- `figures/nmae_bar.png` - per-channel NMAE +/- std, m1/m2/m3 side by side",
        "- `figures/rmse_bar.png` - per-channel RMSE +/- std (kcal/mol)",
        "- `figures/parity_grid.png` - 3 rows (m1/m2/m3) x 6 cols (channels), pooled member-0 across folds",
        "",
        "## Regen",
        "",
        "`scripts/v9_ml/aggregate_m123_v9.py` (via `aggregate_m123_v9.sh`, idempotent).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[md]  {OUT / 'README.md'}")


def main():
    all_cells = {}
    counts = {}
    per_cell = []
    for name, root, _ in MODELS:
        cells = load_cells(root)
        counts[name] = len(cells)
        all_cells[name] = cells
        print(f"[{name}] {len(cells)}/25 cells at {root}")
        for cell in cells:
            yt = np.array(cell["y_true"], dtype=float)
            yp = np.array(cell["y_pred"], dtype=float)
            for i, ch in enumerate(CHANNELS):
                m = channel_metrics(yt[:, i], yp[:, i])
                per_cell.append({"model": name, "fold": cell["fold"],
                                 "member": cell["member"], "channel": ch, **m})
            m = channel_metrics(yt.sum(1), yp.sum(1))
            per_cell.append({"model": name, "fold": cell["fold"],
                             "member": cell["member"], "channel": "barrier", **m})

    df = pd.DataFrame(per_cell)
    df.to_csv(RES / "per_cell_metrics.csv", index=False)
    print(f"[csv] {RES / 'per_cell_metrics.csv'}  ({len(df)} rows)")

    summary = (df.groupby(["model", "channel"])
               [["MAE", "RMSE", "NMAE", "R2"]]
               .agg(["mean", "std"]).round(4))
    # reorder for readability
    summary = summary.reindex(
        pd.MultiIndex.from_product([[n for n, _, _ in MODELS], CHANNELS_BAR],
                                   names=["model", "channel"]))
    summary.to_csv(RES / "summary_per_model.csv")
    print(f"[csv] {RES / 'summary_per_model.csv'}")

    bar_plot(df, "NMAE", "NMAE = MAE / MAD(y_true)",
             FIG / "nmae_bar.png", ymean_hline=1.0)
    bar_plot(df, "RMSE", "RMSE (kcal/mol)", FIG / "rmse_bar.png")
    parity_grid(all_cells, FIG / "parity_grid.png")
    write_readme(summary, counts)

    print()
    print(f"=== m1/m2/m3 v9 (n={N}) summary (mean +/- std across cells) ===")
    for name, _, _ in MODELS:
        print(f"[{name}]  (cells={counts[name]}/25)")
        for ch in CHANNELS_BAR:
            r = summary.loc[(name, ch)]
            print(f"  {ch:>8s}  MAE={r['MAE']['mean']:5.2f}+/-{r['MAE']['std']:4.2f}  "
                  f"RMSE={r['RMSE']['mean']:5.2f}+/-{r['RMSE']['std']:4.2f}  "
                  f"NMAE={r['NMAE']['mean']:.3f}+/-{r['NMAE']['std']:.3f}  "
                  f"R2={r['R2']['mean']:.3f}+/-{r['R2']['std']:.3f}")


if __name__ == "__main__":
    main()
