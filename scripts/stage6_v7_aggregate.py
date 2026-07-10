"""v7 aggregate: reads m1/m2/m3 v7 fold*/member*.json and emits comparison
figures + REPORT for the 776-reaction cohort.

v7 output paths (all three use OUT_TAG=v7_lowlr_no_ood, BASELINE=geom6):
  m1/code/trackB_v7_lowlr_no_ood_geom6/m1_delta/fold{0..4}/member{0..4}.json
  m2/code/trackB_v7_lowlr_no_ood_geom6/m1_delta/fold{0..4}/member{0..4}.json
  m3/code/trackB_v7_lowlr_no_ood_geom6/m1_delta/fold{0..4}/member{0..4}.json

Writes:
  comparison/v7/
      results/summary_per_model.csv
      results/per_cell_metrics.csv
      figures/nmae_bar.png
      figures/rmse_bar.png
      figures/parity_grid.png
      REPORT.md
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
OUT = REPO / "comparison" / "v7"
(OUT / "results").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

TRAIN_TAG = "trackB_v7_lowlr_no_ood_geom6"
MODELS = [
    ("m1", "#1E2761"),
    ("m2", "#1C7293"),
    ("m3", "#C45A4D"),
]
CHANNELS = ["strain", "Pauli", "Velst", "oi", "disp"]


def load_cells(model_name: str) -> list[dict]:
    root = REPO / model_name / "code" / TRAIN_TAG / "m1_delta"
    cells = []
    for f in range(5):
        for m in range(5):
            p = root / f"fold{f}" / f"member{m}.json"
            if p.exists():
                cells.append(json.load(open(p)))
    return cells


def channel_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ybar = float(np.mean(y_true))
    denom = float(np.mean(np.abs(y_true - ybar)))
    nmae = mae / denom if denom > 0 else float("nan")
    ss_tot = float(np.sum((y_true - ybar) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "NMAE": nmae, "R2_det": r2}


def parity_slope(y_true, y_pred):
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    d = float(np.sum(x ** 2))
    return float(np.sum(x * y) / d) if d > 0 else float("nan")


def main():
    per_cell = []
    for name, _ in MODELS:
        cells = load_cells(name)
        print(f"{name}: {len(cells)} cells found")
        for cell in cells:
            y_true = np.array(cell["y_true"], dtype=float)
            y_pred = np.array(cell["y_pred"], dtype=float)
            for i, ch in enumerate(CHANNELS):
                m = channel_metrics(y_true[:, i], y_pred[:, i])
                per_cell.append({"model": name, "fold": cell["fold"],
                                 "member": cell["member"], "channel": ch, **m})
            bt = y_true.sum(axis=1); bp = y_pred.sum(axis=1)
            m = channel_metrics(bt, bp)
            per_cell.append({"model": name, "fold": cell["fold"],
                             "member": cell["member"], "channel": "barrier", **m})

    per_cell_df = pd.DataFrame(per_cell)
    per_cell_df.to_csv(OUT / "results" / "per_cell_metrics.csv", index=False)

    summary = (per_cell_df.groupby(["model", "channel"])
               [["MAE", "RMSE", "NMAE", "R2_det"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(OUT / "results" / "summary_per_model.csv")

    channels_bar = CHANNELS + ["barrier"]
    x = np.arange(len(channels_bar))
    width = 0.27

    # NMAE bar
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, color) in enumerate(MODELS):
        means, stds = [], []
        for ch in channels_bar:
            sub = per_cell_df[(per_cell_df.model == name) & (per_cell_df.channel == ch)]
            means.append(sub.NMAE.mean() if len(sub) else np.nan)
            stds.append(sub.NMAE.std() if len(sub) else 0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=name, color=color, capsize=3, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("NMAE = MAE / MAD(y_true)")
    ax.set_xticks(x); ax.set_xticklabels(channels_bar)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y"); fig.tight_layout()
    fig.savefig(OUT / "figures" / "nmae_bar.png", dpi=160); plt.close(fig)

    # RMSE bar
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, color) in enumerate(MODELS):
        means, stds = [], []
        for ch in channels_bar:
            sub = per_cell_df[(per_cell_df.model == name) & (per_cell_df.channel == ch)]
            means.append(sub.RMSE.mean() if len(sub) else np.nan)
            stds.append(sub.RMSE.std() if len(sub) else 0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=name, color=color, capsize=3, edgecolor="white", linewidth=0.4)
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_xticks(x); ax.set_xticklabels(channels_bar)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y"); fig.tight_layout()
    fig.savefig(OUT / "figures" / "rmse_bar.png", dpi=160); plt.close(fig)

    # Parity grid
    fig, axes = plt.subplots(3, len(channels_bar),
                             figsize=(3.6 * len(channels_bar), 10))
    for r, (name, color) in enumerate(MODELS):
        cells = load_cells(name)
        if not cells:
            continue
        pooled_yt = np.concatenate(
            [np.array(c["y_true"]) for c in cells if c["member"] == 0])
        pooled_yp = np.concatenate(
            [np.array(c["y_pred"]) for c in cells if c["member"] == 0])
        for i, ch in enumerate(channels_bar):
            ax = axes[r, i]
            if ch == "barrier":
                yt = pooled_yt.sum(axis=1); yp = pooled_yp.sum(axis=1)
            else:
                yt = pooled_yt[:, CHANNELS.index(ch)]
                yp = pooled_yp[:, CHANNELS.index(ch)]
            m = channel_metrics(yt, yp); slope = parity_slope(yt, yp)
            ax.scatter(yt, yp, s=6, c=color, alpha=0.55, edgecolor="none")
            lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=0.6)
            b0 = float(yp.mean() - slope * yt.mean())
            ax.plot([lo, hi], [slope * lo + b0, slope * hi + b0], "-", color="orange", lw=1.2)
            ax.text(0.03, 0.97,
                    f"MAE={m['MAE']:.2f}\nNMAE={m['NMAE']:.2f}\nR2={m['R2_det']:.2f}\nslope={slope:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            if r == 0: ax.set_title(ch, fontsize=10)
            if i == 0: ax.set_ylabel(f"{name}\ny_pred", fontsize=9)
            if r == 2: ax.set_xlabel("y_true (kcal/mol)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "parity_grid.png", dpi=160); plt.close(fig)

    # REPORT.md
    lines = ["# v7 m1/m2/m3 comparison (776-reaction cohort, 5-channel ORCA labels)",
             "", "5-fold x 5-member CV, spec-compliant training.",
             "", "## Cell counts", ""]
    for name, _ in MODELS:
        n = len(load_cells(name))
        lines.append(f"- {name}: {n}/25 cells")
    lines += ["", "## Mean +/- std across all cells", ""]
    for name, _ in MODELS:
        lines += [f"### {name}", "",
                  "| channel | MAE | RMSE | NMAE | R2 |",
                  "|---|---|---|---|---|"]
        for ch in channels_bar:
            sub = per_cell_df[(per_cell_df.model == name) & (per_cell_df.channel == ch)]
            if not len(sub): continue
            lines.append(
                f"| {ch} | {sub.MAE.mean():.2f} +/- {sub.MAE.std():.2f} "
                f"| {sub.RMSE.mean():.2f} +/- {sub.RMSE.std():.2f} "
                f"| {sub.NMAE.mean():.3f} +/- {sub.NMAE.std():.3f} "
                f"| {sub.R2_det.mean():.3f} +/- {sub.R2_det.std():.3f} |")
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
