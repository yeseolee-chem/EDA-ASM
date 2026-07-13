"""Regenerate m3-only NMAE/RMSE/MAE bar + parity grid + README for the v9
783-rxn cohort.

Reads:
  m3/code/trackB_lowlr_v9_xtb_geom6_plus_v2/m1_delta/fold{0..4}/member{0..4}.json

Writes:
  m3/figures/nmae_bar.png
  m3/figures/rmse_bar.png
  m3/figures/mae_bar.png
  m3/figures/parity_grid.png
  m3/results/per_cell_metrics.csv
  m3/results/summary_per_channel.csv
  m3/README.md  (regenerated with v9 headline numbers)
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
CELL_ROOT = REPO / "m3/code/trackB_lowlr_v9_xtb_geom6_plus_v2/m1_delta"
FIG = REPO / "m3/figures"
RES = REPO / "m3/results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "Velst", "oi", "disp"]
CHANNELS_BAR = CHANNELS + ["barrier"]
COLOR = "#c26f6b"
N = 783


def load_cells():
    cells = []
    for f in range(5):
        for m in range(5):
            p = CELL_ROOT / f"fold{f}" / f"member{m}.json"
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
    denom = float(np.sum(x ** 2))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def bar_plot(per_cell_df, metric, ylabel, title, path):
    x = np.arange(len(CHANNELS_BAR))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    means, stds = [], []
    for ch in CHANNELS_BAR:
        sub = per_cell_df[per_cell_df.channel == ch]
        means.append(sub[metric].mean() if len(sub) else np.nan)
        stds.append(sub[metric].std() if len(sub) else 0)
    ax.bar(x, means, 0.6, yerr=stds, color=COLOR, capsize=4,
           edgecolor="black", linewidth=0.6)
    if metric == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
        ax.legend(loc="upper right")
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS_BAR)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[fig] {path}")


def parity_grid(cells, path):
    fig, axes = plt.subplots(1, len(CHANNELS_BAR),
                             figsize=(3.6 * len(CHANNELS_BAR), 3.6))
    m0 = [c for c in cells if c["member"] == 0]
    if not m0:
        m0 = cells
    yt_all = np.concatenate([np.array(c["y_true"]) for c in m0])
    yp_all = np.concatenate([np.array(c["y_pred"]) for c in m0])
    for i, ch in enumerate(CHANNELS_BAR):
        ax = axes[i]
        if ch == "barrier":
            yt = yt_all.sum(axis=1)
            yp = yp_all.sum(axis=1)
        else:
            yt = yt_all[:, CHANNELS.index(ch)]
            yp = yp_all[:, CHANNELS.index(ch)]
        m = channel_metrics(yt, yp)
        slope = parity_slope(yt, yp)
        ax.scatter(yt, yp, s=6, c=COLOR, alpha=0.55, edgecolor="none")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=0.6)
        ols_x = np.array([lo, hi])
        b0 = float(yp.mean() - slope * yt.mean())
        ax.plot(ols_x, slope * ols_x + b0, "-", color="orange", lw=1.2)
        ax.text(0.03, 0.97,
                f"MAE={m['MAE']:.2f}\nNMAE={m['NMAE']:.2f}\n"
                f"R^2={m['R2_det']:.2f}\nslope={slope:.2f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
        ax.set_title(ch, fontsize=10)
        if i == 0:
            ax.set_ylabel("y_pred", fontsize=9)
        ax.set_xlabel("y_true (kcal/mol)", fontsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle(f"m3 (v9, n={N}) - parity (pooled member-0 across 5 folds)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def write_readme(summary, n_cells):
    ch_lines = []
    for ch in CHANNELS_BAR:
        sub = summary.loc[ch]
        ch_lines.append(
            f"| {ch:<8s} | "
            f"{sub['MAE']['mean']:.2f} +/- {sub['MAE']['std']:.2f} | "
            f"{sub['RMSE']['mean']:.2f} +/- {sub['RMSE']['std']:.2f} | "
            f"{sub['NMAE']['mean']:.3f} +/- {sub['NMAE']['std']:.3f} | "
            f"{sub['R2_det']['mean']:.3f} +/- {sub['R2_det']['std']:.3f} |"
        )
    body = f"""# m3 (v9, 783-rxn cohort)

Delta-learner over MACE-OFF23_medium features with a 24-d physics baseline
(d1..d21 xTB/geom + d22 = mu^2 / 2 eta, d23 = sum q^2, d24 = sum |WBO_AB|).

## Bundle + splits

- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt`
- Splits: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9/fold{{0..4}}/`
- Labels: `outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet`
- Cells:  `m3/code/trackB_lowlr_v9_xtb_geom6_plus_v2/m1_delta/fold{{0..4}}/member{{0..4}}.json`
- Fold x member = 5 x 5 = 25 cells ({n_cells}/25 available).

## Aggregate metrics (mean +/- std across cells)

| channel  | MAE (kcal/mol) | RMSE (kcal/mol) | NMAE | R^2 |
|----------|----------------|-----------------|------|-----|
{chr(10).join(ch_lines)}

## Figures

- `figures/nmae_bar.png`  — per-channel NMAE +/- std
- `figures/rmse_bar.png`  — per-channel RMSE +/- std (kcal/mol)
- `figures/mae_bar.png`   — per-channel MAE +/- std (kcal/mol)
- `figures/parity_grid.png` — parity (pooled member-0 predictions across 5 folds)

## Regen script

`scripts/v9_ml/regen_m3_figures_v9.py` (submitted via `regen_m3_figures_v9.sh`).
Idempotent: overwrites existing outputs.
"""
    (REPO / "m3/README.md").write_text(body)
    print(f"[md]  {REPO / 'm3/README.md'}")


def main():
    cells = load_cells()
    print(f"[stage6] loaded {len(cells)}/25 cells")
    if not cells:
        raise SystemExit("no cells found — check that training completed")

    per_cell = []
    for cell in cells:
        y_true = np.array(cell["y_true"], dtype=float)
        y_pred = np.array(cell["y_pred"], dtype=float)
        for i, ch in enumerate(CHANNELS):
            m = channel_metrics(y_true[:, i], y_pred[:, i])
            per_cell.append({"fold": cell["fold"], "member": cell["member"],
                             "channel": ch, **m})
        m = channel_metrics(y_true.sum(1), y_pred.sum(1))
        per_cell.append({"fold": cell["fold"], "member": cell["member"],
                         "channel": "barrier", **m})
    per_cell_df = pd.DataFrame(per_cell)
    per_cell_df.to_csv(RES / "per_cell_metrics.csv", index=False)
    print(f"[csv] {RES / 'per_cell_metrics.csv'}  ({len(per_cell_df)} rows)")

    summary = (per_cell_df.groupby("channel")
               [["MAE", "RMSE", "NMAE", "R2_det"]]
               .agg(["mean", "std"]).round(4))
    summary = summary.reindex(CHANNELS_BAR)
    summary.to_csv(RES / "summary_per_channel.csv")
    print(f"[csv] {RES / 'summary_per_channel.csv'}")

    bar_plot(per_cell_df, "NMAE", "NMAE = MAE / MAD(y_true)",
             f"m3 (v9, n={N}) - NMAE by channel", FIG / "nmae_bar.png")
    bar_plot(per_cell_df, "RMSE", "RMSE (kcal/mol)",
             f"m3 (v9, n={N}) - RMSE by channel", FIG / "rmse_bar.png")
    bar_plot(per_cell_df, "MAE", "MAE (kcal/mol)",
             f"m3 (v9, n={N}) - MAE by channel", FIG / "mae_bar.png")
    parity_grid(cells, FIG / "parity_grid.png")
    write_readme(summary, len(cells))

    print()
    print(f"=== m3 (v9, n={N}) summary (mean +/- std, {len(cells)}/25 cells) ===")
    for ch in CHANNELS_BAR:
        r = summary.loc[ch]
        print(f"  {ch:>8s}  MAE={r['MAE']['mean']:5.2f}+/-{r['MAE']['std']:4.2f}  "
              f"RMSE={r['RMSE']['mean']:5.2f}+/-{r['RMSE']['std']:4.2f}  "
              f"NMAE={r['NMAE']['mean']:.3f}+/-{r['NMAE']['std']:.3f}  "
              f"R2={r['R2_det']['mean']:.3f}+/-{r['R2_det']['std']:.3f}")


if __name__ == "__main__":
    main()
