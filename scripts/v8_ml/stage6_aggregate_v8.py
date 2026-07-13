"""Stage 6 (v8) — aggregate m1/m2/m3 fold×member outputs for the 799-rxn v8
cohort and reproduce the NMAE / RMSE / parity figures.

Reads (per cell):
  m1/code/trackB_lowlr_v8_geom6/m1_delta/fold{0..4}/member{0..4}.json
  m2/code/trackB_lowlr_v8_xtb_geom6/m1_delta/fold{0..4}/member{0..4}.json
  m3/code/trackB_lowlr_v8_xtb_geom6_plus_v2/m1_delta/fold{0..4}/member{0..4}.json

Writes:
  outputs/v8_review/results/nmae_v8.png
  outputs/v8_review/results/rmse_v8.png
  outputs/v8_review/results/parity_v8.png
  outputs/v8_review/results/per_cell_v8.csv
  outputs/v8_review/results/summary_v8.csv
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
OUT = REPO / "outputs" / "v8_review" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# (model_name, cell dir under m{k}/code/, bar color).
# Colors from the proposal PDF (m1 dark blue, m2 teal, m3 salmon).
MODELS = [
    ("m1", "trackB_lowlr_v8_geom6",              "#1f2b6b"),
    ("m2", "trackB_lowlr_v8_xtb_geom6",          "#227b8f"),
    ("m3", "trackB_lowlr_v8_xtb_geom6_plus_v2",  "#c26f6b"),
]
# Runner emits y_true[:, i] in the ASR_COMPONENTS order that stage4 wrote
# into the bundle: strain, Pauli, Velst, oi, disp.
CHANNELS = ["strain", "Pauli", "Velst", "oi", "disp"]
N_CH = len(CHANNELS)
CHANNELS_BAR = CHANNELS + ["barrier"]


def load_cells(model_name: str, tag: str) -> list[dict]:
    root = REPO / model_name / "code" / tag / "m1_delta"
    cells = []
    for f in range(5):
        for m in range(5):
            p = root / f"fold{f}" / f"member{m}.json"
            if p.exists():
                cells.append(json.load(open(p)))
    return cells


def channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def parity_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    denom = float(np.sum(x ** 2))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def main():
    # -------------------------------------------------------------------
    # 1) Per-cell metrics dataframe (5 channels + barrier)
    # -------------------------------------------------------------------
    per_cell = []
    for name, tag, _ in MODELS:
        cells = load_cells(name, tag)
        print(f"[{name}] {len(cells)}/25 cells at {tag}")
        for cell in cells:
            y_true = np.array(cell["y_true"], dtype=float)
            y_pred = np.array(cell["y_pred"], dtype=float)
            for i, ch in enumerate(CHANNELS):
                m = channel_metrics(y_true[:, i], y_pred[:, i])
                per_cell.append({"model": name, "fold": cell["fold"],
                                 "member": cell["member"], "channel": ch, **m})
            barrier_true = y_true.sum(axis=1)
            barrier_pred = y_pred.sum(axis=1)
            m = channel_metrics(barrier_true, barrier_pred)
            per_cell.append({"model": name, "fold": cell["fold"],
                             "member": cell["member"], "channel": "barrier", **m})

    per_cell_df = pd.DataFrame(per_cell)
    per_cell_df.to_csv(OUT / "per_cell_v8.csv", index=False)
    print(f"[csv] wrote {OUT / 'per_cell_v8.csv'}  ({len(per_cell_df)} rows)")

    summary = (per_cell_df.groupby(["model", "channel"])
               [["MAE", "RMSE", "NMAE", "R2_det"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(OUT / "summary_v8.csv")
    print(f"[csv] wrote {OUT / 'summary_v8.csv'}")

    if per_cell_df.empty:
        print("[warn] no cells found — skipping plots. Check that stage5 has produced JSONs.")
        return

    # -------------------------------------------------------------------
    # 2) NMAE bar (6 channels: strain, Pauli, elst, oi, disp, barrier)
    # -------------------------------------------------------------------
    x = np.arange(len(CHANNELS_BAR))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, _, color) in enumerate(MODELS):
        means, stds = [], []
        for ch in CHANNELS_BAR:
            sub = per_cell_df[(per_cell_df.model == name) &
                              (per_cell_df.channel == ch)]
            means.append(sub.NMAE.mean() if len(sub) else np.nan)
            stds.append(sub.NMAE.std() if len(sub) else 0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=name, color=color, capsize=3,
               edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("NMAE = MAE / MAD(y_true)")
    ax.set_title(f"v8 cohort (n=799) — NMAE by channel")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS_BAR)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "nmae_v8.png", dpi=160)
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'nmae_v8.png'}")

    # -------------------------------------------------------------------
    # 3) RMSE bar (same 6 groups, kcal/mol)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, _, color) in enumerate(MODELS):
        means, stds = [], []
        for ch in CHANNELS_BAR:
            sub = per_cell_df[(per_cell_df.model == name) &
                              (per_cell_df.channel == ch)]
            means.append(sub.RMSE.mean() if len(sub) else np.nan)
            stds.append(sub.RMSE.std() if len(sub) else 0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=name, color=color, capsize=3,
               edgecolor="white", linewidth=0.4)
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.5)
    ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_title(f"v8 cohort (n=799) — RMSE by channel")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS_BAR)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "rmse_v8.png", dpi=160)
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'rmse_v8.png'}")

    # -------------------------------------------------------------------
    # 4) Parity grid (3 rows × 6 cols). Pool member-0 predictions across folds.
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(3, len(CHANNELS_BAR),
                             figsize=(3.6 * len(CHANNELS_BAR), 10))
    for r, (name, tag, color) in enumerate(MODELS):
        cells = load_cells(name, tag)
        m0_cells = [c for c in cells if c["member"] == 0]
        if not m0_cells:
            # Fall back to all cells if member 0 is missing for this model.
            m0_cells = cells
        if not m0_cells:
            for i, ch in enumerate(CHANNELS_BAR):
                axes[r, i].text(0.5, 0.5, "no cells",
                                ha="center", va="center",
                                transform=axes[r, i].transAxes)
            continue
        pooled_yt = np.concatenate([np.array(c["y_true"]) for c in m0_cells])
        pooled_yp = np.concatenate([np.array(c["y_pred"]) for c in m0_cells])
        for i, ch in enumerate(CHANNELS_BAR):
            ax = axes[r, i]
            if ch == "barrier":
                yt = pooled_yt.sum(axis=1)
                yp = pooled_yp.sum(axis=1)
            else:
                yt = pooled_yt[:, CHANNELS.index(ch)]
                yp = pooled_yp[:, CHANNELS.index(ch)]
            m = channel_metrics(yt, yp)
            slope = parity_slope(yt, yp)
            ax.scatter(yt, yp, s=6, c=color, alpha=0.55, edgecolor="none")
            lo = float(min(yt.min(), yp.min()))
            hi = float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=0.6)
            ols_x = np.array([lo, hi])
            b0 = float(yp.mean() - slope * yt.mean())
            ax.plot(ols_x, slope * ols_x + b0, "-", color="orange", lw=1.2)
            ax.text(0.03, 0.97,
                    f"MAE={m['MAE']:.2f}\nNMAE={m['NMAE']:.2f}\n"
                    f"R²={m['R2_det']:.2f}\nslope={slope:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            if r == 0:
                ax.set_title(ch, fontsize=10)
            if i == 0:
                ax.set_ylabel(f"{name}\ny_pred", fontsize=9)
            if r == 2:
                ax.set_xlabel("y_true (kcal/mol)", fontsize=8)
    fig.suptitle("v8 cohort (n=799) — parity grid (pooled member-0 across folds)",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "parity_v8.png", dpi=160)
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'parity_v8.png'}")

    # -------------------------------------------------------------------
    # 5) Console summary
    # -------------------------------------------------------------------
    print()
    print("=== summary (mean ± std across cells) ===")
    for name, _, _ in MODELS:
        print(f"[{name}]")
        for ch in CHANNELS_BAR:
            sub = per_cell_df[(per_cell_df.model == name) &
                              (per_cell_df.channel == ch)]
            if not len(sub):
                continue
            print(f"  {ch:>8s}  MAE={sub.MAE.mean():5.2f}±{sub.MAE.std():4.2f}  "
                  f"RMSE={sub.RMSE.mean():5.2f}±{sub.RMSE.std():4.2f}  "
                  f"NMAE={sub.NMAE.mean():.3f}±{sub.NMAE.std():.3f}  "
                  f"R²={sub.R2_det.mean():.3f}±{sub.R2_det.std():.3f}")
    print()
    print(f"[stage6] done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
