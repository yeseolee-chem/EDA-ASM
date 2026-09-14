#!/usr/bin/env python3
"""plot_results.py — scatter (per-model) + MAE bar (combined).

Only ESPLEY54 feature set (E5 removed). 8 b^ch channels shown.

Outputs (in <ROOT>/figures/):
    scatter_<model>_ESPLEY54.png   × 4 models (Ridge/KRR/SVR/XGB)
    mae_bar_espley54.png           — grouped bars: 8 channels × 4 models
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    ("dft_d1_kcal",   "strain_1 (dipole)"),
    ("dft_d2_kcal",   "strain_2 (dipolarophile)"),
    ("dft_elst_dft",  "elst"),
    ("dft_pauli_dft", "Pauli"),
    ("dft_oi_dft",    "OI"),
    ("dft_disp_dft",  "disp"),
    ("dft_cpcm_dft",  "CPCM"),
    ("dft_cds_dft",   "CDS"),
]
MODELS = ["Ridge", "KRR_rbf", "SVR_rbf", "XGB"]
MODEL_COLORS = {"Ridge": "#4C72B0", "KRR_rbf": "#DD8452",
                "SVR_rbf": "#55A868", "XGB": "#C44E52"}
N_FEATS = 54   # ESPLEY54 only


def scatter_one(preds, model):
    sub = preds[(preds["model"] == model) & (preds["feature_set"] == N_FEATS)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for k, (col, label) in enumerate(CHANNELS):
        ax = axes.flat[k]
        s = sub[sub["target"] == col]
        if len(s) == 0:
            ax.set_axis_off()
            ax.set_title(f"{label}\n(no preds)")
            continue
        y = s["y"].values
        yp = s["yhat"].values
        mae = float(np.mean(np.abs(y - yp)))
        rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
        r2 = 1 - float(np.sum((y - yp) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
        r = float(np.corrcoef(y, yp)[0, 1]) if (np.std(y) > 0 and np.std(yp) > 0) else float("nan")
        ax.scatter(y, yp, s=6, alpha=0.35, color=MODEL_COLORS[model])
        lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{label}\nMAE {mae:.2f}  RMSE {rmse:.2f}  r² {r2:+.3f}  r {r:+.3f}", fontsize=9)
        ax.set_xlabel("actual (kcal/mol)"); ax.set_ylabel("predicted")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{model} · ESPLEY54  —  5-seed test-fold predictions", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / f"scatter_{model}_ESPLEY54.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def mae_bar(report):
    """Grouped bar: 8 channels (x) × 4 models (bars). Uses Protocol A test_mae from ml_report.json."""
    per_t = report["per_target"]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(CHANNELS))
    w = 0.20
    for i_m, m in enumerate(MODELS):
        maes = []
        sds = []
        for col, _ in CHANNELS:
            A = per_t.get(col, {}).get("protocol_A", {}).get("ESPLEY54", {})
            r = A.get(m)
            if r is None:
                maes.append(np.nan); sds.append(0)
            else:
                maes.append(r["test_mae"]); sds.append(r["test_mae_sd"])
        offset = (i_m - 1.5) * w
        bars = ax.bar(x + offset, maes, w, yerr=sds, capsize=2,
                      color=MODEL_COLORS[m], edgecolor="black", linewidth=0.4, label=m)
        # value labels
        for j, (v, sd) in enumerate(zip(maes, sds)):
            if np.isfinite(v):
                ax.text(x[j] + offset, v + sd + 0.05, f"{v:.2f}",
                        ha="center", fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CHANNELS], rotation=25, ha="right")
    ax.set_ylabel("test MAE (kcal/mol, mean ± sd over 5 seeds)")
    ax.set_title("ESPLEY54  ·  Per-channel test MAE  ·  4 models")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    out = FIG_DIR / "mae_bar_espley54.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    # remove all existing PNGs
    for p in FIG_DIR.glob("*.png"):
        p.unlink()
        print(f"  removed {p.name}")

    preds = pd.read_parquet(ROOT / "predictions.parquet")
    report = json.loads((ROOT / "ml_report.json").read_text())
    print(f"loaded predictions ({len(preds)} rows) + report ({len(report['per_target'])} targets)")

    for m in MODELS:
        out = scatter_one(preds, m)
        print(f"  wrote {out.name}")
    out = mae_bar(report)
    print(f"  wrote {out.name}")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
