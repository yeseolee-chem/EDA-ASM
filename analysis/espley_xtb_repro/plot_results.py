#!/usr/bin/env python3
"""plot_results.py — scatter (per-model) + MAE bar (combined).

One feature set (ESPLEY_PLOT_FS, default 73 = ESPLEY73), the 9 reported targets:
barrier, d1, d2 and the 6 EDA channels (e_bond, eint_spe and c_ghost are trained
but not plotted: derived quantities / method artefact).

Outputs (in <ROOT>/figures/; existing PNGs there are removed first):
    scatter_<model>_ESPLEY<fs>.png   × 4 models (Ridge/KRR/SVR/XGB)
    mae_bar_espley<fs>.png           — grouped bars: 9 targets × 4 models
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

# Reported targets: the barrier and its 8 channels. e_bond (= sum of the 6 EDA
# channels) and eint_spe are derived quantities, and c_ghost is a method
# artefact (BSSE + cavity) reported once in the SI, so none of them is plotted.
CHANNELS = [
    ("dft_barrier_kcal",  "barrier"),
    ("dft_d1_kcal",       "strain_1 (dipole)"),
    ("dft_d2_kcal",       "strain_2 (dipolarophile)"),
    ("dft_elst_dft",      "elst"),
    ("dft_pauli_dft",     "Pauli"),
    ("dft_oi_dft",        "OI"),
    ("dft_disp_dft",      "disp"),
    ("dft_cpcm_dft",      "CPCM"),
    ("dft_cds_dft",       "CDS"),
]
MODELS = ["Ridge", "KRR_rbf", "SVR_rbf", "XGB"]
MODEL_COLORS = {"Ridge": "#4C72B0", "KRR_rbf": "#DD8452",
                "SVR_rbf": "#55A868", "XGB": "#C44E52"}
N_FEATS = int(os.environ.get("ESPLEY_PLOT_FS", "73"))
FS = f"ESPLEY{N_FEATS}"
LABELS_NOTE = "labels: SMD(water) relabel, method ① (d1/d2 from own-basis fragments)"


def scatter_one(preds, model):
    sub = preds[(preds["model"] == model) & (preds["feature_set"] == N_FEATS)]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
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
    fig.suptitle(f"{model} · {FS}  —  5-seed test-fold predictions\n{LABELS_NOTE}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = FIG_DIR / f"scatter_{model}_{FS}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def mae_bar(report):
    """Grouped bar: 9 targets (x) × 4 models (bars). Uses Protocol A test_mae from ml_report.json."""
    per_t = report["per_target"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(CHANNELS))
    w = 0.20
    for i_m, m in enumerate(MODELS):
        maes = []
        sds = []
        for col, _ in CHANNELS:
            A = per_t.get(col, {}).get("protocol_A", {}).get(FS, {})
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
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)          # headroom for the rotated value labels
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CHANNELS], rotation=25, ha="right")
    ax.set_ylabel("test MAE (kcal/mol, mean ± sd over 5 seeds)")
    ax.set_title(f"{FS}  ·  Per-target test MAE (Protocol A)  ·  4 models\n{LABELS_NOTE}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    out = FIG_DIR / f"mae_bar_{FS.lower()}.png"
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
