#!/usr/bin/env python3
"""plot_results.py — MAE bar charts + actual-vs-predicted scatter for every channel.

Inputs:
    ml_report.json           (per-target metrics from Protocol A/B)
    predictions.parquet      (5-seed test-fold predictions from Protocol A)

Outputs (in <ROOT>/figures/):
    mae_per_target.png             — grouped bars, all 11 targets × (models × feature sets)
    scatter_<target>.png           — 8-panel scatter per target (Ridge/KRR/SVR/XGB × E5/ESPLEY54)
    scatter_grid_espley54.png      — 11 targets × 4 models grid using ESPLEY54 predictions only
    pre_vs_post_mae.png            — pre-ML MAE vs best-model test MAE per target
    mae_pct_range.png              — MAE as % of target range (Espley Table style)
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

TARGET_LABELS = {
    "dft_barrier_kcal":  "ΔE‡ barrier",
    "dft_d1_kcal":       "d1 (dipole strain)",
    "dft_d2_kcal":       "d2 (dipolarophile strain)",
    "dft_eint_spe_kcal": "ΔE_int (SPE)",
    "dft_e_bond_kcal":   "Bond E (EDA)",
    "dft_elst_dft":      "elst",
    "dft_pauli_dft":     "Pauli",
    "dft_oi_dft":        "OI",
    "dft_disp_dft":      "disp",
    "dft_cpcm_dft":      "CPCM",
    "dft_cds_dft":       "CDS",
}
MODEL_ORDER = ["Ridge", "KRR_rbf", "SVR_rbf", "XGB"]
MODEL_COLORS = {"Ridge": "#4C72B0", "KRR_rbf": "#DD8452",
                "SVR_rbf": "#55A868", "XGB": "#C44E52"}
FSET_ORDER = ["E5", "ESPLEY54"]
FSET_HATCH = {"E5": "", "ESPLEY54": "//"}


def load():
    report = json.loads((ROOT / "ml_report.json").read_text())
    preds = pd.read_parquet(ROOT / "predictions.parquet")
    return report, preds


def mae_per_target_grouped(report):
    per_t = report["per_target"]
    tgts = list(per_t)
    fig, ax = plt.subplots(figsize=(16, 6.5))
    x = np.arange(len(tgts))
    bar_w = 0.10
    for i_fs, fs in enumerate(FSET_ORDER):
        for i_m, m in enumerate(MODEL_ORDER):
            maes = [per_t[t]["protocol_A"][fs][m]["test_mae"] for t in tgts]
            sds = [per_t[t]["protocol_A"][fs][m]["test_mae_sd"] for t in tgts]
            offset = (i_fs * 4 + i_m) * bar_w - bar_w * 3.5
            ax.bar(x + offset, maes, bar_w, yerr=sds, capsize=2,
                   color=MODEL_COLORS[m], hatch=FSET_HATCH[fs],
                   edgecolor="black", linewidth=0.4,
                   label=f"{m} · {fs}" if (i_m == 0 or i_fs == 0) else None)
    # pre-ML markers
    for j, t in enumerate(tgts):
        p = (per_t[t].get("pre_ml") or {}).get("mae")
        if p is not None and np.isfinite(p):
            ax.plot([x[j] - bar_w * 4.2, x[j] + bar_w * 4.2], [p, p],
                    color="black", lw=1.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in tgts], rotation=30, ha="right")
    ax.set_ylabel("test MAE (kcal/mol)  (mean ± sd over 5 seeds)")
    ax.set_title("Per-channel MAE — 4 models × 2 feature sets (Ridge/KRR/SVR/XGB × E5/ESPLEY54); dashed = pre-ML")
    ax.grid(axis="y", alpha=0.3)
    # custom legend
    hl = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODEL_ORDER]
    hl += [plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch=FSET_HATCH[fs]) for fs in FSET_ORDER]
    ax.legend(hl, MODEL_ORDER + [f"({fs})" for fs in FSET_ORDER],
              loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_per_target.png", dpi=140)
    plt.close(fig)


def scatter_per_target(preds):
    """8-panel per-target scatter: 4 models × 2 feature sets."""
    for tgt, sub in preds.groupby("target"):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
        for i_fs, fs in enumerate(FSET_ORDER):
            n_feats = {"E5": 5, "ESPLEY54": 54}[fs]
            sub_fs = sub[sub["feature_set"] == n_feats]
            for i_m, m in enumerate(MODEL_ORDER):
                ax = axes[i_fs, i_m]
                subm = sub_fs[sub_fs["model"] == m]
                if len(subm) == 0:
                    ax.set_title(f"{m} ({fs}) — no data")
                    continue
                y = subm["y"].values; yp = subm["yhat"].values
                mae = float(np.mean(np.abs(y - yp)))
                r2 = 1 - float(np.sum((y - yp) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
                ax.scatter(y, yp, s=6, alpha=0.35, color=MODEL_COLORS[m])
                lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                ax.set_title(f"{m} · {fs}\nMAE {mae:.2f}  r² {r2:+.3f}", fontsize=9)
                ax.set_xlabel("actual"); ax.set_ylabel("predicted")
                ax.tick_params(labelsize=8)
        fig.suptitle(f"{TARGET_LABELS.get(tgt, tgt)}  ({tgt})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safe = tgt.replace("/", "_")
        fig.savefig(FIG_DIR / f"scatter_{safe}.png", dpi=130)
        plt.close(fig)


def scatter_grid_espley54(preds):
    """11 targets × 4 models grid, ESPLEY54 predictions only."""
    n_feats_e54 = 54
    p54 = preds[preds["feature_set"] == n_feats_e54]
    tgts = sorted(p54["target"].unique(),
                  key=lambda t: list(TARGET_LABELS).index(t) if t in TARGET_LABELS else 999)
    fig, axes = plt.subplots(len(tgts), len(MODEL_ORDER), figsize=(12, 2.4 * len(tgts)),
                             sharex="row", sharey="row")
    for i_t, tgt in enumerate(tgts):
        sub = p54[p54["target"] == tgt]
        for i_m, m in enumerate(MODEL_ORDER):
            ax = axes[i_t, i_m]
            subm = sub[sub["model"] == m]
            if len(subm) == 0:
                ax.set_axis_off(); continue
            y = subm["y"].values; yp = subm["yhat"].values
            mae = float(np.mean(np.abs(y - yp)))
            r2 = 1 - float(np.sum((y - yp) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
            ax.scatter(y, yp, s=4, alpha=0.30, color=MODEL_COLORS[m])
            lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
            ax.set_title(f"MAE {mae:.2f}  r² {r2:+.3f}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i_m == 0:
                ax.set_ylabel(TARGET_LABELS.get(tgt, tgt), fontsize=9)
            if i_t == 0:
                ax.text(0.5, 1.20, m, transform=ax.transAxes, ha="center", fontsize=11, weight="bold")
    fig.suptitle("ESPLEY54 · actual vs predicted (5-seed test-fold pooled)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_DIR / "scatter_grid_espley54.png", dpi=130)
    plt.close(fig)


def pre_vs_post(report):
    per_t = report["per_target"]
    tgts = list(per_t)
    pre = [(per_t[t].get("pre_ml") or {}).get("mae", float("nan")) for t in tgts]
    best_e5 = [min(per_t[t]["protocol_A"]["E5"][m]["test_mae"] for m in MODEL_ORDER) for t in tgts]
    best_e54 = [min(per_t[t]["protocol_A"]["ESPLEY54"][m]["test_mae"] for m in MODEL_ORDER) for t in tgts]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(tgts)); w = 0.28
    ax.bar(x - w, pre, w, color="#888888", label="pre-ML (raw xtb feature)")
    ax.bar(x,     best_e5, w, color="#4C72B0", label="best-A · E5 (5 feats)")
    ax.bar(x + w, best_e54, w, color="#C44E52", label="best-A · ESPLEY54 (54 feats)")
    for j, (p, b5, b54) in enumerate(zip(pre, best_e5, best_e54)):
        if np.isfinite(p) and np.isfinite(b54):
            ax.text(x[j] + w, b54 + 0.5, f"×{p / max(b54, 1e-9):.1f}",
                    ha="center", fontsize=8, color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in tgts], rotation=30, ha="right")
    ax.set_ylabel("test MAE (kcal/mol)")
    ax.set_title("Pre-ML vs Post-ML MAE per channel (best of Ridge/KRR/SVR/XGB)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pre_vs_post_mae.png", dpi=140)
    plt.close(fig)


def mae_pct_range_heatmap(report):
    per_t = report["per_target"]
    tgts = list(per_t)
    data = np.zeros((len(tgts), len(MODEL_ORDER) * len(FSET_ORDER)))
    cols = []
    for i_fs, fs in enumerate(FSET_ORDER):
        for i_m, m in enumerate(MODEL_ORDER):
            j = i_fs * len(MODEL_ORDER) + i_m
            cols.append(f"{m}\n{fs}")
            for i_t, t in enumerate(tgts):
                data[i_t, j] = per_t[t]["protocol_A"][fs][m]["test_mae_pct_range"]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=20)
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(tgts)))
    ax.set_yticklabels([TARGET_LABELS.get(t, t) for t in tgts])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > 10 else "black")
    plt.colorbar(im, ax=ax, label="MAE as % of target range")
    ax.set_title("MAE / range (%)  —  lower (green) = better")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_pct_range.png", dpi=140)
    plt.close(fig)


def main():
    report, preds = load()
    print(f"loaded {len(report['per_target'])} targets, {len(preds)} prediction rows")
    mae_per_target_grouped(report)
    print(f"  wrote {FIG_DIR / 'mae_per_target.png'}")
    scatter_per_target(preds)
    print(f"  wrote {sum(1 for _ in FIG_DIR.glob('scatter_dft_*.png'))} per-target scatters")
    scatter_grid_espley54(preds)
    print(f"  wrote {FIG_DIR / 'scatter_grid_espley54.png'}")
    pre_vs_post(report)
    print(f"  wrote {FIG_DIR / 'pre_vs_post_mae.png'}")
    mae_pct_range_heatmap(report)
    print(f"  wrote {FIG_DIR / 'mae_pct_range.png'}")
    print(f"\nfigures dir: {FIG_DIR}")


if __name__ == "__main__":
    main()
