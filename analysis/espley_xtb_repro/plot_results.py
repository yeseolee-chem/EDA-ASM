#!/usr/bin/env python3
"""plot_results.py — scatter (per-model) + MAE bar (combined).

One feature set (ESPLEY_PLOT_FS, default 73 = ESPLEY73), the 9 reported targets:
barrier, d1, d2 and the 6 EDA channels (e_bond, eint_spe and c_ghost are trained
but not plotted: derived quantities / method artefact).

  [ESPLEY_ML_OUT=<dir>] [ESPLEY_GEOM=g0|g1] python plot_results.py
Reads <ML_OUT>/predictions.parquet + ml_report.json (ML_OUT default $ESPLEY_OUT). The geometry (ESPLEY_GEOM, else
the report's geom) goes into every title: G0 = DFT oracle geometry (upper bound), G1 = GFN2-xTB/ALPB geometry.
The disp panel carries the "analytic identity" note only on G0 (and rev 3, no geom), where b_disp is the disp label,
and only for a feature set that contains b_disp (not ESPLEY46).

Outputs (in <ML_OUT>/figures/; existing PNGs there are removed first; _<geom> suffix when the geometry is known):
    scatter_<model>_ESPLEY<fs>[_<geom>].png   × 4 models (Ridge/KRR/SVR/XGB)
    mae_bar_espley<fs>[_<geom>].png           — grouped bars: 9 targets × 4 models
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ml_single import FEATURE_SETS  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
ML_OUT = Path(os.environ.get("ESPLEY_ML_OUT") or ROOT)
GEOM = os.environ.get("ESPLEY_GEOM") or None
FIG_DIR = ML_OUT / "figures"

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
GEOM_NOTE = {"g0": "G0 = DFT oracle geometry (upper bound)", "g1": "G1 = GFN2-xTB/ALPB geometry"}
# b_disp (feature) is the target by definition on the DFT geometry: its "prediction" is read off, not learned
DISP = "dft_disp_dft"
DISP_NOTE = "same quantity as the b_disp feature\n(analytic identity, not a prediction)"


def caption(geom):
    """Title lines below the plot name: geometry (rev 4) + labels."""
    return (f"{GEOM_NOTE.get(geom, f'geometry {geom}')}\n" if geom else "") + LABELS_NOTE


def disp_is_identity(geom):
    # DFT geometry (rev 3 / G0) and a feature set with b_disp; on G1, or in ESPLEY46, disp is a genuine prediction
    return geom in (None, "g0") and "b_disp" in FEATURE_SETS.get(FS, [])


def scatter_one(preds, model, geom):
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
        if col == DISP and disp_is_identity(geom):
            ax.text(0.04, 0.96, DISP_NOTE, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.6))
    fig.suptitle(f"{model} · {FS}  —  5-seed test-fold predictions\n{caption(geom)}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955 if geom is None else 0.935])
    out = FIG_DIR / f"scatter_{model}_{FS}{f'_{geom}' if geom else ''}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def mae_bar(report, geom):
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
    if disp_is_identity(geom):
        j = [c for c, _ in CHANNELS].index(DISP)
        ax.text(x[j], 0.55, DISP_NOTE, ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CHANNELS], rotation=25, ha="right")
    ax.set_ylabel("test MAE (kcal/mol, mean ± sd over 5 seeds)")
    ax.set_title(f"{FS}  ·  Per-target test MAE (Protocol A)  ·  4 models\n{caption(geom)}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    out = FIG_DIR / f"mae_bar_{FS.lower()}{f'_{geom}' if geom else ''}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    preds = pd.read_parquet(ML_OUT / "predictions.parquet")
    report = json.loads((ML_OUT / "ml_report.json").read_text())
    if GEOM and report.get("geom") not in (None, GEOM):
        sys.exit(f"ESPLEY_GEOM={GEOM} but {ML_OUT / 'ml_report.json'} has geom={report.get('geom')}")
    geom = GEOM or report.get("geom")
    print(f"loaded predictions ({len(preds)} rows) + report ({len(report['per_target'])} targets), geom={geom}")

    # remove all existing PNGs (FIG_DIR is per ML_OUT, i.e. per geometry in rev 4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in FIG_DIR.glob("*.png"):
        p.unlink()
        print(f"  removed {p.name}")

    for m in MODELS:
        out = scatter_one(preds, m, geom)
        print(f"  wrote {out.name}")
    out = mae_bar(report, geom)
    print(f"  wrote {out.name}")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
