#!/usr/bin/env python3
"""plot_results.py — Protocol A figures of one ML run (ml_report.json + predictions.parquet of aggregate_ml.py).

The 9 reported targets: barrier, d1, d2 and the 6 EDA channels (e_bond, eint_spe and c_ghost are trained but not
plotted: derived quantities / method artefact).

  [ESPLEY_ML_OUT=<dir>] [ESPLEY_GEOM=g1|g2] [ESPLEY_PLOT_ARMS=A,B,..] [ESPLEY_PLOT_MODEL=KRR_rbf] python plot_results.py
Arms: ESPLEY_PLOT_ARMS, else the report's `arms` (rev 5 runs), else ESPLEY<ESPLEY_PLOT_FS> (default 73, rev 4 runs).
Predictions of an arm are the rows with that `arm` (rev 5), or with `feature_set` = its column count (rev 4
predictions, whose arms have distinct counts). The geometry (ESPLEY_GEOM, else the report's geom) goes into every
title: G1 = GFN2-xTB/ALPB geometry, G2 = xTB geometry from SMILES.

Outputs in <ML_OUT>/figures/ (its png / pdf / csv files are removed first; _<geom> suffix when the geometry is known):
    scatter_<model>_<arm>[_<geom>].png            per arm x 4 models (Ridge / KRR / SVR / XGB)
    mae_bar_<arm lower>[_<geom>].png              per arm: 9 targets x 4 models
    mae_bar_arms_<model>[_<geom>].{png,pdf,csv}   with >= 2 arms: 9 panels (targets) x arms for ESPLEY_PLOT_MODEL,
        rev 5 figure style (183 mm wide, 7 pt Arial or DejaVu Sans, fixed arm colours, 2 px surface gaps, value
        labels, light grid, 300 dpi PNG + vector PDF, plotted values as CSV); EXT_SEL carries the D-2 note.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev5_common as R5  # noqa: E402
from aggregate_ml import SEL_NOTE  # noqa: E402
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
PLOT_MODEL = os.environ.get("ESPLEY_PLOT_MODEL") or "KRR_rbf"
LABELS_NOTE = "labels: SMD(water) relabel, method ① (d1/d2 from own-basis fragments)"
GEOM_NOTE = {"g1": "G1 = GFN2-xTB/ALPB geometry", "g2": "G2 = xTB geometry from SMILES"}
# arm colours: the four rev 5 entity colours (rev5_common.COLORS); an arm without one takes the next slot of the
# same validated categorical order (5 magenta, 6 green) plus a hatch as second channel (CVD / grayscale)
ARM_STYLE = {"ESPLEY46": (R5.COLORS["ESPLEY46"], None), "ESPLEY73": (R5.COLORS["ESPLEY73"], None),
             "EXT_SEL": (R5.COLORS["EXT_SEL"], None), "EXT_ALL": ("#e87ba4", "////"), "ESPLEY54": ("#008300", "....")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": R5.FONT_PT,
         "axes.titlesize": R5.FONT_PT, "axes.labelsize": R5.FONT_PT, "xtick.labelsize": R5.FONT_PT - 1,
         "ytick.labelsize": R5.FONT_PT - 1, "legend.fontsize": R5.FONT_PT, "text.color": INK,
         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2,
         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "hatch.linewidth": 0.6,
         "pdf.fonttype": 42, "savefig.facecolor": SURFACE}


def suffix(geom):
    return f"_{geom}" if geom else ""


def caption(geom, arm=None):
    """Title lines below the plot name: geometry + labels (+ the D-2 note for EXT_SEL)."""
    return ((f"{GEOM_NOTE.get(geom, f'geometry {geom}')}\n" if geom else "") + LABELS_NOTE
            + ("\n" + textwrap.fill(SEL_NOTE, 110) if arm == "EXT_SEL" else ""))


def save(fig, path, **kw):
    """Atomic savefig; the format comes from the final suffix (the temporary name ends in .tmp)."""
    R5.write_atomic(path, lambda t: fig.savefig(t, format=path.suffix[1:], **kw))


def arm_preds(preds, arm):
    """Prediction rows of one arm: by `arm` (rev 5), else by the column count (rev 4 arms have distinct counts)."""
    if "arm" in preds:
        return preds[preds["arm"] == arm]
    return preds[preds["feature_set"] == len(FEATURE_SETS[arm])]


def scatter_one(sub_arm, model, arm, geom):
    sub = sub_arm[sub_arm["model"] == model]
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
    cap = caption(geom, arm)
    fig.suptitle(f"{model} · {arm}  —  5-seed test-fold predictions\n{cap}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.022 * (2 + cap.count("\n"))])
    out = FIG_DIR / f"scatter_{model}_{arm}{suffix(geom)}.png"
    save(fig, out, dpi=140)
    plt.close(fig)
    return out


def mae_bar(report, arm, geom):
    """Grouped bar: 9 targets (x) × 4 models (bars). Uses Protocol A test_mae from ml_report.json."""
    per_t = report["per_target"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(CHANNELS))
    w = 0.20
    for i_m, m in enumerate(MODELS):
        maes = []
        sds = []
        for col, _ in CHANNELS:
            A = per_t.get(col, {}).get("protocol_A", {}).get(arm, {})
            r = A.get(m)
            if r is None:
                maes.append(np.nan); sds.append(0)
            else:
                maes.append(r["test_mae"]); sds.append(r["test_mae_sd"])
        offset = (i_m - 1.5) * w
        ax.bar(x + offset, maes, w, yerr=sds, capsize=2,
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
    ax.set_title(f"{arm}  ·  Per-target test MAE (Protocol A)  ·  4 models\n{caption(geom, arm)}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    out = FIG_DIR / f"mae_bar_{arm.lower()}{suffix(geom)}.png"
    save(fig, out, dpi=140)
    plt.close(fig)
    return out


def mae_bar_arms(report, arms, model, geom):
    """Small multiples: one panel per target (own y scale), one bar per arm (mean test MAE, sd over seeds)."""
    per_t = report["per_target"]
    n_feat = report.get("arm_n_features") or {}
    rows = []
    for col, label in CHANNELS:
        for a in arms:
            r = per_t.get(col, {}).get("protocol_A", {}).get(a, {}).get(model)
            if r is None:
                sys.exit(f"mae_bar_arms: no Protocol A {model} result for {col} / {a} in the report")
            rows.append(dict(target=col, label=label, arm=a, n_features=n_feat.get(a), model=model,
                             test_mae=r["test_mae"], test_mae_sd=r["test_mae_sd"], test_nmae=r.get("test_nmae"),
                             test_r2=r["test_r2"], color=ARM_STYLE[a][0], hatch=ARM_STYLE[a][1] or ""))
    tab = pd.DataFrame(rows)
    x = np.arange(len(arms))
    notes = [GEOM_NOTE.get(geom, f"geometry {geom}")] if geom else []
    notes += [LABELS_NOTE] + (textwrap.wrap("† " + SEL_NOTE, 140) if "EXT_SEL" in arms else [])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(R5.FIG_W2, R5.FIG_W2 * 0.75))
        for k, (col, label) in enumerate(CHANNELS):
            ax = axes.flat[k]
            s = tab[tab.target == col].set_index("arm").loc[arms]
            v, sd = s.test_mae.to_numpy(float), s.test_mae_sd.to_numpy(float)
            for i, a in enumerate(arms):
                color, hatch = ARM_STYLE[a]
                name = f"{a}{' †' if a == 'EXT_SEL' else ''} ({n_feat.get(a, '?')})"
                ax.bar(x[i], v[i], width=1.0, color=color, edgecolor=SURFACE, linewidth=0.5, hatch=hatch,
                       label=name, zorder=2)                    # 0.5 pt white edges = 2 px gap at 300 dpi
            ax.errorbar(x, v, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, capthick=0.6,
                        zorder=3)
            top = float(np.max(v + sd))
            for i in range(len(arms)):
                ax.text(x[i], v[i] + sd[i] + 0.02 * top, f"{v[i]:.2f}", ha="center", va="bottom",
                        fontsize=R5.FONT_PT - 1, color=INK2, zorder=4)
            ax.set_ylim(0, top * 1.22)
            ax.set_xlim(-0.6, len(arms) - 0.4)
            ax.set_title(label, color=INK, pad=2)
            ax.set_xticks(x)
            if k >= 6:
                ax.set_xticklabels(arms, rotation=30, ha="right")
            else:
                ax.set_xticklabels([])
            if k % 3 == 0:
                ax.set_ylabel("test MAE (kcal/mol)")
            ax.grid(axis="y", color=GRID, linewidth=0.4)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(f"Protocol A test MAE · {model} · mean ± sd over seeds {'/'.join(map(str, R5.PROTO_A_SEEDS))}",
                     y=0.995, fontsize=R5.FONT_PT, color=INK)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(arms),
                   frameon=False, handlelength=1.4, columnspacing=1.4)
        fig.text(0.01, 0.005, "\n".join(notes), ha="left", va="bottom", fontsize=R5.FONT_PT - 1, color=INK2)
        h_pt = fig.get_figheight() * 72
        bottom = (len(notes) * (R5.FONT_PT - 1) * 1.3 + 6) / h_pt
        fig.tight_layout(rect=[0, bottom, 1, 0.915])
        stem = FIG_DIR / f"mae_bar_arms_{model}{suffix(geom)}"
        save(fig, stem.with_suffix(".png"), dpi=300)
        save(fig, stem.with_suffix(".pdf"))
        plt.close(fig)
    R5.write_atomic(stem.with_suffix(".csv"), lambda f: tab.to_csv(f, index=False))
    return stem


def main():
    preds = pd.read_parquet(ML_OUT / "predictions.parquet")
    report = json.loads((ML_OUT / "ml_report.json").read_text())
    if GEOM and report.get("geom") not in (None, GEOM):
        sys.exit(f"ESPLEY_GEOM={GEOM} but {ML_OUT / 'ml_report.json'} has geom={report.get('geom')}")
    geom = GEOM or report.get("geom")
    if geom is not None and geom not in GEOM_NOTE:
        sys.exit(f"geometry {geom!r} is not one of {sorted(GEOM_NOTE)}")
    env_arms = os.environ.get("ESPLEY_PLOT_ARMS")
    if env_arms:
        arms = [a.strip() for a in env_arms.split(",") if a.strip()]
    elif report.get("arms"):
        arms = list(report["arms"])
    else:
        arms = [f"ESPLEY{int(os.environ.get('ESPLEY_PLOT_FS') or 73)}"]
    have = {a for t in report["per_target"].values() for a in t.get("protocol_A", {})}
    bad = [a for a in arms if a not in have or a not in ARM_STYLE]
    if bad or len(set(arms)) != len(arms):
        sys.exit(f"arms {arms}: {bad} not in the report's Protocol A arms {sorted(have)} / without a colour")
    if "arm" not in preds and any(a not in FEATURE_SETS for a in arms):
        sys.exit(f"predictions have no `arm` column, so only {list(FEATURE_SETS)} can be selected, got {arms}")
    if PLOT_MODEL not in MODELS:
        sys.exit(f"ESPLEY_PLOT_MODEL must be one of {MODELS}, got {PLOT_MODEL!r}")
    print(f"loaded predictions ({len(preds)} rows) + report ({len(report['per_target'])} targets), geom={geom}, "
          f"arms={arms}")

    # FIG_DIR is per ML_OUT (scratch), so everything plotted there before is stale
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(FIG_DIR.iterdir()):
        if p.suffix in (".png", ".pdf", ".csv"):
            p.unlink()
            print(f"  removed {p.name}")

    for arm in arms:
        sub = arm_preds(preds, arm)
        if sub.empty:
            sys.exit(f"no predictions for arm {arm}")
        for m in MODELS:
            out = scatter_one(sub, m, arm, geom)
            print(f"  wrote {out.name}")
        out = mae_bar(report, arm, geom)
        print(f"  wrote {out.name}")
    if len(arms) >= 2:
        stem = mae_bar_arms(report, arms, PLOT_MODEL, geom)
        print(f"  wrote {stem.name}.{{png,pdf,csv}}")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} PNG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
