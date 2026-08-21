#!/usr/bin/env python3
"""Aggregate Phase 1/2/3/4 results (new paper-methodology format) and produce
F1-F5 figures. Design borrowed from 06_aggregate.py.

Phase 1: phase1_espley_paper_reproduction/seed_*/ml_results.pkl
Phase 2/3: phase23_paper/seed_*/ml_results.pkl
Phase 4: phase4_stacked_xgb_mace/fold_*/seed_*/metrics.json + predictions.parquet
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
RESULTS = EXP / "results"
COMP = RESULTS / "comparison_new"
COMP.mkdir(parents=True, exist_ok=True)

PHASE_COLORS = {
    "Phase 1 (paper)":  "#7f7f7f",
    "Phase 2/3 (xTB)":  "#3d9d9b",
    "Phase 4 (XGB+MACE)": "#e07a5f",
}


def extract_paper_pkl(pkl_dir):
    """Extract per-seed test MAE for each model_target from paper's ml_results.pkl."""
    rows = []
    for sdir in sorted(pkl_dir.glob("seed_*")):
        try:
            seed = int(sdir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        pkl = sdir / "ml_results.pkl"
        if not pkl.exists():
            continue
        df = pd.read_pickle(pkl)
        for _, r in df.iterrows():
            mt = r["model_target"]
            yt = r.get("y_test_true"); yp = r.get("y_test_pred_values")
            while isinstance(yt, list) and len(yt) == 1: yt = yt[0]
            while isinstance(yp, list) and len(yp) == 1: yp = yp[0]
            if yt is None or yp is None: continue
            yt, yp = np.asarray(yt).flatten(), np.asarray(yp).flatten()
            if len(yt) != len(yp): continue
            model = mt.split("_")[0]  # ridge/krr/svr/rf/2_st_nn/4_st_nn
            target = "_".join(mt.split("_")[1:])
            # handle 2/4 layer NN prefix
            if mt.startswith("2_st_nn_"):
                model = "2L_NN"; target = mt[len("2_st_nn_"):]
            elif mt.startswith("4_st_nn_"):
                model = "4L_NN"; target = mt[len("4_st_nn_"):]
            mae = np.mean(np.abs(yt - yp))
            rmse = np.sqrt(np.mean((yt - yp) ** 2))
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rows.append({"seed": seed, "model": model, "target": target,
                          "mae": mae, "rmse": rmse, "r2": r2,
                          "n_test": len(yt), "y_range": yt.max() - yt.min()})
    return pd.DataFrame(rows)


def extract_phase4(root):
    """Aggregate Phase 4 XGB+MACE per (fold, seed, target) into test MAE / R²."""
    rows = []
    TARGET_COLS = ["d1_own", "d2_own", "elst_dft", "pauli_dft",
                    "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
    for fdir in sorted(root.glob("fold_*")):
        try:
            fold = int(fdir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        for sdir in sorted(fdir.glob("seed_*")):
            try:
                seed = int(sdir.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            m = sdir / "metrics.json"; p = sdir / "predictions.parquet"
            if not (m.exists() and p.exists()): continue
            metrics = json.loads(m.read_text())
            preds = pd.read_parquet(p)
            # long format: reaction_number, target, y_true, y_pred, y_baseline, delta
            for tgt in TARGET_COLS:
                sub = preds[preds["target"] == tgt]
                if len(sub) == 0: continue
                yt = sub["y_true"].values; yp = sub["y_pred"].values
                mae = np.mean(np.abs(yt - yp))
                rmse = np.sqrt(np.mean((yt - yp) ** 2))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - yt.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rows.append({"seed": seed, "fold": fold, "model": "XGB+MACE",
                              "target": tgt, "mae": mae, "rmse": rmse, "r2": r2,
                              "n_test": len(yt), "y_range": yt.max() - yt.min()})
    return pd.DataFrame(rows)


def build_summary(p1, p23, p4):
    """Aggregate per-phase best-model results per canonical channel."""
    # Canonical channels mapping — Phase-specific label name
    canonical = {
        "d1":          {"Phase 1 (paper)": ("distortion_energy_1_dft", "svr"),
                         "Phase 2/3 (xTB)": ("d1_own_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("d1_own", None)},
        "d2":          {"Phase 1 (paper)": ("distortion_energy_2_dft", "svr"),
                         "Phase 2/3 (xTB)": ("d2_own_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("d2_own", None)},
        "interaction": {"Phase 1 (paper)": ("interaction_energies_dft", "svr"),
                         "Phase 2/3 (xTB)": ("interaction_own_dft", "svr"),
                         "Phase 4 (XGB+MACE)": None},
        "pauli":       {"Phase 1 (paper)": None,
                         "Phase 2/3 (xTB)": ("pauli_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("pauli_dft", None)},
        "oi":          {"Phase 1 (paper)": None,
                         "Phase 2/3 (xTB)": ("oi_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("oi_dft", None)},
        "elst":        {"Phase 1 (paper)": None,
                         "Phase 2/3 (xTB)": ("elst_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("elst_dft", None)},
        "disp":        {"Phase 1 (paper)": None,
                         "Phase 2/3 (xTB)": ("disp_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("disp_dft", None)},
        "cpcm":        {"Phase 1 (paper)": None,
                         "Phase 2/3 (xTB)": ("cpcm_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("cpcm_dft", None)},
        "cds":         {"Phase 1 (paper)": None,
                         "Phase 2/3 (xTB)": ("cds_dft", "svr"),
                         "Phase 4 (XGB+MACE)": ("cds_dft", None)},
    }
    rows = []
    for canon, phase_map in canonical.items():
        for phase, spec in phase_map.items():
            if spec is None: continue
            target, model = spec
            if phase == "Phase 1 (paper)":
                sub = p1[(p1["target"] == target) & (p1["model"] == model)]
            elif phase == "Phase 2/3 (xTB)":
                sub = p23[(p23["target"] == target) & (p23["model"] == model)]
            elif phase == "Phase 4 (XGB+MACE)":
                sub = p4[p4["target"] == target]
            if len(sub) == 0: continue
            for metric in ["mae", "rmse", "r2"]:
                mean_v = sub[metric].mean()
                std_v = sub[metric].std()
                rng = sub["y_range"].iloc[0] if "y_range" in sub.columns else np.nan
                nmae = mean_v / rng if metric == "mae" and rng and rng > 0 else np.nan
                rows.append({"canonical": canon, "phase": phase,
                              "target_used": target, "model": model or "XGB+MACE",
                              "metric": metric, "mean": mean_v, "std": std_v,
                              "y_range": rng, "nmae": nmae if metric == "mae" else np.nan})
    return pd.DataFrame(rows)


def figure_bar(summary, metric, canonicals, title, ylabel, out_path,
                subplots=False, ncol=None, special_channels=(),):
    """Bar chart per canonical channel. If subplots=True, use 2×4 grid."""
    phases = list(PHASE_COLORS.keys())
    if subplots:
        n = len(canonicals)
        ncol = ncol or 4
        nrow = int(np.ceil((n + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2*ncol, 3.4*nrow))
        axes_flat = np.array(axes).flatten()
    else:
        fig, ax = plt.subplots(figsize=(max(6, 1.1*len(canonicals)+3), 4.5))

    x_labels = canonicals
    xpos = np.arange(len(x_labels))
    width = 0.24

    for i, canon in enumerate(canonicals):
        if subplots:
            ax = axes_flat[i]
            ax.set_title(canon, fontsize=10)
            for j, phase in enumerate(phases):
                sub = summary[(summary["canonical"] == canon) &
                              (summary["phase"] == phase) &
                              (summary["metric"] == metric)]
                if len(sub) == 0: continue
                y = sub["mean"].iloc[0]; yerr = sub["std"].iloc[0]
                ax.bar([j], [y], width=0.7, yerr=[yerr] if not np.isnan(yerr) else None,
                        color=PHASE_COLORS[phase], label=phase if i == 0 else "",
                        capsize=3)
            ax.set_xticks([]); ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", alpha=0.25)
        else:
            for j, phase in enumerate(phases):
                sub = summary[(summary["canonical"] == canon) &
                              (summary["phase"] == phase) &
                              (summary["metric"] == metric)]
                if len(sub) == 0: continue
                y = sub["mean"].iloc[0]; yerr = sub["std"].iloc[0]
                offset = (j - (len(phases)-1)/2) * width
                ax.bar(xpos[i] + offset, y, width=width,
                        yerr=yerr if not np.isnan(yerr) else None,
                        color=PHASE_COLORS[phase], label=phase if i == 0 else "",
                        capsize=3)

    if subplots:
        # legend in last cell
        legend_ax = axes_flat[len(canonicals)]
        legend_ax.axis("off")
        handles = [plt.Rectangle((0,0), 1, 1, color=PHASE_COLORS[p]) for p in phases]
        legend_ax.legend(handles, phases, loc="center", frameon=True, fontsize=10)
        for k in range(len(canonicals)+1, len(axes_flat)):
            axes_flat[k].axis("off")
    else:
        ax.set_xticks(xpos); ax.set_xticklabels(x_labels, rotation=0)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3); ax.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    print("=" * 70)
    print("Aggregate + F1-F5 figures (paper-methodology format)")
    print("=" * 70)

    p1  = extract_paper_pkl(RESULTS / "phase1_espley_paper_reproduction")
    p23 = extract_paper_pkl(RESULTS / "phase23_paper")
    p4  = extract_phase4(RESULTS / "phase4_stacked_xgb_mace")
    print(f"Phase 1: {len(p1)} rows (targets: {p1['target'].nunique()})")
    print(f"Phase 2/3: {len(p23)} rows (targets: {p23['target'].nunique()})")
    print(f"Phase 4: {len(p4)} rows (targets: {p4['target'].nunique()})")

    summary = build_summary(p1, p23, p4)
    summary.to_csv(COMP / "summary_all.csv", index=False)
    print(f"summary_all.csv: {len(summary)} rows")

    d12 = ["d1", "d2"]
    other = ["interaction", "pauli", "oi", "elst", "disp", "cpcm", "cds"]

    # F1: MAE d1, d2
    figure_bar(summary, "mae", d12, "F1: MAE — d1, d2 (strain)",
                "MAE (kcal/mol)", COMP / "F1_MAE_d12.png")
    # F2: MAE interaction + 6 EDA (2×4 grid)
    figure_bar(summary, "mae", other,
                "F2: MAE — interaction + 6 EDA channels",
                "MAE (kcal/mol)", COMP / "F2_MAE_other.png", subplots=True, ncol=4)
    # F3: NMAE d1, d2 (use nmae column via metric='mae' + nmae)
    # Build a "nmae" metric view
    summary_nmae = summary[summary["metric"] == "mae"].copy()
    summary_nmae["metric"] = "nmae_view"
    summary_nmae["mean"] = summary_nmae["nmae"]
    summary_nmae["std"] = summary_nmae["std"] / summary_nmae["y_range"]
    combined = pd.concat([summary, summary_nmae])
    figure_bar(combined, "nmae_view", d12, "F3: NMAE — d1, d2",
                "NMAE = MAE / y-range", COMP / "F3_NMAE_d12.png")
    figure_bar(combined, "nmae_view", other,
                "F4: NMAE — interaction + 6 EDA channels",
                "NMAE = MAE / y-range", COMP / "F4_NMAE_other.png", subplots=True, ncol=4)
    # F5: R² for all channels
    figure_bar(summary, "r2", d12 + other,
                "F5: R² per channel", "R²", COMP / "F5_R2.png")

    # REPORT.md
    with open(COMP / "REPORT.md", "w") as f:
        f.write("# Phase 1 vs 2/3 vs 4 Comparison (paper-methodology)\n\n")
        f.write("Dataset: bath_1480, 3504 rxns\n\n")
        f.write("## Per-channel test MAE (mean ± std across 5 seeds, best model)\n\n")
        f.write("| Canonical | Phase 1 (paper) | Phase 2/3 (xTB) | Phase 4 (XGB+MACE) |\n")
        f.write("|---|---:|---:|---:|\n")
        for canon in d12 + other:
            row_str = f"| **{canon}** |"
            for phase in PHASE_COLORS.keys():
                sub = summary[(summary["canonical"] == canon) &
                              (summary["phase"] == phase) &
                              (summary["metric"] == "mae")]
                if len(sub) == 0:
                    row_str += " — |"
                else:
                    row_str += f" {sub['mean'].iloc[0]:.3f} ± {sub['std'].iloc[0]:.3f} |"
            f.write(row_str + "\n")

    print(f"\nDone. Output: {COMP}")


if __name__ == "__main__":
    main()
