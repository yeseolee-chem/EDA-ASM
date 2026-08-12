#!/usr/bin/env python3
"""Phase 5: aggregate 4-phase results, single-row bar chart per metric.

Design:
- Phase 1 predicted 6 targets but we only display 3 comparable ones (d1, d2, interaction)
- Phase 2/3/4 display all 8 predicted (d1, d2, 6 EDA channels)
- Single figure per metric, x-axis = channels lined up left-to-right
- Within each channel: bars for each (phase, model) config
- Total 9 unique channels shown side by side
"""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results")
COMP = RESULTS / "comparison_v1"
COMP.mkdir(parents=True, exist_ok=True)

# Phase 1: predict 3 channels only (filter from 6 predicted)
# Phase 2/3/4: predict 8 channels each
PHASE_TARGETS = {
    "phase1_espley": ["d1_dft_espley", "d2_dft_espley", "interaction_espley"],
    "phase2_armB_tau1e-10": ["d1_own", "d2_own", "elst_dft", "pauli_dft",
                              "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"],
    "phase3_armB_tau0.05":  ["d1_own", "d2_own", "elst_dft", "pauli_dft",
                              "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"],
    "phase4_stacked_xgb_mace": ["d1_own", "d2_own", "elst_dft", "pauli_dft",
                                 "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"],
}

PHASE_MODELS = {
    "phase1_espley":       ["ridge", "krr", "svr"],
    "phase2_armB_tau1e-10": ["ridge", "krr", "svr"],
    "phase3_armB_tau0.05":  ["ridge", "krr", "svr"],
    "phase4_stacked_xgb_mace": ["stacked"],
}

# Canonical target → variant per phase
CANONICAL_TO_PHASE_TARGET = {
    "d1":          {"phase1_espley": "d1_dft_espley",
                    "phase2_armB_tau1e-10": "d1_own",
                    "phase3_armB_tau0.05":  "d1_own",
                    "phase4_stacked_xgb_mace": "d1_own"},
    "d2":          {"phase1_espley": "d2_dft_espley",
                    "phase2_armB_tau1e-10": "d2_own",
                    "phase3_armB_tau0.05":  "d2_own",
                    "phase4_stacked_xgb_mace": "d2_own"},
    "interaction": {"phase1_espley": "interaction_espley"},
    "elst":  {"phase2_armB_tau1e-10": "elst_dft",  "phase3_armB_tau0.05": "elst_dft",  "phase4_stacked_xgb_mace": "elst_dft"},
    "pauli": {"phase2_armB_tau1e-10": "pauli_dft", "phase3_armB_tau0.05": "pauli_dft", "phase4_stacked_xgb_mace": "pauli_dft"},
    "oi":    {"phase2_armB_tau1e-10": "oi_dft",    "phase3_armB_tau0.05": "oi_dft",    "phase4_stacked_xgb_mace": "oi_dft"},
    "disp":  {"phase2_armB_tau1e-10": "disp_dft",  "phase3_armB_tau0.05": "disp_dft",  "phase4_stacked_xgb_mace": "disp_dft"},
    "cpcm":  {"phase2_armB_tau1e-10": "cpcm_dft",  "phase3_armB_tau0.05": "cpcm_dft",  "phase4_stacked_xgb_mace": "cpcm_dft"},
    "cds":   {"phase2_armB_tau1e-10": "cds_dft",   "phase3_armB_tau0.05": "cds_dft",   "phase4_stacked_xgb_mace": "cds_dft"},
}
CANONICAL_ORDER = ["d1", "d2", "interaction", "elst", "pauli", "oi", "disp", "cpcm", "cds"]

PHASE_COLORS = {
    "phase1_espley":            "#7f7f7f",
    "phase2_armB_tau1e-10":     "#1f77b4",
    "phase3_armB_tau0.05":      "#2ca02c",
    "phase4_stacked_xgb_mace":  "#d62728",
}
PHASE_SHORT = {
    "phase1_espley":            "P1-espley",
    "phase2_armB_tau1e-10":     "P2-armB-τ1e-10",
    "phase3_armB_tau0.05":      "P3-armB-τ0.05",
    "phase4_stacked_xgb_mace":  "P4-stacked",
}


def load_phase123(phase_name):
    rows = []
    for seed_dir in sorted((RESULTS / phase_name).glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        for model_file in seed_dir.glob("*_results.pkl"):
            model = model_file.stem.replace("_results", "")
            data = pickle.load(model_file.open("rb"))
            for target, fold_results in data.items():
                if target not in PHASE_TARGETS[phase_name]:
                    continue  # filter to displayed targets only
                for r in fold_results:
                    rows.append({
                        "phase": phase_name, "model": model, "seed": seed,
                        "target": target, "fold": r["fold"],
                        "mae": r["test_mae"], "rmse": r["test_rmse"], "r2": r["test_r2"],
                        "n_test": len(r["y_true"]),
                    })
    return pd.DataFrame(rows)


def load_phase4(phase_name):
    rows = []
    for fold_dir in sorted((RESULTS / phase_name).glob("fold_*")):
        fold = int(fold_dir.name.split("_")[1])
        for seed_dir in sorted(fold_dir.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            metrics_json = seed_dir / "metrics.json"
            if not metrics_json.exists():
                continue
            metrics = json.loads(metrics_json.read_text())
            for target, m in metrics.items():
                if target not in PHASE_TARGETS[phase_name]:
                    continue
                rows.append({
                    "phase": phase_name, "model": "stacked", "seed": seed,
                    "target": target, "fold": fold,
                    "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                    "n_test": None,
                })
    return pd.DataFrame(rows)


def compute_nmae(df, labels_pkl):
    labels = pd.read_pickle(labels_pkl)
    y_range = {}
    for c in df["target"].unique():
        if c in labels.columns:
            y_range[c] = labels[c].max() - labels[c].min()
        else:
            y_range[c] = 1.0
    df = df.copy()
    df["nmae"] = df.apply(lambda r: r["mae"] / y_range.get(r["target"], 1.0), axis=1)
    return df


def make_bar_positions():
    """Build x-positions for each (canonical, phase, model) bar in single-row layout.
    Returns list of (canonical, phase, model, x_pos, color, label).
    """
    positions = []
    x = 0.0
    bar_w = 0.9
    intra_gap = 0.1  # between bars in same channel group
    inter_gap = 1.2  # between channel groups
    tick_centers = []  # channel group center x-positions
    tick_labels = []

    for canonical in CANONICAL_ORDER:
        variants = CANONICAL_TO_PHASE_TARGET[canonical]
        # collect (phase, model) pairs that have data for this channel
        entries = []
        for phase in ["phase1_espley", "phase2_armB_tau1e-10",
                       "phase3_armB_tau0.05", "phase4_stacked_xgb_mace"]:
            if phase not in variants:
                continue
            target_col = variants[phase]
            for model in PHASE_MODELS[phase]:
                entries.append((canonical, phase, model, target_col))
        if not entries:
            continue
        group_start = x
        for canonical, phase, model, target_col in entries:
            color = PHASE_COLORS[phase]
            label = f"{PHASE_SHORT[phase]}/{model}"
            positions.append((canonical, phase, model, target_col, x, color, label))
            x += bar_w + intra_gap
        group_end = x - intra_gap
        tick_centers.append((group_start + group_end) / 2)
        tick_labels.append(canonical)
        x += inter_gap
    return positions, tick_centers, tick_labels, bar_w


def figure_metric_single(agg_df, metric_col, title, ylabel, out_path):
    """Single subplot with all 9 channels lined up (used for R²)."""
    positions, tick_centers, tick_labels, bar_w = make_bar_positions()
    fig, ax = plt.subplots(figsize=(max(14, len(positions) * 0.32), 5.5))
    seen_labels = set()
    for canonical, phase, model, target_col, x, color, label in positions:
        sub = agg_df[(agg_df["phase"] == phase) &
                     (agg_df["model"] == model) &
                     (agg_df["target"] == target_col)]
        if len(sub) == 0:
            continue
        mean = sub[metric_col].mean()
        std = sub[metric_col].std()
        lbl = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.bar(x, mean, width=bar_w, color=color, yerr=std, capsize=2,
               edgecolor="black", linewidth=0.4, label=lbl)
    ax.set_xticks(tick_centers)
    ax.set_xticklabels(tick_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.9,
               handlelength=1.2, handletextpad=0.3, columnspacing=0.6, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def figure_metric_subplots(agg_df, metric_col, canonicals, title, ylabel, out_path,
                             special_channels=(), grid_2x4=False):
    """One subplot per canonical channel.
    Channels NOT in special_channels share a common y-axis (unified scale).
    Channels in special_channels get their own auto-scale.
    grid_2x4: if True, arrange 7 subplots as 2 rows × 4 cols (top 4, bottom 3 + legend slot),
              figure sized ~16:9.
    """
    n = len(canonicals)
    if grid_2x4:
        # 2 rows × 4 cols; last cell reserved for legend
        fig, axes_grid = plt.subplots(2, 4, figsize=(14, 7.9))
        axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[0, 2], axes_grid[0, 3],
                axes_grid[1, 0], axes_grid[1, 1], axes_grid[1, 2]]
        legend_ax = axes_grid[1, 3]
        legend_ax.axis("off")
    else:
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 1.0, 4.4), sharey=False)
        if n == 1:
            axes = [axes]
        legend_ax = None

    # Pre-compute means for unified y-lim of shared subplots
    shared_max = 0.0
    for canonical in canonicals:
        if canonical in special_channels:
            continue
        variants = CANONICAL_TO_PHASE_TARGET.get(canonical, {})
        for phase in variants:
            for model in PHASE_MODELS[phase]:
                sub = agg_df[(agg_df["phase"] == phase) &
                             (agg_df["model"] == model) &
                             (agg_df["target"] == variants[phase])]
                if len(sub):
                    mean = sub[metric_col].mean()
                    std = sub[metric_col].std()
                    shared_max = max(shared_max, mean + std)

    seen_labels = set()
    for ax, canonical in zip(axes, canonicals):
        variants = CANONICAL_TO_PHASE_TARGET.get(canonical, {})
        entries = []
        for phase in ["phase1_espley", "phase2_armB_tau1e-10",
                       "phase3_armB_tau0.05", "phase4_stacked_xgb_mace"]:
            if phase not in variants:
                continue
            for model in PHASE_MODELS[phase]:
                entries.append((phase, model, variants[phase]))

        xs = np.arange(len(entries))
        for i, (phase, model, target_col) in enumerate(entries):
            sub = agg_df[(agg_df["phase"] == phase) &
                         (agg_df["model"] == model) &
                         (agg_df["target"] == target_col)]
            if len(sub) == 0:
                continue
            mean = sub[metric_col].mean()
            std = sub[metric_col].std()
            color = PHASE_COLORS[phase]
            label = f"{PHASE_SHORT[phase]}/{model}"
            lbl = label if label not in seen_labels else None
            seen_labels.add(label)
            ax.bar(i, mean, width=0.8, color=color, yerr=std, capsize=2,
                   edgecolor="black", linewidth=0.4, label=lbl)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{PHASE_SHORT[p]}\n{m}" for p, m, _ in entries],
                           fontsize=6, rotation=45, ha="right")
        title_str = canonical + (" (own scale)" if canonical in special_channels else "")
        ax.set_title(title_str, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # Apply unified ylim to non-special channels
        if canonical not in special_channels and shared_max > 0:
            ax.set_ylim(0, shared_max * 1.1)

    # suptitle removed per user request
    # Collect unique legend entries
    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if legend_ax is not None:
        # Legend in the reserved empty subplot cell
        legend_ax.legend(handles, labels, loc="center", fontsize=8,
                         ncol=1, framealpha=0.95, title="Configs",
                         title_fontsize=9, handlelength=1.6,
                         handletextpad=0.5, borderpad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.legend(handles, labels, loc="lower center", fontsize=6,
                   bbox_to_anchor=(0.5, -0.02), ncol=min(len(labels), 5),
                   framealpha=0.9, handlelength=1.2, handletextpad=0.3,
                   columnspacing=0.8, borderpad=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    print("=" * 70)
    print("PHASE 5: aggregate + single-row figures")
    print("=" * 70)

    all_dfs = []
    for phase in PHASE_TARGETS:
        if phase == "phase4_stacked_xgb_mace":
            df = load_phase4(phase)
        else:
            df = load_phase123(phase)
        if len(df):
            all_dfs.append(df)
            print(f"  {phase}: {len(df)} rows (targets: {sorted(df['target'].unique())})")
    agg = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if agg.empty:
        print("No results.")
        return

    # NMAE via label range
    agg = compute_nmae(agg, "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/labels_v1.pkl")

    agg.to_parquet(COMP / "all_metrics.parquet", index=False)
    print(f"Wrote all_metrics.parquet ({len(agg)} rows)")

    # Figures — 5 total
    print("\n=== Figures ===")
    d12 = ["d1", "d2"]
    other = ["interaction", "elst", "pauli", "oi", "disp", "cpcm", "cds"]
    # F1: MAE d1,d2 (shared y-axis, no special)
    figure_metric_subplots(agg, "mae", d12,
        "F1: MAE — d1, d2 (strain)", "MAE (kcal/mol)",
        COMP / "F1_MAE_d12.png", special_channels=())
    # F2: MAE 나머지 (2×4 grid, disp/cds own scale, legend in 8th cell)
    figure_metric_subplots(agg, "mae", other,
        "F2: MAE — interaction + 6 EDA channels (disp, cds have own scale)",
        "MAE (kcal/mol)", COMP / "F2_MAE_other.png",
        special_channels=("disp", "cds"), grid_2x4=True)
    # F3: NMAE d1,d2 (shared y-axis, no special)
    figure_metric_subplots(agg, "nmae", d12,
        "F3: NMAE — d1, d2", "NMAE = MAE / y-range",
        COMP / "F3_NMAE_d12.png", special_channels=())
    # F4: NMAE 나머지 (2×4 grid, disp own scale, legend in 8th cell)
    figure_metric_subplots(agg, "nmae", other,
        "F4: NMAE — interaction + 6 EDA channels (disp has own scale)",
        "NMAE = MAE / y-range", COMP / "F4_NMAE_other.png",
        special_channels=("disp",), grid_2x4=True)
    # F5: R² 전체 (single-row layout, unchanged from user request)
    figure_metric_single(agg, "r2",
        "F5: R² per channel (all configs side-by-side)", "R²",
        COMP / "F5_R2.png")

    # Summary
    summary = agg.groupby(["phase", "model", "target"]).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        nmae_mean=("nmae", "mean"), nmae_std=("nmae", "std"),
        r2_mean=("r2", "mean"),
    ).reset_index()
    summary.to_csv(COMP / "summary.csv", index=False)
    print(f"Wrote summary.csv ({len(summary)} rows)")

    # Report
    with open(COMP / "REPORT.md", "w") as f:
        f.write("# 4-Phase Experiment (MAE + NMAE)\n\n")
        f.write(f"- Cohort: 926 rxns (5-fold × 5-seed CV, 25 evaluations per config)\n")
        f.write(f"- Phase 1 (Espley baseline): predicts 3 channels (d1, d2, interaction)\n")
        f.write(f"- Phase 2/3/4: predict 8 channels (d1, d2, elst, pauli, oi, disp, cpcm, cds)\n\n")
        f.write("## Summary (all configs × targets)\n\n")
        f.write(summary.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
