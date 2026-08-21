#!/usr/bin/env python3
"""Aggregate Phase 1/2/3/4 with per-model bars.

Interpretation:
  Phase 1 = paper reproduction (Espley features only, Espley labels)
  Phase 2 = paper methodology on xTB-augmented features, Espley original labels
  Phase 3 = paper methodology on xTB-augmented features, our own labels + EDA channels
  Phase 4 = XGB + MACE residual, our own labels + EDA channels

Both Phase 2 and Phase 3 share phase23_paper/ ml_results.pkl but split by target set.
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
COMP = RESULTS / "comparison_final"
COMP.mkdir(parents=True, exist_ok=True)

# Phase colors — visually distinct
PHASE_COLORS = {
    "Phase 1 (paper)":     "#7f7f7f",  # grey
    "Phase 2 (τ=1e-10)":   "#4c9fd8",  # blue
    "Phase 3 (τ=0.05)":    "#3d9d9b",  # teal
    "Phase 4 (XGB+MACE)":  "#e07a5f",  # coral
    "Phase 5 (no filter)": "#9b59b6",  # purple
}
MODEL_ORDER = ["ridge", "krr", "svr", "rf", "xgb"]

# Which targets belong to each phase
# Phase 1: paper reproduction — Espley original DFT labels only (no EDA channels available)
PHASE1_TARGETS = {
    "d1":          "distortion_energy_1_dft",
    "d2":          "distortion_energy_2_dft",
    "interaction": "interaction_energies_dft",
}
# Phase 2/3/5 all use our own labels + EDA channels
COMMON_OWN_TARGETS = {
    "d1":          "d1_own_dft",
    "d2":          "d2_own_dft",
    "interaction": "interaction_own_dft",
    "pauli":       "pauli_dft",
    "oi":          "oi_dft",
    "elst":        "elst_dft",
    "disp":        "disp_dft",
    "cpcm":        "cpcm_dft",
    "cds":         "cds_dft",
}
PHASE2_TARGETS = COMMON_OWN_TARGETS
PHASE3_TARGETS = COMMON_OWN_TARGETS
PHASE5_TARGETS = COMMON_OWN_TARGETS
PHASE4_TARGETS = {  # matches TARGET_COLS in 05_train_e3.py — column name differs (no _dft suffix)
    "d1":          "d1_own",
    "d2":          "d2_own",
    "pauli":       "pauli_dft",
    "oi":          "oi_dft",
    "elst":        "elst_dft",
    "disp":        "disp_dft",
    "cpcm":        "cpcm_dft",
    "cds":         "cds_dft",
}


def extract_paper_pkl(pkl_dir):
    rows = []
    for sdir in sorted(pkl_dir.glob("seed_*")):
        try: seed = int(sdir.name.split("_")[1])
        except: continue
        pkl = sdir / "ml_results.pkl"
        if not pkl.exists(): continue
        df = pd.read_pickle(pkl)
        for _, r in df.iterrows():
            mt = r["model_target"]
            yt = r.get("y_test_true"); yp = r.get("y_test_pred_values")
            while isinstance(yt, list) and len(yt) == 1: yt = yt[0]
            while isinstance(yp, list) and len(yp) == 1: yp = yp[0]
            if yt is None or yp is None: continue
            yt = np.asarray(yt).flatten(); yp = np.asarray(yp).flatten()
            if len(yt) != len(yp): continue
            # parse model + target
            if mt.startswith("2_st_nn_"):
                model = "2L_NN"; target = mt[len("2_st_nn_"):]
            elif mt.startswith("4_st_nn_"):
                model = "4L_NN"; target = mt[len("4_st_nn_"):]
            else:
                model = mt.split("_")[0]; target = "_".join(mt.split("_")[1:])
            mae = np.mean(np.abs(yt - yp))
            ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rows.append({"seed": seed, "model": model, "target": target,
                          "mae": mae, "r2": r2, "y_range": yt.max() - yt.min()})
    return pd.DataFrame(rows)


def extract_xgb(pkl_dir):
    """Load xgb_results.pkl per seed dir. Returns rows compatible with paper format."""
    rows = []
    for sdir in sorted(pkl_dir.glob("seed_*")):
        try: seed = int(sdir.name.split("_")[1])
        except: continue
        pkl = sdir / "xgb_results.pkl"
        if not pkl.exists(): continue
        d = pickle.load(open(pkl, "rb"))
        for target, res in d.items():
            yt = np.asarray(res["y_test_true"]).flatten()
            rows.append({"seed": seed, "model": "xgb", "target": target,
                          "mae": res["test_mae"], "r2": res["test_r2"],
                          "y_range": yt.max() - yt.min()})
    return pd.DataFrame(rows)


def extract_phase4(root):
    rows = []
    for fdir in sorted(root.glob("fold_*")):
        try: fold = int(fdir.name.split("_")[1])
        except: continue
        for sdir in sorted(fdir.glob("seed_*")):
            try: seed = int(sdir.name.split("_")[1])
            except: continue
            p = sdir / "predictions.parquet"
            if not p.exists(): continue
            preds = pd.read_parquet(p)
            for tgt in preds["target"].unique():
                sub = preds[preds["target"] == tgt]
                yt = sub["y_true"].values; yp = sub["y_pred"].values
                mae = np.mean(np.abs(yt - yp))
                ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rows.append({"seed": seed, "fold": fold, "model": "XGB+MACE",
                              "target": tgt, "mae": mae, "r2": r2,
                              "y_range": yt.max() - yt.min()})
    return pd.DataFrame(rows)


def build_records(p1, p2, p3, p5, p4):
    """Long-format: (phase, canonical, model, mae_mean, mae_std, r2_mean, y_range)."""
    rows = []
    def add_phase(phase, src_df, mapping, models):
        for canon, target in mapping.items():
            for m in models:
                sub = src_df[(src_df["target"] == target) & (src_df["model"] == m)]
                if len(sub) == 0: continue
                rng = sub["y_range"].iloc[0]
                rows.append({
                    "phase": phase, "canonical": canon, "model": m,
                    "target_used": target,
                    "mae_mean": sub["mae"].mean(), "mae_std": sub["mae"].std(),
                    "r2_mean": sub["r2"].mean(), "r2_std": sub["r2"].std(),
                    "y_range": rng, "n_seeds": len(sub),
                })

    add_phase("Phase 1 (paper)",     p1, PHASE1_TARGETS, MODEL_ORDER)
    add_phase("Phase 2 (τ=1e-10)",   p2, PHASE2_TARGETS, MODEL_ORDER)
    add_phase("Phase 3 (τ=0.05)",    p3, PHASE3_TARGETS, MODEL_ORDER)
    add_phase("Phase 5 (no filter)", p5, PHASE5_TARGETS, MODEL_ORDER)
    # Phase 4 has only one model per cell (XGB+MACE)
    add_phase("Phase 4 (XGB+MACE)",  p4, PHASE4_TARGETS, ["XGB+MACE"])
    return pd.DataFrame(rows)


def figure_grouped_bars(records, canonicals, metric_col, err_col, title, ylabel, out_path,
                          subplots=False, ncol=4, ymin=None,
                          phase_filter=None, exclude=None):
    """
    phase_filter: list of phase names to include (None = all)
    exclude: list of (phase, model) tuples to skip
    """
    if phase_filter is not None:
        records = records[records["phase"].isin(phase_filter)].copy()
    if exclude:
        for ph, mo in exclude:
            records = records[~((records["phase"] == ph) & (records["model"] == mo))]
    """Grouped bars per canonical channel: outer = phase, inner = model."""
    phases = [p for p in PHASE_COLORS.keys() if phase_filter is None or p in phase_filter]

    if subplots:
        n = len(canonicals)
        nrow = int(np.ceil((n + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6*ncol, 3.8*nrow))
        axes_flat = np.array(axes).flatten()
    else:
        fig, ax = plt.subplots(figsize=(max(7, 1.6*len(canonicals)+3), 5.0))

    # For each canonical channel, plot grouped bars
    for i, canon in enumerate(canonicals):
        target_ax = axes_flat[i] if subplots else ax
        # Collect data for this canonical
        # Per phase, list of (model, mae, std)
        bar_positions = []
        bar_heights = []
        bar_errs = []
        bar_colors = []
        bar_labels = []  # x labels
        cursor = 0.0
        legend_seen = set()
        for phase in phases:
            phase_rows = records[(records["canonical"] == canon) & (records["phase"] == phase)]
            if len(phase_rows) == 0: continue
            models_here = phase_rows["model"].unique().tolist()
            # Preserve MODEL_ORDER
            models_here = [m for m in MODEL_ORDER + ["XGB+MACE"] if m in models_here]
            for m in models_here:
                sub = phase_rows[phase_rows["model"] == m]
                if len(sub) == 0: continue
                bar_positions.append(cursor)
                bar_heights.append(sub[metric_col].iloc[0])
                bar_errs.append(sub[err_col].iloc[0] if err_col in sub.columns else 0)
                bar_colors.append(PHASE_COLORS[phase])
                bar_labels.append(m if m != "XGB+MACE" else "XGB")
                cursor += 1.0
                if phase not in legend_seen and not subplots:
                    legend_seen.add(phase)
            cursor += 0.5  # gap between phases

        if bar_positions:
            target_ax.bar(bar_positions, bar_heights, width=0.8,
                          yerr=bar_errs, color=bar_colors, capsize=2.5,
                          edgecolor="black", linewidth=0.4)
            target_ax.set_xticks(bar_positions)
            target_ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
            target_ax.set_title(canon, fontsize=10)
            target_ax.set_ylabel(ylabel, fontsize=9)
            target_ax.grid(axis="y", alpha=0.25)
            if ymin is not None:
                target_ax.set_ylim(bottom=ymin)

    if subplots:
        # Legend in extra cell
        legend_ax = axes_flat[len(canonicals)]
        legend_ax.axis("off")
        handles = [plt.Rectangle((0,0), 1, 1, color=PHASE_COLORS[p]) for p in phases]
        legend_ax.legend(handles, phases, loc="center", frameon=True, fontsize=11, title="Phase")
        for k in range(len(canonicals)+1, len(axes_flat)):
            axes_flat[k].axis("off")
    else:
        handles = [plt.Rectangle((0,0), 1, 1, color=PHASE_COLORS[p]) for p in phases]
        ax.legend(handles, phases, loc="best", fontsize=9, title="Phase")

    fig.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    print("Aggregating with per-model bars (Phase 1/2/3/4 separate)")

    p1  = extract_paper_pkl(RESULTS / "phase1_espley_paper_reproduction")
    p2  = extract_paper_pkl(RESULTS / "phase2_paper")
    p3  = extract_paper_pkl(RESULTS / "phase3_paper")
    p5  = extract_paper_pkl(RESULTS / "phase5_paper")
    p4  = extract_phase4(RESULTS / "phase4_stacked_xgb_mace")
    # Merge XGB rows into p2/p3/p5
    p2 = pd.concat([p2, extract_xgb(RESULTS / "phase2_paper")], ignore_index=True)
    p3 = pd.concat([p3, extract_xgb(RESULTS / "phase3_paper")], ignore_index=True)
    p5 = pd.concat([p5, extract_xgb(RESULTS / "phase5_paper")], ignore_index=True)
    for name, df in [("Phase 1", p1), ("Phase 2", p2), ("Phase 3", p3),
                       ("Phase 4", p4), ("Phase 5", p5)]:
        print(f"  {name}: {len(df)} rows")

    records = build_records(p1, p2, p3, p5, p4)
    records.to_csv(COMP / "records_all.csv", index=False)
    print(f"records_all.csv: {len(records)} rows")

    d12 = ["d1", "d2"]
    other = ["interaction", "pauli", "oi", "elst", "disp", "cpcm", "cds"]

    # NMAE column
    records["nmae_mean"] = records["mae_mean"] / records["y_range"]
    records["nmae_std"] = records["mae_std"] / records["y_range"]

    # 4 base figures × 2 splits = 8 figures
    # Split A: Phase 1/3/4 only (Phase 3 without XGB)
    filter_A = ["Phase 1 (paper)", "Phase 3 (τ=0.05)", "Phase 4 (XGB+MACE)"]
    exclude_A = [("Phase 3 (τ=0.05)", "xgb")]
    # Split B: Phase 2/3/5 only (all models)
    filter_B = ["Phase 2 (τ=1e-10)", "Phase 3 (τ=0.05)", "Phase 5 (no filter)"]

    def render_set(suffix, phase_filter, exclude=None):
        figure_grouped_bars(records, d12, "mae_mean", "mae_std",
            f"F1: MAE — d1, d2 [{suffix}]",
            "MAE (kcal/mol)", COMP / f"F1_MAE_d12_{suffix}.png",
            subplots=True, ncol=3, ymin=0,
            phase_filter=phase_filter, exclude=exclude)
        figure_grouped_bars(records, other, "mae_mean", "mae_std",
            f"F2: MAE — 7 channels [{suffix}]",
            "MAE (kcal/mol)", COMP / f"F2_MAE_other_{suffix}.png",
            subplots=True, ncol=4, ymin=0,
            phase_filter=phase_filter, exclude=exclude)
        figure_grouped_bars(records, d12, "nmae_mean", "nmae_std",
            f"F3: NMAE — d1, d2 [{suffix}]",
            "NMAE = MAE / y-range", COMP / f"F3_NMAE_d12_{suffix}.png",
            subplots=True, ncol=3, ymin=0,
            phase_filter=phase_filter, exclude=exclude)
        figure_grouped_bars(records, other, "nmae_mean", "nmae_std",
            f"F4: NMAE — 7 channels [{suffix}]",
            "NMAE = MAE / y-range", COMP / f"F4_NMAE_other_{suffix}.png",
            subplots=True, ncol=4, ymin=0,
            phase_filter=phase_filter, exclude=exclude)

    render_set("P1_P3_P4", filter_A, exclude_A)
    render_set("P2_P3_P5", filter_B, None)
    # F5: R²
    figure_grouped_bars(records, d12 + other, "r2_mean", "r2_std",
                          "F5: R² per channel",
                          "R²", COMP / "F5_R2.png",
                          subplots=True, ncol=5)

    # REPORT.md
    with open(COMP / "REPORT.md", "w") as f:
        f.write("# Phase 1/2/3/4/5 Comparison (per-model bars)\n\n")
        f.write("Dataset: bath_1480, 3504 rxns  \n\n")
        f.write("- **Phase 1** = paper reproduction (Espley 46 features, Espley DFT labels)\n")
        f.write("- **Phase 2** = xTB-augmented, **VarianceThreshold τ=1e-10** (61 features)\n")
        f.write("- **Phase 3** = xTB-augmented, **VarianceThreshold τ=0.05** (46 features)\n")
        f.write("- **Phase 5** = xTB-augmented, **no filter** (63 features, paper's exact methodology)\n")
        f.write("- **Phase 4** = XGB + MACE residual, our own labels + EDA (independent architecture)\n\n")
        f.write("Phase 2/3/5 use identical HPs (from hps_phase23.pkl), same targets ")
        f.write("(our self-consistent d1_own/d2_own + 6 EDA channels), same 5 seeds ")
        f.write("(22, 23, 14, 1, 2). Only feature count differs by τ.\n\n")
        f.write("## Test MAE (mean ± std across 5 seeds) per channel × model × phase\n\n")
        for canon in d12 + other:
            sub = records[records["canonical"] == canon]
            if len(sub) == 0: continue
            f.write(f"### {canon}\n\n")
            f.write("| Phase | Model | MAE | NMAE |\n|---|---|---:|---:|\n")
            for _, r in sub.iterrows():
                f.write(f"| {r['phase']} | {r['model']} | "
                        f"{r['mae_mean']:.3f} ± {r['mae_std']:.3f} | "
                        f"{r['nmae_mean']:.3f} |\n")
            f.write("\n")

    print(f"Done. Output: {COMP}")


if __name__ == "__main__":
    main()
