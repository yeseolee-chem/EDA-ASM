#!/usr/bin/env python3
"""v3 aggregation for oracle-clean v2 rerun.

Changes vs 07_aggregate_v2.py:
  1. Reads from phase{2,3,5}_paper_v2/ and phase4_stacked_xgb_mace_v2/.
  2. NMAE normalization: **meanAD** (mean absolute deviation) of the label across
     all rxns (v9 house standard; not per-seed test range/iloc[0]). Constant
     across configs → fair.
  3. disp channel marked as ORACLE (analytical D4-derived), excluded from
     best-model ranking narrative but still reported in tables/figures.
  4. Same 8-figure split (P1_P3_P4, P2_P3_P5) as v2.

Methodology notes (see README):
  - HPs (`hps_phase23_v2.pkl`) are aliased from paper Arm A tuning (46
    Espley features). No per-arm re-tuning against v2 60-feature space.
    Sklearn arms are therefore slightly HP-underfit for their actual
    feature set; XGB advantage is conservative (XGB uses fixed defaults
    + early-stopping-on-val).
"""
from pathlib import Path
import json
import os
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
RESULTS = EXP / "results"
COMP = RESULTS / "comparison_v3"
COMP.mkdir(parents=True, exist_ok=True)

PHASE_COLORS = {
    "Phase 1 (paper)":            "#7f7f7f",
    "Phase 2 (τ=1e-10)":          "#4c9fd8",
    "Phase 3 (τ=0.05)":           "#3d9d9b",
    "Phase 4 (XGB+MACE)":         "#e07a5f",
    "Phase 5 (no filter)":        "#9b59b6",
}
MODEL_ORDER = ["ridge", "krr", "svr", "rf", "xgb"]

PHASE1_TARGETS = {
    "d1":          "distortion_energy_1_dft",
    "d2":          "distortion_energy_2_dft",
    "interaction": "interaction_energies_dft",
}
COMMON_OWN = {
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
PHASE4_TARGETS = {
    "d1": "d1_own", "d2": "d2_own",
    "pauli": "pauli_dft", "oi": "oi_dft", "elst": "elst_dft",
    "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft",
}
ORACLE_CHANNELS = {"disp"}  # D4 dispersion is analytical from geometry — declared oracle


def compute_label_mad(dataset_pkl, target_col):
    """meanAD of a target across the whole cohort (v9 house standard)."""
    df = pd.read_pickle(dataset_pkl)
    if target_col not in df.columns:
        return None
    v = df[target_col].values.astype(float)
    return float(np.mean(np.abs(v - v.mean())))


def extract_paper_pkl(pkl_dir):
    rows = []
    if not pkl_dir.exists(): return pd.DataFrame(rows)
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
            if mt.startswith("2_st_nn_"):
                model = "2L_NN"; target = mt[len("2_st_nn_"):]
            elif mt.startswith("4_st_nn_"):
                model = "4L_NN"; target = mt[len("4_st_nn_"):]
            else:
                model = mt.split("_")[0]; target = "_".join(mt.split("_")[1:])
            mae = np.mean(np.abs(yt - yp))
            ss_res = np.sum((yt - yp)**2); ss_tot = np.sum((yt - yt.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rows.append({"seed": seed, "model": model, "target": target,
                          "mae": mae, "r2": r2})
    return pd.DataFrame(rows)


def extract_xgb(pkl_dir):
    rows = []
    if not pkl_dir.exists(): return pd.DataFrame(rows)
    for sdir in sorted(pkl_dir.glob("seed_*")):
        try: seed = int(sdir.name.split("_")[1])
        except: continue
        pkl = sdir / "xgb_results.pkl"
        if not pkl.exists(): continue
        d = pickle.load(open(pkl, "rb"))
        for target, res in d.items():
            rows.append({"seed": seed, "model": "xgb", "target": target,
                          "mae": res["test_mae"], "r2": res["test_r2"]})
    return pd.DataFrame(rows)


def extract_phase4(root):
    rows = []
    if not root.exists(): return pd.DataFrame(rows)
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
                ss_res = np.sum((yt - yp)**2); ss_tot = np.sum((yt - yt.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rows.append({"seed": seed, "fold": fold, "model": "XGB+MACE",
                              "target": tgt, "mae": mae, "r2": r2})
    return pd.DataFrame(rows)


def build_records(p1, p2, p3, p5, p4, mad_lookup):
    rows = []

    def add_phase(phase, src, mapping, models, mad_ds_key):
        for canon, target in mapping.items():
            mad = mad_lookup.get((mad_ds_key, target))
            for m in models:
                sub = src[(src["target"] == target) & (src["model"] == m)]
                if len(sub) == 0: continue
                mae_mean = sub["mae"].mean()
                rows.append({
                    "phase": phase, "canonical": canon, "model": m,
                    "target_used": target,
                    "mae_mean": mae_mean, "mae_std": sub["mae"].std(),
                    "r2_mean": sub["r2"].mean(), "r2_std": sub["r2"].std(),
                    "nmae_mean": mae_mean / mad if mad else np.nan,
                    "nmae_std": sub["mae"].std() / mad if mad else np.nan,
                    "mad": mad, "n_seeds": len(sub), "oracle": canon in ORACLE_CHANNELS,
                })

    add_phase("Phase 1 (paper)",         p1, PHASE1_TARGETS,      MODEL_ORDER + ["XGB+MACE"], "phase1")
    add_phase("Phase 2 (τ=1e-10)",       p2, COMMON_OWN,           MODEL_ORDER, "phase2")
    add_phase("Phase 3 (τ=0.05)",        p3, COMMON_OWN,           MODEL_ORDER, "phase3")
    add_phase("Phase 5 (no filter)",     p5, COMMON_OWN,           MODEL_ORDER, "phase5")
    add_phase("Phase 4 (XGB+MACE)",      p4, PHASE4_TARGETS,       ["XGB+MACE"], "phase4")
    return pd.DataFrame(rows)


def figure_grouped_bars(records, canonicals, metric_col, err_col, title, ylabel, out_path,
                          subplots=False, ncol=4, ymin=None,
                          phase_filter=None, exclude=None):
    if phase_filter is not None:
        records = records[records["phase"].isin(phase_filter)].copy()
    if exclude:
        for ph, mo in exclude:
            records = records[~((records["phase"] == ph) & (records["model"] == mo))]

    phases = [p for p in PHASE_COLORS.keys() if phase_filter is None or p in phase_filter]

    if subplots:
        n = len(canonicals)
        nrow = int(np.ceil((n + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6*ncol, 3.8*nrow))
        axes_flat = np.array(axes).flatten()
    else:
        fig, ax = plt.subplots(figsize=(max(7, 1.6*len(canonicals)+3), 5.0))

    for i, canon in enumerate(canonicals):
        target_ax = axes_flat[i] if subplots else ax
        bar_positions, bar_heights, bar_errs, bar_colors, bar_labels = [], [], [], [], []
        cursor = 0.0
        oracle = canon in ORACLE_CHANNELS
        for phase in phases:
            phase_rows = records[(records["canonical"] == canon) & (records["phase"] == phase)]
            if len(phase_rows) == 0: continue
            models_here = [m for m in MODEL_ORDER + ["XGB+MACE"] if m in phase_rows["model"].unique()]
            for m in models_here:
                sub = phase_rows[phase_rows["model"] == m]
                if len(sub) == 0: continue
                bar_positions.append(cursor)
                bar_heights.append(sub[metric_col].iloc[0])
                bar_errs.append(sub[err_col].iloc[0] if err_col in sub.columns else 0)
                bar_colors.append(PHASE_COLORS[phase])
                bar_labels.append(m if m != "XGB+MACE" else "XGB")
                cursor += 1.0
            cursor += 0.5

        if bar_positions:
            target_ax.bar(bar_positions, bar_heights, width=0.8,
                          yerr=bar_errs, color=bar_colors, capsize=2.5,
                          edgecolor="black", linewidth=0.4)
            target_ax.set_xticks(bar_positions)
            target_ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
            title_txt = canon + (" ⚠ oracle" if oracle else "")
            target_ax.set_title(title_txt, fontsize=10)
            target_ax.set_ylabel(ylabel, fontsize=9)
            target_ax.grid(axis="y", alpha=0.25)
            if ymin is not None: target_ax.set_ylim(bottom=ymin)

    if subplots:
        legend_ax = axes_flat[len(canonicals)]
        legend_ax.axis("off")
        handles = [plt.Rectangle((0,0), 1, 1, color=PHASE_COLORS[p]) for p in phases]
        legend_labels = [re.sub(r"^Phase \d+ \((.*)\)$", r"\1", p) for p in phases]
        legend_ax.legend(handles, legend_labels, loc="center", frameon=True, fontsize=11)
        for k in range(len(canonicals)+1, len(axes_flat)):
            axes_flat[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    print("v3 aggregation — oracle-clean + meanAD-normalized NMAE")
    # Label MAD lookup (canonical MAD per dataset+target)
    ds = {"phase1": EXP/"cohort_v1/features_espley_v1.pkl",  # not really — Phase 1 uses labels_v1
          "phase2": EXP/"cohort_v1/phase2_dataset_v2.pkl",
          "phase3": EXP/"cohort_v1/phase3_dataset_v2.pkl",
          "phase5": EXP/"cohort_v1/phase5_dataset_v2.pkl",
          "phase4": EXP/"cohort_v1/labels_v1.pkl"}
    # For Phase 1 label MAD, use labels_v1 (has distortion_energy_1/2_dft via merge with orig)
    # Simpler: derive MAD once per canonical from phase5_dataset_v2 (all label cols there)
    p5_ds = pd.read_pickle(EXP/"cohort_v1/phase5_dataset_v2.pkl")
    labels_v1 = pd.read_pickle(EXP/"cohort_v1/labels_v1.pkl")
    def mad(series):
        # v9 house standard: MEAN absolute deviation (not median-AD).
        v = np.asarray(series, dtype=float)
        return float(np.mean(np.abs(v - v.mean())))
    mad_lookup = {}
    for name, tgt in [("phase2","d1_own_dft"),("phase2","d2_own_dft"),("phase2","interaction_own_dft"),
                       ("phase3","d1_own_dft"),("phase3","d2_own_dft"),("phase3","interaction_own_dft"),
                       ("phase5","d1_own_dft"),("phase5","d2_own_dft"),("phase5","interaction_own_dft")]:
        if tgt in p5_ds.columns: mad_lookup[(name, tgt)] = mad(p5_ds[tgt])
    for name in ["phase2","phase3","phase5"]:
        for c in ["pauli_dft","oi_dft","elst_dft","disp_dft","cpcm_dft","cds_dft"]:
            if c in p5_ds.columns: mad_lookup[(name, c)] = mad(p5_ds[c])
    for c in ["distortion_energy_1_dft","distortion_energy_2_dft","interaction_energies_dft"]:
        if c in p5_ds.columns: mad_lookup[("phase1", c)] = mad(p5_ds[c])
    # Phase 4 labels (d1_own without _dft suffix — from labels_v1)
    for c_own, tgt_col in [("d1_own","d1_own"),("d2_own","d2_own")]:
        if tgt_col in labels_v1.columns: mad_lookup[("phase4", c_own)] = mad(labels_v1[tgt_col])
    for c in ["pauli_dft","oi_dft","elst_dft","disp_dft","cpcm_dft","cds_dft"]:
        if c in p5_ds.columns: mad_lookup[("phase4", c)] = mad(p5_ds[c])

    # Load results
    p1  = extract_paper_pkl(RESULTS / "phase1_espley_paper_reproduction")
    p2  = pd.concat([extract_paper_pkl(RESULTS / "phase2_paper_v2"),
                       extract_xgb(RESULTS / "phase2_paper_v2")], ignore_index=True)
    p3  = pd.concat([extract_paper_pkl(RESULTS / "phase3_paper_v2"),
                       extract_xgb(RESULTS / "phase3_paper_v2")], ignore_index=True)
    p5  = pd.concat([extract_paper_pkl(RESULTS / "phase5_paper_v2"),
                       extract_xgb(RESULTS / "phase5_paper_v2")], ignore_index=True)
    p4  = extract_phase4(RESULTS / "phase4_stacked_xgb_mace_v2")
    for name, df in [("Phase 1", p1), ("Phase 2", p2), ("Phase 3", p3),
                       ("Phase 4", p4), ("Phase 5", p5)]:
        print(f"  {name}: {len(df)} rows")

    records = build_records(p1, p2, p3, p5, p4, mad_lookup)
    records.to_csv(COMP / "records_all.csv", index=False)
    print(f"records_all.csv: {len(records)} rows")

    d12 = ["d1", "d2"]
    other = ["interaction", "pauli", "oi", "elst", "disp", "cpcm", "cds"]

    filter_A = ["Phase 1 (paper)", "Phase 3 (τ=0.05)", "Phase 4 (XGB+MACE)"]
    exclude_A = [("Phase 3 (τ=0.05)", "xgb")]
    filter_B = ["Phase 2 (τ=1e-10)", "Phase 3 (τ=0.05)", "Phase 5 (no filter)"]

    def render_set(suffix, phase_filter, exclude=None):
        for target_list, ncol, name_prefix, ylabel_mae, ylabel_nmae, ylabel_r2 in [
            (d12, 3, "F1_MAE_d12", "MAE (kcal/mol)", "NMAE (÷meanAD)", "R²"),
            (other, 4, "F2_MAE_other", "MAE (kcal/mol)", "NMAE (÷meanAD)", "R²"),
        ]:
            figure_grouped_bars(records, target_list, "mae_mean", "mae_std",
                f"MAE — {'d1,d2' if target_list==d12 else '7 channels'} [{suffix}]",
                ylabel_mae,
                COMP / f"{name_prefix}_{suffix}.png",
                subplots=True, ncol=ncol, ymin=0,
                phase_filter=phase_filter, exclude=exclude)
        for target_list, ncol, name_prefix in [
            (d12, 3, "F3_NMAE_d12"),
            (other, 4, "F4_NMAE_other"),
        ]:
            figure_grouped_bars(records, target_list, "nmae_mean", "nmae_std",
                f"NMAE (÷meanAD) — {'d1,d2' if target_list==d12 else '7 channels'} [{suffix}]",
                "NMAE = MAE / meanAD",
                COMP / f"{name_prefix}_{suffix}.png",
                subplots=True, ncol=ncol, ymin=0,
                phase_filter=phase_filter, exclude=exclude)

    only = os.environ.get("RENDER_ONLY", "").strip()
    if not only or only == "all":
        render_sets = ["P1_P3_P4", "P2_P3_P5"]
    else:
        render_sets = [s.strip() for s in only.split(",")]
    if "P1_P3_P4" in render_sets:
        render_set("P1_P3_P4", filter_A, exclude_A)
    if "P2_P3_P5" in render_sets:
        render_set("P2_P3_P5", filter_B, None)

    # F5 R² (single full)
    figure_grouped_bars(records, d12 + other, "r2_mean", "r2_std",
        "F5: R² per channel", "R²", COMP / "F5_R2.png",
        subplots=True, ncol=5)

    # REPORT.md
    with open(COMP / "REPORT.md", "w") as f:
        f.write("# Phase 1/2/3/4/5 v3 (oracle-clean + MAD-normalized)\n\n")
        f.write("Dataset: bath_1480, 3504 rxns  \n\n")
        f.write("## Methodology notes\n")
        f.write("- HPs (`hps_phase23_v2.pkl`) are **aliased from paper Arm A tuning** "
                "(46 Espley features). No per-arm re-tuning against v2 60-feature space; "
                "sklearn arms are therefore slightly HP-underfit for their actual feature "
                "set. XGB uses fixed defaults + early-stopping-on-val, so any XGB advantage "
                "reported here is conservative under this HP handicap.\n")


        f.write("## Fixes applied\n")
        f.write("- Removed oracle features `disp_xtb`, `dispd4_xtb`, `eint_total_xtb` from all datasets.\n")
        f.write("- Fixed strain frag index swap: `strain_1_xtb ↔ strain_2_xtb`.\n")
        f.write("- NMAE = MAE / meanAD (v9 house standard; label meanAD across cohort, seed-invariant).\n")
        f.write("- `disp` channel declared **ORACLE** (analytical D4 dispersion) — reported but "
                "excluded from best-model ranking.\n\n")
        f.write("## Phase definitions\n")
        f.write("- Phase 1: paper reproduction (Espley features + labels, unchanged)\n")
        f.write("- Phase 2: xTB-aug, τ=1e-10, our own labels + 6 EDA channels\n")
        f.write("- Phase 3: xTB-aug, τ=0.05\n")
        f.write("- Phase 5: xTB-aug, no filter (paper methodology)\n")
        f.write("- Phase 4: XGB + MACE residual (5-fold CV, different protocol)\n\n")
        f.write("## Test MAE per channel × model × phase\n\n")
        for canon in d12 + other:
            oracle_tag = " ⚠ ORACLE" if canon in ORACLE_CHANNELS else ""
            sub = records[records["canonical"] == canon]
            if len(sub) == 0: continue
            f.write(f"### {canon}{oracle_tag}\n\n")
            f.write("| Phase | Model | MAE | NMAE (÷meanAD) |\n|---|---|---:|---:|\n")
            for _, r in sub.iterrows():
                f.write(f"| {r['phase']} | {r['model']} | "
                        f"{r['mae_mean']:.3f} ± {r['mae_std']:.3f} | "
                        f"{r['nmae_mean']:.3f} |\n")
            f.write("\n")

    print(f"Done. Output: {COMP}")


if __name__ == "__main__":
    main()
