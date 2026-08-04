#!/usr/bin/env python3
"""
Post-process ml_results.pkl → metrics tables + parity plots + REPORT.md.

Input:
  pipeline_work/ml_results.pkl   Grayson ml_analysis output

Outputs:
  results/metrics_per_seed.csv       one row per (model, target, seed)
  results/metrics_summary.csv        aggregated mean/std/min per (model, target)
  figures/mae_bar_by_target.png      bar chart: mean test MAE per (model, target)
  figures/parity_best_<target>.png   parity plot for best-model/best-seed per target
  REPORT.md                          human-readable summary + paper comparison
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "pipeline_work"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

MODELS = ["ridge", "krr", "svr", "rf"]
TARGETS = [
    "e_barrier_dft",
    "distortion_dft",
    "interaction_dft",
    "distortion_dipole_dft",
    "distortion_dipolarophile_dft",
]

# Paper Table 1 ds3 SVR test MAE (kcal/mol)
PAPER_DS3 = {
    "distortion_dipole_dft":       (3.59, 2.55),   # (pre-ML AM1-DFT, SVR test MAE)
    "distortion_dipolarophile_dft": (3.81, 2.37),
    "interaction_dft":              (20.01, 2.46),
}


def build_per_seed_df(ml_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ml_results.iterrows():
        mt = r["model_target"]
        model = mt.split("_", 1)[0]
        target = mt.split("_", 1)[1]
        for i, seed in enumerate(r["random_state"]):
            rows.append({
                "model": model,
                "target": target,
                "seed": seed,
                "test_mae": r["y_test_pred"][i],
                "val_mae": r["y_val_pred"][i],
                "std_error_test": r["std_error_test"][i],
            })
    return pd.DataFrame(rows)


def build_summary_df(per_seed: pd.DataFrame) -> pd.DataFrame:
    return (per_seed.groupby(["model", "target"])
            .agg(mean_test_mae=("test_mae", "mean"),
                 std_test_mae=("test_mae", "std"),
                 min_test_mae=("test_mae", "min"),
                 mean_val_mae=("val_mae", "mean"))
            .reset_index())


def bar_by_target(summary: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = np.arange(len(TARGETS))
    width = 0.2
    colors = {"ridge": "#4C72B0", "krr": "#DD8452", "svr": "#55A467", "rf": "#C44E52"}
    for i, m in enumerate(MODELS):
        vals, errs = [], []
        for t in TARGETS:
            row = summary[(summary["model"] == m) & (summary["target"] == t)]
            if len(row) == 1:
                vals.append(row["mean_test_mae"].iloc[0])
                errs.append(row["std_test_mae"].iloc[0])
            else:
                vals.append(np.nan); errs.append(0)
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3,
               label=m.upper(), color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_dft", "").replace("_", " ") for t in TARGETS], rotation=15, ha="right")
    ax.set_ylabel("Test MAE (kcal/mol) — mean ± std over 5 seeds")
    ax.set_title("paper_reproduction_203: model comparison per target")
    ax.axhline(1.0, ls="--", c="gray", alpha=0.5, label="1 kcal/mol threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parity_best_per_target(ml_results: pd.DataFrame, per_seed: pd.DataFrame, out_dir: Path):
    for target in TARGETS:
        sub = per_seed[per_seed["target"] == target]
        best = sub.loc[sub["test_mae"].idxmin()]
        model, seed = best["model"], best["seed"]
        mt = f"{model}_{target}"
        row = ml_results[ml_results["model_target"] == mt].iloc[0]
        seed_idx = list(row["random_state"]).index(seed)

        y_true = np.asarray(row["y_test_true"][seed_idx]).flatten()
        y_pred = np.asarray(row["y_test_pred_values"][seed_idx]).flatten()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(y_pred, y_true, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        lo = min(min(y_true), min(y_pred))
        hi = max(max(y_true), max(y_pred))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="perfect")
        # ±1 kcal band
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                        color='gray', alpha=0.2, label="±1 kcal/mol")
        ax.set_xlabel("Predicted (kcal/mol)")
        ax.set_ylabel("True DFT (kcal/mol)")
        ax.set_title(f"{model.upper()} on {target.replace('_dft', '')}\nseed={seed}, test MAE = {best['test_mae']:.2f} kcal/mol")
        ax.legend()
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        plt.savefig(out_dir / f"parity_best_{target}.png", dpi=150)
        plt.close()


def write_report(summary: pd.DataFrame, per_seed: pd.DataFrame, out_md: Path):
    lines = []
    lines.append("# paper_reproduction_203 — ML pipeline results\n")
    lines.append("Reproduction of Espley et al. 2024 (*Digital Discovery* **3**, 2479, DOI 10.1039/D4DD00224E) —")
    lines.append("SQM (AM1)-based ML prediction of DFT distortion + interaction energies.\n")
    lines.append("## Setup\n")
    lines.append("- **Dataset**: 203 / 400 dipolar cycloaddition reactions (snapshot 2026-08-04, 5/5 ORCA jobs complete)")
    lines.append("- **DFT target**: ωB97X-D3BJ/def2-TZVP + CPCM(water) SPE + EDA-NOCV (ORCA 6.1.1)")
    lines.append("- **AM1 features**: Gaussian 16 `#P AM1 Freq NoSymm` (gas phase), Mulliken + APT charges + Morfeus (BuriedVolume, SASA, Sterimol)")
    lines.append("- **5-atom labels**: user-manual (400/400 confirmed via port 5578 viz)")
    lines.append("- **Fragment A/B partition + type**: user-manual (400/400)")
    lines.append("- **Models**: Ridge, KRR (rbf), SVR (rbf), RandomForest — sklearn only (Keras 3 NN branch skipped)")
    lines.append("- **Splits**: 80/10/10 train/val/test per seed, 5 seeds `[23, 22, 14, 1, 2]`")
    lines.append("- **Features**: 155 after correlation + variance filter (from 309 raw)\n")

    lines.append("## Test MAE — mean ± std over 5 seeds (kcal/mol)\n")
    lines.append("| target | Ridge | KRR | SVR | RF | best |")
    lines.append("|---|---|---|---|---|---|")
    for t in TARGETS:
        cells = [t.replace("_dft", "")]
        best_mae, best_model = float("inf"), ""
        for m in MODELS:
            row = summary[(summary["model"] == m) & (summary["target"] == t)]
            if len(row) == 1:
                mean = row["mean_test_mae"].iloc[0]
                std = row["std_test_mae"].iloc[0]
                cells.append(f"{mean:.2f} ± {std:.2f}")
                if mean < best_mae:
                    best_mae, best_model = mean, m
            else:
                cells.append("—")
        cells.append(f"**{best_model.upper()}: {best_mae:.2f}**")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Comparison with paper Table 1 (ds3, N=730)\n")
    lines.append("| target | our best (5-seed mean) | paper ds3 SVR | paper pre-ML AM1-DFT |")
    lines.append("|---|---|---|---|")
    for t in TARGETS:
        our_best = summary[summary["target"] == t]["mean_test_mae"].min()
        if t in PAPER_DS3:
            pre_ml, svr = PAPER_DS3[t]
            lines.append(f"| {t.replace('_dft','')} | {our_best:.2f} | {svr:.2f} | {pre_ml:.2f} |")
        else:
            lines.append(f"| {t.replace('_dft','')} | {our_best:.2f} | (not in paper) | — |")

    lines.append("\n## Best model per target — parity plots\n")
    for t in TARGETS:
        sub = per_seed[per_seed["target"] == t]
        best = sub.loc[sub["test_mae"].idxmin()]
        lines.append(f"- **{t.replace('_dft','')}** — {best['model'].upper()} (seed {best['seed']}, test MAE {best['test_mae']:.2f}): [parity_best_{t}.png](figures/parity_best_{t}.png)")

    lines.append("\n## Pre-ML AM1-DFT MAE (before ML correction)\n")
    lines.append("From `code/60_import_am1_results.py`:")
    lines.append("- distortion:               4.91 kcal/mol")
    lines.append("- interaction:              40.43 kcal/mol")
    lines.append("- e_barrier:                41.90 kcal/mol")
    lines.append("- distortion_dipole:         5.46 kcal/mol")
    lines.append("- distortion_dipolarophile:  5.71 kcal/mol\n")
    lines.append("ML predictions reduce AM1's systematic bias — especially large for interaction "
                 "(where paper reported pre-ML MAE 20 kcal/mol; ours is 40 kcal/mol, larger because "
                 "our AM1 was gas phase while paper used IEFPCM water — bias-correctable by ML).\n")

    lines.append("## Reproducibility\n")
    lines.append("Full pipeline: `code/70_run_grayson_pipeline.sh` (sbatch, 48h walltime).")
    lines.append("All intermediate + final files under `pipeline_work/`.")

    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    ml_results = pd.read_pickle(WORK / "ml_results.pkl")
    print(f"loaded ml_results: {ml_results.shape}")

    per_seed = build_per_seed_df(ml_results)
    per_seed.to_csv(RESULTS / "metrics_per_seed.csv", index=False)
    print(f"wrote metrics_per_seed.csv: {per_seed.shape}")

    summary = build_summary_df(per_seed)
    summary.to_csv(RESULTS / "metrics_summary.csv", index=False)
    print(f"wrote metrics_summary.csv:\n{summary.to_string(index=False)}")

    bar_by_target(summary, FIGS / "mae_bar_by_target.png")
    print(f"wrote {FIGS / 'mae_bar_by_target.png'}")

    parity_best_per_target(ml_results, per_seed, FIGS)
    print(f"wrote parity plots × {len(TARGETS)}")

    write_report(summary, per_seed, ROOT / "REPORT.md")
    print(f"wrote {ROOT / 'REPORT.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
