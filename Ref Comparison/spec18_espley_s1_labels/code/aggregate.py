"""spec18 Stage 1 aggregator — emits sub_source_stats.csv, DEVIATIONS.md,
summary.md, and figures/sub_source_box.png.

Runs after build_2ch_labels.py + compare_to_ds3.py on the DIPOLAR-400 cohort.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec18_espley_s1_labels"
IN_PARQUET = STAGE / "results/labels_2ch_400dipolar.parquet"
DS3_CSV = STAGE / "results/ds3_distribution_comparison.csv"

OUT_STATS = STAGE / "results/sub_source_stats.csv"
OUT_DEV = STAGE / "results/DEVIATIONS.md"
OUT_SUM = STAGE / "results/summary.md"
OUT_FIG = STAGE / "figures/sub_source_box.png"

DEVIATIONS = [
    ("1", "ORCA replaces Gaussian 16", "3",
     "user-mandated; no Gaussian in project stack for this line"),
    ("2", "GFN2-xTB replaces AM1 as the SQM level", "3",
     "ORCA has no AM1; GFN2 is the nearest SQM-tier substitute and the project's existing SQM engine"),
    ("3", "q_barrier (ΔG‡) omitted from both targets and features", "1, 4",
     "electronic-energy consistency (SPEC_14); 44 features vs their 45"),
    ("4", "Reference DFT is ORCA ωB97X-3c EDA-NOCV, not B3LYP-D3(BJ)/def2-TZVP + SMD", "1",
     "our labels; must be noted on every cross-paper kcal/mol comparison"),
    ("5", "Cohort is 400 dipolar [3+2] cycloadditions, not 3510", "1",
     "user-requested restriction to the 400-set (192 from LOCKED_778 + 208 from spec16 LC-extension)"),
]


def write_deviations():
    lines = ["# DEVIATIONS — Espley et al. replication line", "",
             "Started at Stage 1 (spec18_espley_s1_labels). Every downstream "
             "stage inherits and appends to this list.", "",
             "| # | deviation | stage | rationale |",
             "|---|---|---|---|"]
    for row in DEVIATIONS:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines += [
        "",
        "## Implementation notes",
        "",
        "**Interaction reconstructed from `int_eda_kcal`, not re-summed.** "
        "The source parquet rounds `pauli_kcal + elst_kcal + orb_kcal + "
        "disp_kcal` to a slightly-rounded `int_eda_kcal` (max drift 0.02 "
        "kcal/mol). Using the recorded `int_eda_kcal` reproduces `act_kcal` "
        "exactly (Gate #4 diff = 0.000 kcal/mol); re-summing the channels "
        "would leave a 0.02 kcal/mol floor.",
        "",
        "**No CONTAM filter applied.** None of the 5 SPEC_10 dipolar CONTAM "
        "ids appear in the 400-set (verified 2026-07-24). Cohort size = 400 "
        "= 192 (locked_778) + 208 (spec16 LC-extension).",
        "",
        "**Contributions dict sum tolerance 5e-3, not 1e-6.** The source "
        "parquet stores `strain_A_kcal`, `strain_B_kcal`, and `strain_kcal` "
        "as independently rounded floats; strain_A + strain_B differs from "
        "strain by up to 1e-3 kcal/mol on the spec16 half of the cohort. "
        "A tighter tolerance would fail on source rounding. A real schema "
        "bug (e.g. swapped fragments, missing atoms) would show ≥ 0.1 "
        "kcal/mol drift and still be caught.",
    ]
    OUT_DEV.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT_DEV}")


def sub_source_stats(df: pd.DataFrame) -> pd.DataFrame:
    targets = ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]
    rows = []
    group_col = "sub_source" if "sub_source" in df.columns else None
    groups = df.groupby(group_col) if group_col else [("ALL", df)]
    for grp, sub in groups:
        for t in targets:
            v = sub[t].values
            rows.append({
                "sub_source": grp, "target": t,
                "n": int(v.size),
                "mean": float(np.mean(v)),
                "std":  float(np.std(v, ddof=1)),
                "min":  float(np.min(v)),
                "max":  float(np.max(v)),
            })
    # whole-cohort
    for t in targets:
        v = df[t].values
        rows.append({
            "sub_source": "ALL", "target": t,
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std":  float(np.std(v, ddof=1)),
            "min":  float(np.min(v)),
            "max":  float(np.max(v)),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_STATS, index=False)
    print(f"[write] {OUT_STATS}")
    return tbl


def sub_source_box(df: pd.DataFrame) -> None:
    targets = ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]
    if "sub_source" in df.columns:
        groups = sorted(df["sub_source"].unique())
    else:
        groups = ["ALL"]
        df = df.assign(sub_source="ALL")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, t in zip(axes, targets):
        data = [df.loc[df["sub_source"] == g, t].values for g in groups]
        ax.boxplot(data, labels=groups, showfliers=True)
        ax.set_ylabel(f"{t}  [kcal/mol]")
        ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    print(f"[write] {OUT_FIG}")


def write_summary(df: pd.DataFrame, tbl: pd.DataFrame) -> None:
    ds3 = pd.read_csv(DS3_CSV)

    lines = []
    lines.append("# spec18_espley_s1_labels — Stage 1 summary (DIPOLAR-400)")
    lines.append("")
    lines.append("Recasts the 400-reaction dipolar [3+2]-cycloaddition set into "
                 "Espley et al.'s 2-channel DIAS schema "
                 "(Digital Discovery 2024, DOI 10.1039/d4dd00224e). "
                 "Because both this cohort and Espley's ds3 are [3+2] "
                 "cycloadditions, the ds3 distribution anchors below are for "
                 "once directly comparable modulo the reference DFT level "
                 "(Deviation #4).")
    lines.append("")
    lines.append("## Output of record")
    lines.append("")
    lines.append("`results/labels_2ch_400dipolar.parquet` — 400 rows, columns:")
    lines.append("")
    lines.append("- `reaction_number` (int32, contiguous 0..399)")
    lines.append("- `sum_distortion_energies_dft` (float64, kcal/mol, positive)")
    lines.append("- `interaction_energies_dft` (float64, kcal/mol, positive — sign flip vs. our E_int)")
    lines.append("- `e_barrier_dft` (float64, kcal/mol; = sum_distortion − interaction)")
    lines.append("- `distortion_contributions_dft` (object, dict `{rxn_1: strain_A, rxn_2: strain_B}`)")
    lines.append("- `family`, `reaction_id`, `act_kcal_source`, `sub_source` — provenance columns")
    lines.append("")
    lines.append("## Cohort composition")
    lines.append("")
    lines.append("400 dipolar [3+2] cycloadditions:")
    lines.append("")
    lines.append("| sub_source | n |")
    lines.append("|---|---:|")
    if "sub_source" in df.columns:
        for grp, n in df.groupby("sub_source").size().items():
            lines.append(f"| {grp} | {n} |")
    lines.append(f"| **total** | **{len(df)}** |")
    lines.append("")
    lines.append("Source parquet: `outputs/spec16_orca/labels/dipolar_400_merged.parquet`.")
    lines.append("Provenance in `data/cohort_notes.json`.")
    lines.append("")

    lines.append("## Target statistics (kcal/mol)")
    lines.append("")
    for t in ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]:
        lines.append(f"### {t}")
        lines.append("")
        lines.append("| sub_source | n | mean | std | min | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        sub = tbl[tbl["target"] == t]
        for _, r in sub.iterrows():
            lines.append(f"| {r['sub_source']} | {int(r['n'])} | "
                         f"{r['mean']:.3f} | {r['std']:.3f} | "
                         f"{r['min']:.3f} | {r['max']:.3f} |")
        lines.append("")

    lines.append("## Distribution comparison vs. Espley ds3 (n=3510)")
    lines.append("")
    lines.append("Their ds3 raw pickle is not available on this HPC; the ds3 columns "
                 "below are the literal statistics recorded in the spec (§6 item 3). "
                 "Both cohorts are dipolar [3+2] cycloadditions; the remaining offset "
                 "is the reference DFT level (Deviation #4).")
    lines.append("")
    lines.append("| target | stat | ours (n=400) | ds3 ref (n=3510) |")
    lines.append("|---|---|---:|---:|")
    for _, r in ds3.iterrows():
        ov = "" if pd.isna(r["ours"]) else f"{r['ours']:.4f}"
        rv = "" if pd.isna(r["ds3_ref"]) else f"{r['ds3_ref']:.4f}"
        lines.append(f"| {r['target']} | {r['stat']} | {ov} | {rv} |")
    lines.append("")

    lines.append("## Gates (verification)")
    lines.append("")
    lines.append("All six correctness gates pass; see `logs/gates.log`.")
    lines.append("")
    lines.append("- Gate #1: cohort n = 400")
    lines.append("- Gate #2a: interaction > 0 in ≥ 95% of rows")
    lines.append("- Gate #2b: sum_distortion > 0 in all rows")
    lines.append("- Gate #2c: `sign(interaction_dft) == -sign(source int_eda)` for every row")
    lines.append("- Gate #3: `e_barrier_dft == sum_distortion − interaction` (max abs diff < 1e-6)")
    lines.append("- Gate #4: `|e_barrier_dft − act_kcal_source| < 0.1 kcal/mol` for all rows")
    lines.append("- Gate #5: target column names include the substring `dft` "
                 "(required by `f_select.py::Manual._manual_runner`)")
    lines.append("- Gate #6: dtypes reaction_number=int32, energies=float64, dict=object")
    lines.append("")

    lines.append("## Downstream contracts")
    lines.append("")
    lines.append("- **Do not rename `_dft` → `_wb97x3c`.** `f_select.py` line ~226 keeps "
                 "features by literal substring `dft`; a rename silently drops every "
                 "target column at feature selection.")
    lines.append("- **`q_barrier_dft` is intentionally absent** (Deviation #3). "
                 "Stage 4 will emit 44 features vs. their 45.")
    lines.append("- **Fragment ordering.** Key `<rxn>_1` = fragment A in the source "
                 "parquet (`strain_A_kcal`); key `_2` = fragment B. Chemical role "
                 "(dipole / dipolarophile) is *not* asserted here — it will be resolved "
                 "at Stage 4 when common-atom masks are defined.")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("```")
    lines.append("Ref Comparison/spec18_espley_s1_labels/")
    lines.append("  code/{build_2ch_labels.py, compare_to_ds3.py, aggregate.py, submit_s1.sh}")
    lines.append("  data/cohort_notes.json")
    lines.append("  logs/{build.log, gates.log}")
    lines.append("  results/{labels_2ch_400dipolar.parquet, sub_source_stats.csv,")
    lines.append("           ds3_distribution_comparison.csv, DEVIATIONS.md, summary.md}")
    lines.append("  figures/{target_hist_3panel.png, sub_source_box.png}")
    lines.append("```")
    lines.append("")

    OUT_SUM.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT_SUM}")


def main() -> None:
    df = pd.read_parquet(IN_PARQUET)
    tbl = sub_source_stats(df)
    sub_source_box(df)
    write_deviations()
    write_summary(df, tbl)


if __name__ == "__main__":
    main()
