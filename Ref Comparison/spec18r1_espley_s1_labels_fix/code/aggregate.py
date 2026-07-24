"""spec18r1 Stage 1 aggregator — summary.md, DEVIATIONS.md (with #6),
sub_source_stats.csv, ds3_distribution_comparison.csv, figures.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix"
PKL = STAGE / "results/labels_2ch_400dipolar.pkl"
GATES_LOG = STAGE / "logs/gates.log"

OUT_STATS = STAGE / "results/sub_source_stats.csv"
OUT_DS3 = STAGE / "results/ds3_distribution_comparison.csv"
OUT_DEV = STAGE / "results/DEVIATIONS.md"
OUT_SUM = STAGE / "results/summary.md"
FIG_HIST = STAGE / "figures/target_hist_3panel.png"
FIG_BOX = STAGE / "figures/sub_source_box.png"

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
     "user-requested restriction to the 400-set (192 locked_778 + 208 spec16)"),
    ("6", "f_select.py line ~226 substring `am1` → `am1|gfn2` (one-line patch required)", "3",
     "Rev 1 G-E confirmed `_gfn2` targets are dropped by the unmodified _manual_runner; either patch the substring or rename `_gfn2` → `_am1` at Stage 3"),
]

DS3 = {
    "e_barrier_dft":               {"n": 3510, "mean": 5.92,  "std": 8.49,  "min": -14.81, "max": 44.65},
    "sum_distortion_energies_dft": {"n": 3510, "mean": 27.13, "std": None,  "min":   2.47, "max": 79.15},
    "interaction_energies_dft":    {"n": 3510, "mean": 21.21, "std": None,  "min":   5.81, "max": 49.36},
}


def write_deviations() -> None:
    lines = [
        "# DEVIATIONS — Espley et al. replication line", "",
        "Started at Stage 1 (spec18r1_espley_s1_labels_fix). Every downstream "
        "stage inherits and appends to this list.", "",
        "| # | deviation | stage | rationale |",
        "|---|---|---|---|",
    ]
    for row in DEVIATIONS:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines += [
        "",
        "## Implementation notes",
        "",
        "**Artifact of record is `.pkl`, not `.parquet`.** pyarrow unions "
        "per-row dict keys into a single struct schema and drops keys not "
        "present on every row. With `distortion_contributions_dft` keys of "
        "the form `{n_1, n_2}` differing per row, parquet expands into an "
        "800-field struct populated with ~319,200 `None`s (Rev 0's failure "
        "mode, F1). Pickle preserves the dicts exactly. An inspection "
        "parquet is also written with the dict column DROPPED to prevent "
        "confusion.",
        "",
        "**Sign convention.** `interaction_energies_dft = −int_eda_kcal` "
        "(paper stores positive, our EDA E_int is negative and stabilising). "
        "Independence verified in G-C by re-summing the four channels from "
        "the source parquet.",
        "",
        "**ASM identity residual = source rounding floor.** G-D reports "
        "`|strain + resum(channels) − act_kcal|` per row. `int_eda_kcal` in "
        "the source is a 2-decimal round of the channel sum, so this "
        "residual sits at ~0.02 kcal/mol; larger residuals surface real "
        "label-pipeline drift.",
        "",
        "**Reaction_number is order-stable.** `reaction_number` is assigned "
        "after `sort_values('reaction_id', kind='mergesort')` (fixes F7). "
        "G-F confirms sorting the artifact by reaction_id reproduces "
        "`reaction_number == 0..399`.",
        "",
        "**Contributions dict sum tolerance 5e-3, not 1e-6.** Source-parquet "
        "rounding of `strain_A_kcal`, `strain_B_kcal`, `strain_kcal` "
        "produces up to 1e-3 kcal/mol drift on the 208 spec16 rows. A real "
        "schema bug would show ≥ 0.1 kcal/mol and still be caught.",
        "",
        "**No CONTAM filter applied.** None of the 5 SPEC_10 dipolar CONTAM "
        "ids appear in the 400-set (verified 2026-07-24).",
    ]
    OUT_DEV.write_text("\n".join(lines) + "\n")


def sub_source_stats(df: pd.DataFrame) -> pd.DataFrame:
    targets = ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]
    rows = []
    for grp, sub in df.groupby("sub_source"):
        for t in targets:
            v = sub[t].values
            rows.append({"sub_source": grp, "target": t,
                         "n": int(v.size),
                         "mean": float(v.mean()),
                         "std":  float(v.std(ddof=1)),
                         "min":  float(v.min()),
                         "max":  float(v.max())})
    for t in targets:
        v = df[t].values
        rows.append({"sub_source": "ALL", "target": t,
                     "n": int(v.size),
                     "mean": float(v.mean()),
                     "std":  float(v.std(ddof=1)),
                     "min":  float(v.min()),
                     "max":  float(v.max())})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_STATS, index=False)
    return tbl


def ds3_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]:
        v = df[t].values
        ours = {"n": int(v.size), "mean": float(v.mean()),
                "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}
        theirs = DS3[t]
        for stat in ["n", "mean", "std", "min", "max"]:
            rows.append({"target": t, "stat": stat,
                         "ours": ours[stat], "ds3_ref": theirs[stat]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_DS3, index=False)
    return tbl


def hist_3panel(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, t in zip(axes, ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]):
        v = df[t].values
        ax.hist(v, bins=40, color="#3b7dbf", edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.axvline(0, color="k", linewidth=0.6, alpha=0.5)
        ax.axvline(float(v.mean()), color="#e07b00", linewidth=1.4,
                   label=f"ours mean = {float(v.mean()):.2f}")
        ax.axvline(DS3[t]["mean"], color="#666", linewidth=1.0, linestyle="--",
                   label=f"ds3 mean = {DS3[t]['mean']:.2f}")
        ax.set_xlabel(f"{t}  [kcal/mol]")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_HIST, dpi=140)


def sub_source_box(df: pd.DataFrame) -> None:
    targets = ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]
    groups = sorted(df["sub_source"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, t in zip(axes, targets):
        ax.boxplot([df.loc[df["sub_source"] == g, t].values for g in groups],
                   labels=groups, showfliers=True)
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(f"{t}  [kcal/mol]")
    fig.tight_layout()
    fig.savefig(FIG_BOX, dpi=140)


def summarise_gates() -> tuple[int, int, str]:
    text = GATES_LOG.read_text() if GATES_LOG.exists() else ""
    n_fail = text.count("FAIL ")
    n_warn = text.count("WARN ")
    # Trim to a compact digest
    lines = [ln for ln in text.splitlines() if any(tag in ln for tag in
             ("PASS", "FAIL", "WARN", "INFO", "SUMMARY"))]
    return n_fail, n_warn, "\n".join(f"    {ln}" for ln in lines)


def write_summary(df: pd.DataFrame, stats: pd.DataFrame, ds3_tbl: pd.DataFrame) -> None:
    def _safe_len(p: Path) -> int:
        if not p.exists() or p.stat().st_size == 0:
            return 0
        try:
            return len(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            return 0
    outlier_csv = STAGE / "results/asm_residual_outliers.csv"
    triage_csv = STAGE / "results/anomaly_triage.csv"
    n_outliers = _safe_len(outlier_csv)
    n_flagged = _safe_len(triage_csv)
    n_fail, n_warn, gate_digest = summarise_gates()

    cohort_json = json.loads((STAGE / "data/cohort_notes.json").read_text())

    lines = []
    lines.append("# spec18r1_espley_s1_labels_fix — Stage 1 summary (DIPOLAR-400)")
    lines.append("")
    lines.append("Revision 1 of `spec18_espley_s1_labels`. Fixes Rev 0's parquet "
                 "serialisation bug (F1), replaces three tautological gates with "
                 "real ones (F3–F5), and re-arms the ASM identity tripwire on the "
                 "four-channel sum (F5).")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- python: {platform.python_version()}")
    lines.append(f"- pandas: {pd.__version__}")
    lines.append(f"- numpy:  {np.__version__}")
    lines.append("")
    lines.append("Espley's own pickles were written with `pandas==2.1.1`; if the "
                 "downstream env pins that version, this artifact will need to be "
                 "re-pickled from a matching env (see `logs/build.log` for the "
                 "version stamp at build time).")
    lines.append("")

    lines.append("## Artifact of record")
    lines.append("")
    lines.append(f"`results/{PKL.name}` (pickle; size {PKL.stat().st_size} bytes)")
    lines.append("")
    lines.append("Inspection parquet with the dict column dropped:")
    insp = STAGE / "results/labels_2ch_400dipolar.INSPECTION_ONLY.parquet"
    if insp.exists():
        lines.append(f"`results/{insp.name}` (size {insp.stat().st_size} bytes) — "
                     "**NOT the artifact of record.** Do not pass to their scripts.")
    lines.append("")

    lines.append("## Cohort composition")
    lines.append("")
    lines.append("| sub_source | n |")
    lines.append("|---|---:|")
    for grp, n in df.groupby("sub_source").size().items():
        lines.append(f"| {grp} | {n} |")
    lines.append(f"| **total** | **{len(df)}** |")
    lines.append("")
    lines.append(f"Source parquet: `{cohort_json['source_parquet']}`.")
    lines.append("")

    lines.append("## Gates (all run on the reloaded pickle)")
    lines.append("")
    lines.append(f"- **{n_fail} FAIL**, **{n_warn} WARN** — see `logs/gates.log`.")
    lines.append("")
    lines.append("- G-A round-trip fidelity — all 400 rows carry a 2-key dict, no NaN, "
                 "keys match `^\\d+_[12]$`, per-row prefix equals the row's "
                 "`reaction_number`, `sum(dict) − sum_distortion` within 5e-3 kcal/mol.")
    lines.append("- G-B file sanity — artifact suffix `.pkl`; inspection parquet has "
                 "no `distortion_contributions_dft` column.")
    lines.append("- G-C independent interaction — `|interaction_dft + resum(channels)|` "
                 "from the four **source** channels; independent from the "
                 "identity used to build the column.")
    lines.append("- G-D ASM identity residual — `|strain + resum − act_kcal|`. Rev 0 "
                 "substituted `int_eda_kcal` for `resum`, which reduced this to 0 "
                 "by construction. Rev 1 re-derives from the four channels.")
    lines.append("- G-E end-to-end contract — imports Espley's own "
                 "`General._clean_dist_contr` and `Manual._manual_runner` and runs "
                 "them on the reloaded artifact.")
    lines.append("- G-F reaction_number stability — sorting by `reaction_id` "
                 "reproduces `reaction_number == 0..399`; two rebuilds hash-compared.")
    lines.append("- G-G anomaly triage — count-and-cross-tab flagged rows against "
                 "`sub_source` and G-D residual. **No rows excluded.**")
    lines.append("- Regression guards (labelled) — the Rev 0 sign / algebraic-identity "
                 "checks are kept as future-edit guards. They are not evidence.")
    lines.append("")
    lines.append("### Gate digest")
    lines.append("```")
    lines.append(gate_digest)
    lines.append("```")
    lines.append("")

    lines.append("## G-G anomaly triage")
    lines.append("")
    lines.append(f"- {n_flagged} rows carry at least one anomaly flag "
                 f"(see `results/anomaly_triage.csv`).")
    lines.append(f"- {n_outliers} rows exceed the G-D 0.1 kcal/mol residual "
                 f"(see `results/asm_residual_outliers.csv`).")
    lines.append("- **No exclusions applied.** Any decision on removing rows "
                 "belongs to the user.")
    if n_outliers > 0:
        lines.append("")
        lines.append("### ⚠ UNSCREENED ASM RESIDUALS")
        lines.append("")
        lines.append(f"{n_outliers} row(s) have `|strain + resum − act_kcal| > 0.1 "
                     f"kcal/mol`. The full list is in "
                     f"`results/asm_residual_outliers.csv`. If any concentrate in "
                     f"the `spec16` sub-source, that half of the cohort was never "
                     f"CONTAM-screened.")
    lines.append("")

    lines.append("## Target statistics (kcal/mol)")
    lines.append("")
    for t in ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]:
        lines.append(f"### {t}")
        lines.append("")
        lines.append("| sub_source | n | mean | std | min | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        sub = stats[stats["target"] == t]
        for _, r in sub.iterrows():
            lines.append(f"| {r['sub_source']} | {int(r['n'])} | "
                         f"{r['mean']:.3f} | {r['std']:.3f} | "
                         f"{r['min']:.3f} | {r['max']:.3f} |")
        lines.append("")

    lines.append("## Distribution vs. Espley ds3 (n=3510)")
    lines.append("")
    lines.append("Both cohorts are dipolar [3+2] cycloadditions. Systematic ~1.6–1.9× "
                 "scale in `sum_distortion` and `interaction` while `e_barrier` is "
                 "similar in magnitude — the two large terms cancel. Candidate "
                 "explanations (not yet distinguished): reference DFT level "
                 "(Deviation #4), broader coverage of distorted TSs, or pathological "
                 "rows. **G-D separates 'pathological rows' from the other two.** "
                 "Do not conclude — surface.")
    lines.append("")
    lines.append("| target | stat | ours (n=400) | ds3 ref (n=3510) |")
    lines.append("|---|---|---:|---:|")
    for _, r in ds3_tbl.iterrows():
        ov = "" if pd.isna(r["ours"]) else f"{r['ours']:.4f}"
        rv = "" if pd.isna(r["ds3_ref"]) else f"{r['ds3_ref']:.4f}"
        lines.append(f"| {r['target']} | {r['stat']} | {ov} | {rv} |")
    lines.append("")

    lines.append("## Downstream contracts")
    lines.append("")
    lines.append("- **Do not rename `_dft` → `_wb97x3c`.** `f_select.py` line ~226 "
                 "selects by the literal substring `dft`.")
    lines.append("- **`q_barrier_dft` is intentionally absent** (Deviation #3). "
                 "Stage 4 will emit 44 features vs their 45.")
    lines.append("- **`_gfn2` will be dropped** by the unpatched `_manual_runner` — "
                 "confirmed in Rev 1 G-E. Either patch the line-226 substring to "
                 "`am1|gfn2`, or rename Stage-3 SQM columns to `_am1` at write.")
    lines.append("- **Stage 4 hazard: `_clean_dist_contr` overwrite.** The function "
                 "loops every `contributions` column and emits `'1'` / `'2'` "
                 "(renamed to `distortion_energy_{1,2}_{method}`). When both "
                 "`distortion_contributions_am1` and `distortion_contributions_dft` "
                 "are present at Stage 4, the second overwrites the first on "
                 "columns named identically. Prefix or method-suffix must be "
                 "preserved by construction.")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("```")
    lines.append("Ref Comparison/spec18r1_espley_s1_labels_fix/")
    lines.append("  code/{build_2ch_labels.py, verify_artifact.py, aggregate.py, submit_s1.sh}")
    lines.append("  data/cohort_notes.json")
    lines.append("  logs/{build.log, gates.log}")
    lines.append("  results/{labels_2ch_400dipolar.pkl,")
    lines.append("           labels_2ch_400dipolar.INSPECTION_ONLY.parquet,")
    lines.append("           anomaly_triage.csv, asm_residual_outliers.csv,")
    lines.append("           sub_source_stats.csv, ds3_distribution_comparison.csv,")
    lines.append("           DEVIATIONS.md, summary.md}")
    lines.append("  figures/{target_hist_3panel.png, sub_source_box.png,")
    lines.append("           asm_residual_hist.png}")
    lines.append("```")
    lines.append("")

    OUT_SUM.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_pickle(PKL)
    stats = sub_source_stats(df)
    ds3_tbl = ds3_comparison(df)
    hist_3panel(df)
    sub_source_box(df)
    write_deviations()
    write_summary(df, stats, ds3_tbl)


if __name__ == "__main__":
    main()
