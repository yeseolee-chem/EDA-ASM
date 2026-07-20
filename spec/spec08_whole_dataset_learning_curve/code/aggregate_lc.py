"""SPEC_08 whole-dataset LC — aggregate cells into table + PNG plots.

Reads:
  spec/spec08_whole_dataset_learning_curve/oof/size{N}/fold{f}/member{m}.json

Writes:
  results/learning_curve.csv        long-form: size × arm × channel × metric
  results/summary.csv               size × arm: mean/std across folds
  results/REPORT.md                 human-readable summary
  figures/learning_curve.png        5 channels + barrier NMAE vs actual N,
                                    both arms overlaid
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec08_whole_dataset_learning_curve"
OOF_ROOT = SPEC / "oof"
RES_DIR = SPEC / "results"
FIG_DIR = SPEC / "figures"
LC_SPLITS = SPEC / "splits/lc_splits.json"
CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARMS = ["xgb_base", "xgb_delta"]
METRICS = ["nmae", "mae", "rmse"]


def load_cells(only_member: int | None):
    rows = []
    for size_dir in sorted(OOF_ROOT.glob("size*")):
        size = int(size_dir.name.replace("size", ""))
        for fold_dir in sorted(size_dir.glob("fold*")):
            fold = int(fold_dir.name.replace("fold", ""))
            for j in sorted(fold_dir.glob("member*.json")):
                member = int(j.stem.replace("member", ""))
                if only_member is not None and member != only_member:
                    continue
                with open(j) as fh:
                    d = json.load(fh)
                for arm in ARMS:
                    if arm not in d:
                        continue
                    m = d[arm]["metrics"]
                    row = {
                        "size_target": size,
                        "size_actual": d["size_actual"],
                        "fold": fold, "member": member, "arm": arm,
                    }
                    row.update(m)
                    rows.append(row)
    return pd.DataFrame(rows)


def missing_report(only_member: int | None):
    with open(LC_SPLITS) as fh:
        lc = json.load(fh)
    sizes = lc["sizes"]
    folds = sorted(int(k) for k in lc["folds"].keys())
    members = [only_member] if only_member is not None else [0]
    missing = []
    for m in members:
        for size in sizes:
            for f in folds:
                p = OOF_ROOT / f"size{size}" / f"fold{f}" / f"member{m}.json"
                if not p.exists():
                    missing.append((size, f, m))
    return missing


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns
                   if any(c.startswith(m + "_") for m in METRICS)]
    # Group by (size_target, arm) only. Do NOT split by size_actual, which
    # differs by 1 across folds when the target hits the fold-train cap
    # (e.g. targets 700 / 786 land on actual_n = 626 for 3 folds and 627
    # for 2 folds). Splitting would create phantom clusters at x ≈ 626-627.
    # We also carry size_actual_mean along for the plot's x-axis when it's
    # more informative than the nominal target.
    keys = ["size_target", "arm"]
    agg = (df.groupby(keys)
             .agg({**{c: ["mean", "std", "count"] for c in metric_cols},
                   "size_actual": ["mean", "min", "max"]})
             .reset_index())
    agg.columns = [
        "_".join([c for c in col if c]).rstrip("_")
        for col in agg.columns.to_flat_index()
    ]
    return agg


def plot_curves(summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels_plus = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    colors = {"xgb_base": "#1f77b4", "xgb_delta": "#d62728"}

    for i, ch in enumerate(channels_plus):
        ax = axes[i]
        for arm in ARMS:
            sub = (summary[summary["arm"] == arm]
                   .sort_values("size_target"))
            if sub.empty:
                continue
            mcol = f"nmae_{ch}_mean"
            scol = f"nmae_{ch}_std"
            if mcol not in sub.columns:
                continue
            # Plot at the nominal size_target so all 8 requested sizes
            # get distinct x-positions (700 and 786 no longer collapse
            # onto ~626-627). Actual n is annotated below.
            x = sub["size_target"].values
            y = sub[mcol].values
            yerr = sub[scol].values if scol in sub.columns else None
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3,
                        label=arm, color=colors.get(arm))
        ax.set_title(f"NMAE — {ch}")
        ax.set_xlabel("train size (target N; sizes ≥ 700 cap at ~626)")
        ax.set_ylabel("NMAE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("SPEC_08 whole-dataset LC  •  xgb28 base vs xgb28+δ", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_report(df: pd.DataFrame, summary: pd.DataFrame,
                 missing: list, out_path: Path, spec_tag: str):
    lines = []
    lines.append(f"# {spec_tag} — REPORT\n")
    lines.append(f"Cells present: {len(df)//len(ARMS)}  (rows over both arms = {len(df)})\n")
    if missing:
        lines.append(f"Missing cells: {len(missing)}\n")
        preview = missing[:30]
        for size, fold, member in preview:
            lines.append(f"- size={size} fold={fold} member={member}")
        if len(missing) > len(preview):
            lines.append(f"- ... ({len(missing) - len(preview)} more)")
        lines.append("")
    else:
        lines.append("All planned cells present.\n")

    lines.append("## Barrier NMAE (mean ± std across folds)\n")
    lines.append("| size | actual | xgb_base | xgb_delta | Δ (base − delta) |")
    lines.append("|-----:|-------:|---------:|----------:|-----------------:|")
    sizes = sorted(summary["size_target"].unique())
    for s in sizes:
        row_b = summary[(summary["size_target"] == s) & (summary["arm"] == "xgb_base")]
        row_d = summary[(summary["size_target"] == s) & (summary["arm"] == "xgb_delta")]
        if not row_b.empty:
            n_min = int(row_b["size_actual_min"].iloc[0])
            n_max = int(row_b["size_actual_max"].iloc[0])
            actual = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"
        else:
            actual = "-"
        b_mean = row_b["nmae_barrier_mean"].iloc[0] if not row_b.empty else float("nan")
        b_std = row_b["nmae_barrier_std"].iloc[0] if not row_b.empty else float("nan")
        d_mean = row_d["nmae_barrier_mean"].iloc[0] if not row_d.empty else float("nan")
        d_std = row_d["nmae_barrier_std"].iloc[0] if not row_d.empty else float("nan")
        diff = b_mean - d_mean if not (np.isnan(b_mean) or np.isnan(d_mean)) else float("nan")
        lines.append(f"| {s} | {actual} | {b_mean:.3f} ± {b_std:.3f} "
                     f"| {d_mean:.3f} ± {d_std:.3f} | {diff:+.3f} |")
    lines.append("")

    for ch in CHANNELS:
        lines.append(f"### Channel: {ch}\n")
        lines.append("| size | actual | xgb_base | xgb_delta | Δ |")
        lines.append("|-----:|-------:|---------:|----------:|--:|")
        for s in sizes:
            row_b = summary[(summary["size_target"] == s) & (summary["arm"] == "xgb_base")]
            row_d = summary[(summary["size_target"] == s) & (summary["arm"] == "xgb_delta")]
            if not row_b.empty:
                n_min = int(row_b["size_actual_min"].iloc[0])
                n_max = int(row_b["size_actual_max"].iloc[0])
                actual = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"
            else:
                actual = "-"
            b = row_b[f"nmae_{ch}_mean"].iloc[0] if not row_b.empty else float("nan")
            d = row_d[f"nmae_{ch}_mean"].iloc[0] if not row_d.empty else float("nan")
            diff = b - d if not (np.isnan(b) or np.isnan(d)) else float("nan")
            lines.append(f"| {s} | {actual} | {b:.3f} | {d:.3f} | {diff:+.3f} |")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--spec-tag", default="SPEC_08 whole-dataset LC")
    args = ap.parse_args()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cells(args.member)
    if df.empty:
        print("[aggregate] no cells found yet; run chain first")
        missing = missing_report(args.member)
        print(f"[aggregate] {len(missing)} cells planned")
        return

    df.to_csv(RES_DIR / "learning_curve.csv", index=False)
    summary = summarise(df)
    summary.to_csv(RES_DIR / "summary.csv", index=False)
    print(f"[aggregate] {len(df)} rows → summary rows {len(summary)}")

    missing = missing_report(args.member)
    write_report(df, summary, missing,
                 RES_DIR / "REPORT.md", spec_tag=args.spec_tag)
    plot_curves(summary, FIG_DIR / "learning_curve.png")
    print(f"[aggregate] wrote  {RES_DIR/'REPORT.md'}")
    print(f"[aggregate] wrote  {FIG_DIR/'learning_curve.png'}")
    if missing:
        print(f"[aggregate] {len(missing)} cells still missing")


if __name__ == "__main__":
    main()
