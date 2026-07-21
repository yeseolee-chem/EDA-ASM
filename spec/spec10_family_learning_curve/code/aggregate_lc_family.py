"""SPEC_10 — aggregate per-family learning-curve cells.

Reads:
  spec/spec10_family_learning_curve/oof/{family}/size{N}/fold{f}/member{m}.json

Writes:
  results/learning_curve.csv        long form: family × size × arm × channel × metric
  results/summary.csv               family × size × arm × channel: mean/std/count
  results/REPORT.md
  figures/learning_curve_{family}.pdf   per-family panel plot (5 channels + barrier)
  figures/learning_curve_all.pdf        4 families × barrier NMAE overview

Only aggregates whatever cells are on disk; prints a missing-cell report
so progress can be inspected while the chain launcher is still running.

CLI:
  --member M       aggregate only member M (default: all present)
  --spec-tag TAG   overrides "SPEC_10 family LC" in the output header
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec10_family_learning_curve"
OOF_ROOT = SPEC / "oof"
RES_DIR = SPEC / "results"
FIG_DIR = SPEC / "figures"
LC_SPLITS = SPEC / "splits/lc_family_splits.json"
CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARMS = ["xgb_base", "xgb_delta"]
METRICS = ["nmae", "mae", "rmse"]


def load_cells(only_member: int | None):
    rows = []
    for fam_dir in sorted(OOF_ROOT.glob("*")):
        if not fam_dir.is_dir():
            continue
        fam = fam_dir.name
        for size_dir in sorted(fam_dir.glob("size*")):
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
                            "family": fam,
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
    families = lc["families"]
    folds = [0, 1, 2, 3, 4]
    members = [only_member] if only_member is not None else [0]
    missing = []
    for m in members:
        for fam in families:
            for size in sizes:
                for f in folds:
                    p = OOF_ROOT / fam / f"size{size}" / f"fold{f}" / f"member{m}.json"
                    if not p.exists():
                        missing.append((fam, size, f, m))
    return missing


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns
                   if any(c.startswith(m + "_") for m in METRICS)]
    keys = ["family", "size_target", "size_actual", "arm"]
    agg = (df.groupby(keys)[metric_cols]
             .agg(["mean", "std", "count"])
             .reset_index())
    agg.columns = [
        "_".join([c for c in col if c]).rstrip("_")
        for col in agg.columns.to_flat_index()
    ]
    return agg


def plot_family(summary: pd.DataFrame, family: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub_fam = summary[summary["family"] == family]
    if sub_fam.empty:
        return
    channels_plus = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    colors = {"xgb_base": "#1f77b4", "xgb_delta": "#d62728"}

    for i, ch in enumerate(channels_plus):
        ax = axes[i]
        for arm in ARMS:
            s = (sub_fam[sub_fam["arm"] == arm]
                 .sort_values("size_target"))
            if s.empty:
                continue
            mcol = f"nmae_{ch}_mean"
            scol = f"nmae_{ch}_std"
            if mcol not in s.columns:
                continue
            x = s["size_actual"].values
            y = s[mcol].values
            yerr = s[scol].values if scol in s.columns else None
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3,
                        label=arm, color=colors.get(arm))
        ax.set_xlabel("train size (actual)")
        ax.set_ylabel("NMAE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_all_barrier(summary: pd.DataFrame, families: list, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()
    colors = {"xgb_base": "#1f77b4", "xgb_delta": "#d62728"}

    for i, fam in enumerate(families):
        ax = axes[i]
        sub_fam = summary[summary["family"] == fam]
        for arm in ARMS:
            s = (sub_fam[sub_fam["arm"] == arm]
                 .sort_values("size_target"))
            if s.empty:
                continue
            x = s["size_actual"].values
            y = s["nmae_barrier_mean"].values
            yerr = (s["nmae_barrier_std"].values
                    if "nmae_barrier_std" in s.columns else None)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3,
                        label=arm, color=colors.get(arm))
        ax.set_xlabel("train size (actual)")
        ax.set_ylabel("barrier NMAE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
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
        for fam, size, fold, member in preview:
            lines.append(f"- {fam} size={size} fold={fold} member={member}")
        if len(missing) > len(preview):
            lines.append(f"- ... ({len(missing) - len(preview)} more)")
        lines.append("")
    else:
        lines.append("All planned cells present.\n")

    families = sorted(summary["family"].unique()) if not summary.empty else []
    for fam in families:
        lines.append(f"## {fam} — barrier NMAE (mean ± std)\n")
        lines.append("| size | actual | xgb_base | xgb_delta | Δ (base − delta) |")
        lines.append("|-----:|-------:|---------:|----------:|-----------------:|")
        sub_fam = summary[summary["family"] == fam]
        sizes = sorted(sub_fam["size_target"].unique())
        for s in sizes:
            row_b = sub_fam[(sub_fam["size_target"] == s) & (sub_fam["arm"] == "xgb_base")]
            row_d = sub_fam[(sub_fam["size_target"] == s) & (sub_fam["arm"] == "xgb_delta")]
            actual = int(row_b["size_actual"].iloc[0]) if not row_b.empty else "-"
            b_mean = row_b["nmae_barrier_mean"].iloc[0] if not row_b.empty else float("nan")
            b_std = row_b["nmae_barrier_std"].iloc[0] if not row_b.empty else float("nan")
            d_mean = row_d["nmae_barrier_mean"].iloc[0] if not row_d.empty else float("nan")
            d_std = row_d["nmae_barrier_std"].iloc[0] if not row_d.empty else float("nan")
            diff = b_mean - d_mean if not (np.isnan(b_mean) or np.isnan(d_mean)) else float("nan")
            lines.append(f"| {s} | {actual} | {b_mean:.3f} ± {b_std:.3f} "
                         f"| {d_mean:.3f} ± {d_std:.3f} | {diff:+.3f} |")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--spec-tag", default="SPEC_10 family LC")
    args = ap.parse_args()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cells(args.member)
    if df.empty:
        print("[aggregate] no cells found yet; run chain_lc_family.sh first")
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
    families = sorted(summary["family"].unique())
    for fam in families:
        plot_family(summary, fam, FIG_DIR / f"learning_curve_{fam}.png")
    plot_all_barrier(summary, families, FIG_DIR / "learning_curve_all.png")

    print(f"[aggregate] wrote  {RES_DIR/'REPORT.md'}")
    print(f"[aggregate] wrote  {FIG_DIR/'learning_curve_all.png'}")
    if missing:
        print(f"[aggregate] {len(missing)} cells still missing")


if __name__ == "__main__":
    main()
