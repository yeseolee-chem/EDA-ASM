#!/usr/bin/env python3
"""SPEC17rev2 Step 11 — assemble REPORT.md from gate statuses + figures.

Aggregates:
  - GATE{0..9}_STATUS.txt entries
  - reactant_build.csv summary
  - folds.csv summary
  - generation_quality.csv summary (RMSD + Δd histograms as figures)
  - channel_impact.csv (per-channel σ + δ_lin)

Also emits figures/rmsd_hist.png, figures/dd_scatter.png, figures/channel_delta.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train")
ART = BASE / "artifacts"
FIG = BASE / "figures"

for _d in ("artifacts", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / _d).mkdir(parents=True, exist_ok=True)


def read_gate(k: int) -> str:
    p = ART / f"GATE{k}_STATUS.txt"
    return p.read_text().strip() if p.exists() else "(missing)"


def maybe_read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def plot_rmsd(df: pd.DataFrame) -> None:
    v = df.rmsd.dropna()
    if v.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(v, bins=40, alpha=0.8)
    ax.set_xlabel("RMSD gen–ref (Å)")
    ax.set_ylabel("count")
    ax.axvline(0.053, color="k", lw=1, ls="--", label="React-OT paper (T1x)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "rmsd_hist.png", dpi=140)
    plt.close(fig)


def plot_dd(df: pd.DataFrame) -> None:
    if "dd1" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(df.dd1, df.dd2, s=6, alpha=0.4)
    lim = np.nanpercentile(np.abs(np.concatenate([df.dd1.dropna(), df.dd2.dropna()])), 99)
    lim = max(lim, 0.5)
    for x in (-0.144, 0.144, -0.173, 0.173):
        ax.axhline(x, color="gray", lw=0.5)
        ax.axvline(x, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δd1 (Å)")
    ax.set_ylabel("Δd2 (Å)")
    ax.set_title("forming-bond error\n(grey = AM1 baseline)")
    fig.tight_layout()
    fig.savefig(FIG / "dd_scatter.png", dpi=140)
    plt.close(fig)


def plot_channels(ci: pd.DataFrame) -> None:
    if ci is None or ci.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ci = ci.copy()
    ci["delta_lin"].plot.bar(ax=ax)
    ax.set_ylabel("δ_lin (kcal/mol)")
    ax.axhline(5, color="g", ls="--", lw=1, label="PASS ≤ 5")
    ax.axhline(15, color="r", ls="--", lw=1, label="FAIL > 15")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "channel_delta.png", dpi=140)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)

    build = maybe_read_csv(ART / "reactant_build.csv")
    folds = maybe_read_csv(ART / "folds.csv")
    gq = maybe_read_csv(ART / "generation_quality.csv")
    ci_path = BASE / "channel_impact" / "channel_impact.csv"
    ci = pd.read_csv(ci_path, index_col=0) if ci_path.exists() else None

    if gq is not None:
        plot_rmsd(gq[gq.ok])
        plot_dd(gq[gq.ok])
    if ci is not None:
        plot_channels(ci)

    lines = ["# SPEC17rev2 REPORT — otfm_train Coley 5269", ""]

    lines.append("## Gate statuses")
    for k in range(10):
        lines.append(f"- **GATE-{k}**  `{read_gate(k).splitlines()[0]}`")
    lines.append("")

    if build is not None:
        ok = build[build.ok] if "ok" in build.columns else build
        lines += [
            "## Step 2 — reactant complex build",
            f"- rows: {len(build)}  ok: {len(ok)}",
            f"- max_iso hit rate: {ok.hit_cap.mean():.1%}" if "hit_cap" in ok.columns else "",
            f"- inter_R median: {ok.inter_R.median():.3f} Å  (<1.0 Å: {int((ok.inter_R < 1.0).sum())})"
            if "inter_R" in ok.columns else "",
            f"- R-TS RMSD median: {ok.rmsd_R_TS.median():.3f} Å"
            if "rmsd_R_TS" in ok.columns else "",
            "",
        ]

    if folds is not None:
        counts = folds.fold.value_counts().sort_index()
        lines += [
            "## Step 5 — fold sizes",
            "| fold | n |",
            "|---|---|",
            *[f"| {f} | {n} |" for f, n in counts.items()],
            "",
        ]

    if gq is not None:
        ok = gq[gq.ok]
        lines += [
            "## Step 9 — generation quality",
            f"- evaluable: {len(ok)}",
            f"- RMSD median: {np.median(ok.rmsd):.4f} Å   "
            f"p95: {np.percentile(ok.rmsd, 95):.4f} Å",
        ]
        for c in ("dd1", "dd2"):
            if c in ok.columns:
                v = ok[c].dropna()
                lines.append(
                    f"- |Δ{c}| median: {v.abs().median():.4f} Å   "
                    f"(AM1 baseline: 0.144–0.173)"
                )
        lines += ["", "![RMSD](figures/rmsd_hist.png)",
                  "", "![Δd scatter](figures/dd_scatter.png)", ""]

    if ci is not None:
        lines += [
            "## Step 10 — channel Δ error",
            ci.to_markdown(),
            "",
            "![channel δ_lin](figures/channel_delta.png)",
            "",
        ]

    (BASE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {BASE / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
