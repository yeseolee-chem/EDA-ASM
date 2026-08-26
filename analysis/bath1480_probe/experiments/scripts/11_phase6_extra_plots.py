#!/usr/bin/env python3
"""Extra Phase 6 diagnostic plots.

Outputs (in results/phase6_p5xgb_mace_v1/):
  mace_contribution_bar.png   — per-channel MACE residual contribution
  per_seed_dots.png           — per-seed MAE dots for XGB-only vs XGB+MACE
  training_loss_curves.png    — MACE residual training loss vs epoch (5 seeds)
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
P6_ROOT = BASE / "results/phase6_p5xgb_mace_v1"

TGT_TO_CANON = {
    "d1_own_dft": "d1", "d2_own_dft": "d2",
    "elst_dft": "elst", "pauli_dft": "pauli", "oi_dft": "oi",
    "disp_dft": "disp", "cpcm_dft": "cpcm", "cds_dft": "cds",
}
CHANNELS = ["d1", "d2", "pauli", "oi", "elst", "disp", "cpcm", "cds"]
SEEDS = [22, 23, 14, 1, 2]


def gather_metrics():
    rows = []
    for seed in SEEDS:
        f = P6_ROOT / f"seed_{seed}" / "metrics.json"
        m = json.loads(f.read_text())
        for tgt, d in m.items():
            rows.append({
                "seed": seed, "canonical": TGT_TO_CANON[tgt],
                "mae_full": d["mae"], "mae_xgb_only": d["xgb_baseline_mae"],
            })
    return pd.DataFrame(rows)


def find_log(seed):
    """Find the SLURM log corresponding to this seed."""
    for f in sorted(BASE.glob("results/phase6.*.log")):
        head = f.read_text().splitlines()[:5]
        if any(f"seed={seed}" in ln for ln in head):
            return f
    return None


def parse_loss_curve(logpath):
    """Extract (epoch, loss) pairs from a Phase 6 log."""
    xs, ys = [], []
    pat = re.compile(r"epoch\s+(\d+)\s+loss=([\d.]+)")
    with logpath.open() as f:
        for line in f:
            m = pat.search(line)
            if m:
                xs.append(int(m.group(1)))
                ys.append(float(m.group(2)))
    return xs, ys


def main():
    df = gather_metrics()

    # 1) MACE contribution bar (with per-seed dots)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(CHANNELS))
    contrib = df.assign(contrib=lambda d: d["mae_xgb_only"] - d["mae_full"])
    contrib_mean = contrib.groupby("canonical")["contrib"].mean().reindex(CHANNELS)
    contrib_std = contrib.groupby("canonical")["contrib"].std().reindex(CHANNELS)
    colors = ["#3d9d9b" if v >= 0 else "#e07a5f" for v in contrib_mean]
    ax.bar(x, contrib_mean, 0.6, yerr=contrib_std, color=colors,
           edgecolor="black", linewidth=0.4, capsize=3,
           label="MACE contribution (mean ± std, 5 seeds)")
    for i, ch in enumerate(CHANNELS):
        vals = contrib[contrib["canonical"] == ch]["contrib"].values
        ax.scatter([i]*len(vals), vals, color="black", s=15, zorder=3, alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("XGB-only MAE − Full MAE (kcal/mol)")
    ax.set_title("MACE residual contribution per channel  (positive = residual helped)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(P6_ROOT / "mace_contribution_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote mace_contribution_bar.png")

    # 2) Per-seed dots — XGB-only vs XGB+MACE side by side
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, ch in enumerate(CHANNELS):
        ax = axes[i // 4, i % 4]
        sub = df[df["canonical"] == ch].sort_values("seed")
        seeds_str = [str(s) for s in sub["seed"]]
        xs = np.arange(len(sub))
        ax.scatter(xs - 0.15, sub["mae_xgb_only"], s=60, color="#9b59b6",
                   marker="o", label="XGB alone")
        ax.scatter(xs + 0.15, sub["mae_full"], s=60, color="#e07a5f",
                   marker="s", label="XGB + MACE")
        for j in xs:
            ax.plot([j - 0.15, j + 0.15],
                    [sub["mae_xgb_only"].iloc[j], sub["mae_full"].iloc[j]],
                    "k-", linewidth=0.5, alpha=0.4)
        ax.set_xticks(xs); ax.set_xticklabels(seeds_str, fontsize=8)
        ax.set_xlabel("seed", fontsize=8)
        ax.set_ylabel("MAE (kcal/mol)", fontsize=9)
        ax.set_title(ch, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(P6_ROOT / "per_seed_dots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote per_seed_dots.png")

    # 3) Training loss curves (MACE residual, 5 seeds)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for seed in SEEDS:
        logp = find_log(seed)
        if logp is None:
            continue
        xs, ys = parse_loss_curve(logp)
        if not xs:
            continue
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1,
                label=f"seed {seed}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("residual loss (normed L1)")
    ax.set_title("MACE residual net — training loss (5 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(P6_ROOT / "training_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote training_loss_curves.png")


if __name__ == "__main__":
    main()
