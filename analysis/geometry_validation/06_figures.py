#!/usr/bin/env python3
"""SPEC14 Step 6 — three figures at 300 dpi (English labels)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")

CH = ["elst", "pauli", "oi", "disp", "cpcm", "cds"]


def main():
    R = pd.read_csv(BASE / "artifacts" / "channel_comparison.csv")
    new = pd.read_csv(BASE / "artifacts" / "channels_b3lyp.csv").set_index("rxn_id")
    old = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    old["reaction_number"] = old["reaction_number"].astype(int)
    old = old.set_index("reaction_number")

    # ---- Fig 1: geometry-induced error + δ propagation ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(R)); w = 0.38
    ax.bar(x - w/2, R["mae"], w, label="single reaction |Δ|", color="#2c6fbb")
    ax.bar(x + w/2, R["delta_propagated"], w,
           label="propagated to δ (×√2)", color="#8fbce6")
    ax.axhline(1.0, ls="--", c="#c0392b", lw=1.5, label="target 1.0 kcal/mol")
    ax.set_xticks(x); ax.set_xticklabels(R["channel"])
    ax.set_ylabel("geometry-induced error (kcal/mol)")
    ax.set_title("Channel sensitivity to TS geometry level (low-level vs B3LYP)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "fig1_geometry_error.png", dpi=300)
    plt.close(fig)

    # ---- Fig 2: parity plot per channel ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, ch in zip(axes.flat, CH):
        idx = new.index.intersection(old.index)
        a = old.loc[idx, f"{ch}_dft"].values
        b = new.loc[idx, ch].values
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else float("nan")
        ax.scatter(a, b, s=15, alpha=0.55, color="#2c6fbb")
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, alpha=0.6)
        ax.set_xlabel(f"low-level TS  {ch} (kcal/mol)")
        ax.set_ylabel(f"B3LYP TS  {ch}")
        ax.set_title(f"{ch}   r={r:.3f}   n={len(a)}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "fig2_parity_by_channel.png", dpi=300)
    plt.close(fig)

    # ---- Fig 3: bond-length shift vs channel shift ----
    bpath = BASE / "artifacts" / "bond_length_shifts.csv"
    if bpath.exists():
        bl = pd.read_csv(bpath).set_index("rxn_id")
        mean_bd = 0.5 * (bl["d1_diff"].abs() + bl["d2_diff"].abs())
        idx = new.index.intersection(old.index).intersection(bl.index)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, ch in zip(axes, ["pauli", "elst", "oi"]):
            a = old.loc[idx, f"{ch}_dft"].values
            b = new.loc[idx, ch].values
            mask = ~(np.isnan(a) | np.isnan(b))
            common = idx[mask]
            ch_shift = np.abs(b[mask] - a[mask])
            bd_shift = mean_bd.loc[common].values
            r = float(np.corrcoef(ch_shift, bd_shift)[0, 1]) if len(common) >= 3 else float("nan")
            ax.scatter(bd_shift, ch_shift, s=15, alpha=0.55, color="#e07a5f")
            ax.set_xlabel("|Δ formed-bond length| (Å)")
            ax.set_ylabel(f"|Δ {ch}| (kcal/mol)")
            ax.set_title(f"{ch}   r={r:.3f}")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(BASE / "figures" / "fig3_bondlen_vs_channel.png", dpi=300)
        plt.close(fig)
    print("figures saved: fig1_geometry_error.png, fig2_parity_by_channel.png, fig3_bondlen_vs_channel.png")


if __name__ == "__main__":
    main()
