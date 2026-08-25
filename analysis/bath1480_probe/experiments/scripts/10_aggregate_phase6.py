#!/usr/bin/env python3
"""Aggregate Phase 6 (Phase 5 XGB + MACE residual) vs Phase 5 XGB direct comparison.

Phase 5 XGB baseline: comparison_v3/records_all.csv (seed-averaged MAE).
Phase 6:              phase6_p5xgb_mace_v1/seed_*/metrics.json (5 seeds).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
P5_CSV = BASE / "results/comparison_v3/records_all.csv"
P6_ROOT = BASE / "results/phase6_p5xgb_mace_v1"
OUT = BASE / "results/phase6_p5xgb_mace_v1"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_TO_CANONICAL = {
    "d1_own_dft": "d1", "d2_own_dft": "d2",
    "elst_dft": "elst", "pauli_dft": "pauli", "oi_dft": "oi",
    "disp_dft": "disp", "cpcm_dft": "cpcm", "cds_dft": "cds",
}
CHANNELS = ["d1", "d2", "pauli", "oi", "elst", "disp", "cpcm", "cds"]


def main():
    # Phase 6 per-seed metrics
    rows = []
    for f in sorted(P6_ROOT.glob("seed_*/metrics.json")):
        seed = int(f.parent.name.split("_")[1])
        m = json.loads(f.read_text())
        for tgt, d in m.items():
            rows.append({
                "seed": seed,
                "target": tgt,
                "canonical": TARGET_TO_CANONICAL[tgt],
                "mae_full": d["mae"],
                "mae_xgb_only": d["xgb_baseline_mae"],
                "r2": d["r2"],
            })
    p6 = pd.DataFrame(rows)
    p6.to_csv(OUT / "phase6_per_seed.csv", index=False)

    p6_agg = p6.groupby("canonical").agg(
        mae_p6_full=("mae_full", "mean"),
        mae_p6_full_std=("mae_full", "std"),
        mae_p6_xgb_only=("mae_xgb_only", "mean"),
        r2_p6=("r2", "mean"),
    ).reset_index()

    # Phase 5 XGB from records_all
    p5 = pd.read_csv(P5_CSV)
    p5x = p5[(p5["phase"] == "Phase 5 (no filter)") & (p5["model"] == "xgb")]
    p5x_agg = p5x.groupby("canonical").agg(
        mae_p5=("mae_mean", "first"),
        mae_p5_std=("mae_std", "first"),
        r2_p5=("r2_mean", "first"),
    ).reset_index()

    merged = p5x_agg.merge(p6_agg, on="canonical")
    merged = merged.set_index("canonical").reindex(CHANNELS).reset_index()
    merged["delta_mae"] = merged["mae_p6_full"] - merged["mae_p5"]
    merged["pct_change"] = merged["delta_mae"] / merged["mae_p5"] * 100
    merged["mace_contrib"] = merged["mae_p6_xgb_only"] - merged["mae_p6_full"]
    merged.to_csv(OUT / "phase5_vs_phase6.csv", index=False)

    print("\nPhase 5 XGB vs Phase 6 (Phase 5 XGB + MACE residual) — per channel:")
    print(merged[["canonical", "mae_p5", "mae_p6_xgb_only", "mae_p6_full",
                  "delta_mae", "pct_change", "mace_contrib"]].to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CHANNELS))
    w = 0.28
    ax.bar(x - w, merged["mae_p5"], w, yerr=merged["mae_p5_std"], color="#9b59b6",
           label="Phase 5 XGB", edgecolor="black", linewidth=0.4, capsize=2.5)
    ax.bar(x, merged["mae_p6_xgb_only"], w, color="#c8b0dd",
           label="Phase 6: XGB alone (rerun)", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, merged["mae_p6_full"], w, yerr=merged["mae_p6_full_std"], color="#e07a5f",
           label="Phase 6: XGB + MACE residual", edgecolor="black", linewidth=0.4, capsize=2.5)
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS, fontsize=10)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "phase5_vs_phase6_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {OUT / 'phase5_vs_phase6_bars.png'}")

    # Short REPORT
    lines = [
        "# Phase 6 vs Phase 5 XGB",
        "",
        "Setup: Phase 5 features (78, oracle-clean, no filter) + paper 80/10/10 split (5 seeds: 22/23/14/1/2).",
        "Phase 6 adds an attention-pooled MACE-OFF23 residual head on top of the XGB base.",
        "",
        "## Per-channel MAE (seed-averaged, kcal/mol)",
        "```",
        merged[["canonical", "mae_p5", "mae_p6_xgb_only", "mae_p6_full",
                "delta_mae", "pct_change", "mace_contrib"]].to_string(index=False),
        "```",
        "",
        "- `mace_contrib` = XGB-only MAE − full (XGB+MACE) MAE; positive → residual helped.",
        "- `pct_change` = Phase 6 vs Phase 5 XGB percentage change (positive → Phase 6 worse).",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
