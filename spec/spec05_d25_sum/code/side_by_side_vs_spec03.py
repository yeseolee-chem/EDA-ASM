"""Rebuild SPEC_03 baseline_bars.png without the neural-star marker,
and produce a side-by-side comparison figure (SPEC_03 classical | SPEC_05 2x2).

Outputs:
  - spec/spec03_bmax/figures/baseline_bars.png            (overwritten, no star)
  - spec/spec05_d25_sum/figures/side_by_side_spec03_spec05.png   (new)
"""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC03 = REPO / "spec/spec03_bmax/results/baseline_leaderboard.csv"
SPEC05 = REPO / "spec/spec05_d25_sum/results/2x2_metrics.csv"
FIG03 = REPO / "spec/spec03_bmax/figures/baseline_bars.png"
FIG_COMBINED = REPO / "spec/spec05_d25_sum/figures/side_by_side_spec03_spec05.png"

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]

sp03 = pd.read_csv(SPEC03)
sp05 = pd.read_csv(SPEC05)

# ============ SPEC_03 panel ============
methods_03 = ["ridge", "lasso", "enet", "xgb"]
colors_03 = {"ridge": "#4b779a", "lasso": "#c05e2b", "enet": "#4ba36c", "xgb": "#8b3a62"}
channels_plot_03 = CHANNELS + ["barrier_sum"]

def sp03_val(method, ch):
    row = sp03[(sp03.model == method) & (sp03.channel == ch)]
    return float(row.NMAE.iloc[0]) if len(row) else np.nan

x03 = np.arange(len(channels_plot_03))
w03 = 0.2

# Overwrite spec03 baseline_bars.png WITHOUT the neural star
fig, ax = plt.subplots(figsize=(13, 5.5))
for i, m in enumerate(methods_03):
    vals = [sp03_val(m, ch) for ch in channels_plot_03]
    ax.bar(x03 + (i - 1.5) * w03, vals, w03, label=m, color=colors_03[m],
           edgecolor="white", lw=0.4)
ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
ax.set_xticks(x03); ax.set_xticklabels(channels_plot_03)
ax.set_ylabel("NMAE (5-fold CV)")
ax.set_title("SPEC_03 - Classical baselines (m3 v9, 783 rxn)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG03, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"overwrote {FIG03} (no star)")

# ============ SPEC_05 panel data ============
variants_05 = ["M0", "M1", "M2", "M3"]
colors_05 = {"M0": "#4b779a", "M1": "#c05e2b", "M2": "#4ba36c", "M3": "#8b3a62"}
channels_plot_05 = CHANNELS + ["barrier"]

def sp05_val(variant, ch):
    row = sp05[(sp05.variant == variant) & (sp05.channel == ch)]
    return float(row.NMAE.iloc[0]) if len(row) else np.nan

x05 = np.arange(len(channels_plot_05))
w05 = 0.2

# ============ Combined side-by-side figure ============
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

for i, m in enumerate(methods_03):
    vals = [sp03_val(m, ch) for ch in channels_plot_03]
    ax_l.bar(x03 + (i - 1.5) * w03, vals, w03, label=m, color=colors_03[m],
             edgecolor="white", lw=0.4)
ax_l.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
ax_l.set_xticks(x03); ax_l.set_xticklabels(channels_plot_03)
ax_l.set_ylabel("NMAE (5-fold pooled)")
ax_l.set_title("SPEC_03 - Classical baselines (ridge / lasso / enet / xgb)")
ax_l.legend(fontsize=9, loc="upper right"); ax_l.grid(alpha=0.3, axis="y")

variant_labels = {
    "M0": "M0 (24-d, per-ch)",
    "M1": "M1 (25-d, +d25)",
    "M2": "M2 (24-d, +sum)",
    "M3": "M3 (25-d, +d25 +sum)",
}
for i, v in enumerate(variants_05):
    vals = [sp05_val(v, ch) for ch in channels_plot_05]
    ax_r.bar(x05 + (i - 1.5) * w05, vals, w05, label=variant_labels[v],
             color=colors_05[v], edgecolor="white", lw=0.4)
ax_r.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
ax_r.set_xticks(x05); ax_r.set_xticklabels(channels_plot_05)
ax_r.set_title("SPEC_05 - XGB 2x2 ablation (d25 x sum-consistency)")
ax_r.legend(fontsize=9, loc="upper right"); ax_r.grid(alpha=0.3, axis="y")

fig.suptitle("SPEC_03 classical baselines vs SPEC_05 2x2 XGB ablation (m3 v9, 783 rxns)",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_COMBINED, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG_COMBINED}")
