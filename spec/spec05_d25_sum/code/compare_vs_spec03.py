"""Compare SPEC_05 2x2 XGB ablation against SPEC_03 xgb baseline.

SPEC_03 xgb (24-d, plain per-channel XGB) = the reference to beat.
SPEC_05 M0 (24-d, per-channel) should match SPEC_03 xgb closely.
SPEC_05 M1/M2/M3 quantify the improvement from d25 / sum-consistency / both.

Outputs (spec/spec05_d25_sum/figures/):
  - vs_spec03_xgb_bars.png     (grouped NMAE bar: xgb / M0 / M1 / M2 / M3)
  - vs_spec03_xgb_deltas.png   (delta_NMAE vs xgb baseline, per channel)
  - vs_spec03_xgb_table.csv    (numeric side-by-side)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC03 = REPO / "spec/spec03_bmax/results/baseline_leaderboard.csv"
SPEC05 = REPO / "spec/spec05_d25_sum/results/2x2_metrics.csv"
OUT_FIG = REPO / "spec/spec05_d25_sum/figures"
OUT_RES = REPO / "spec/spec05_d25_sum/results"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]

sp03 = pd.read_csv(SPEC03)
sp05 = pd.read_csv(SPEC05)

# SPEC_03 xgb: use barrier_sum (comparable to SPEC_05 which sums 5 channels)
xgb_ref = sp03[sp03.model == "xgb"].set_index("channel")

def get_ref(ch):
    key = "barrier_sum" if ch == "barrier" else ch
    return float(xgb_ref.loc[key, "NMAE"])

def get_m(tag, ch):
    row = sp05[(sp05.variant == tag) & (sp05.channel == ch)]
    return float(row.NMAE.iloc[0])

channels_plot = CHANNELS + ["barrier"]
rows = []
for ch in channels_plot:
    ref = get_ref(ch)
    m0 = get_m("M0", ch); m1 = get_m("M1", ch)
    m2 = get_m("M2", ch); m3 = get_m("M3", ch)
    rows.append({
        "channel": ch,
        "spec03_xgb": ref,
        "M0_24d_plain": m0,
        "M1_25d_plain (+d25)": m1,
        "M2_24d_sum (+sum)": m2,
        "M3_25d_sum (+d25 +sum)": m3,
        "delta_M0_vs_ref": m0 - ref,
        "delta_M1_vs_ref": m1 - ref,
        "delta_M2_vs_ref": m2 - ref,
        "delta_M3_vs_ref": m3 - ref,
        "pct_improve_M3_vs_ref": 100.0 * (ref - m3) / ref,
    })
table = pd.DataFrame(rows)
table.to_csv(OUT_RES / "vs_spec03_xgb_table.csv", index=False)
print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ============ Fig 1: grouped bar chart ============
labels_bar = ["SPEC_03 xgb\n(24-d, ref)", "M0 (24-d)", "M1 (+d25)", "M2 (+sum)", "M3 (+d25 +sum)"]
colors = ["#7a7a7a", "#4b779a", "#c05e2b", "#4ba36c", "#8b3a62"]
values = np.array([[get_ref(ch), get_m("M0", ch), get_m("M1", ch),
                     get_m("M2", ch), get_m("M3", ch)] for ch in channels_plot])

x = np.arange(len(channels_plot)); w = 0.16
fig, ax = plt.subplots(figsize=(13, 5.5))
for i, (lbl, c) in enumerate(zip(labels_bar, colors)):
    ax.bar(x + (i - 2) * w, values[:, i], w, label=lbl, color=c,
           edgecolor="white", lw=0.4)
ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
ax.set_xticks(x); ax.set_xticklabels(channels_plot)
ax.set_ylabel("NMAE (5-fold pooled OOF)")
ax.set_title("SPEC_05 (M0..M3) vs SPEC_03 xgb baseline (m3 v7, 776 rxns)\n"
             "smaller = better; strain improves with d25, barrier improves with sum-consistency")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_FIG / "vs_spec03_xgb_bars.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============ Fig 2: delta bar chart ============
fig, ax = plt.subplots(figsize=(13, 5.5))
delta_labels = ["M0 - ref", "M1 - ref (+d25)", "M2 - ref (+sum)", "M3 - ref (+d25 +sum)"]
delta_colors = ["#4b779a", "#c05e2b", "#4ba36c", "#8b3a62"]
for i, (lbl, c) in enumerate(zip(delta_labels, delta_colors)):
    vals = [values[j, i + 1] - values[j, 0] for j in range(len(channels_plot))]
    ax.bar(x + (i - 1.5) * (w * 1.2), vals, w * 1.2, label=lbl, color=c,
           edgecolor="white", lw=0.4)
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(channels_plot)
ax.set_ylabel("delta NMAE (variant - SPEC_03 xgb)")
ax.set_title("Improvement vs SPEC_03 xgb baseline (negative = better)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_FIG / "vs_spec03_xgb_deltas.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nwrote {OUT_FIG / 'vs_spec03_xgb_bars.png'}")
print(f"wrote {OUT_FIG / 'vs_spec03_xgb_deltas.png'}")
print(f"wrote {OUT_RES / 'vs_spec03_xgb_table.csv'}")
