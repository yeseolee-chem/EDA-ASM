#!/usr/bin/env python3
"""SPEC13 Step 3 — three PNG figures at 300 dpi. English labels to avoid
Korean font issues on matplotlib default fontset.

Outputs:
  figures/fig1_topk_bar.png         Top-1 / Top-3 per channel with random baselines
  figures/fig2_topk_by_size.png     Top-1 vs candidate-group size, per channel
  figures/fig3_spearman_violin.png  Per-channel Spearman distribution
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval")

res = pd.read_csv(BASE / "artifacts" / "topk_results.csv")
pg = pd.read_csv(BASE / "artifacts" / "topk_per_group.csv")

NICE = {"d1_own_dft": "strain d1", "d2_own_dft": "strain d2",
        "elst_dft": "elst", "pauli_dft": "Pauli",
        "oi_dft": "orb.int.", "disp_dft": "disp", "cpcm_dft": "CPCM"}
res["label"] = res.channel.map(NICE)

# ---- Fig 1: Top-1 / Top-3 bar with random baselines ----
r = res.sort_values("top1", ascending=False)
x = np.arange(len(r)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, r.top1, w, label="Top-1", color="#2c6fbb")
ax.bar(x + w/2, r.top3, w, label="Top-3", color="#8fbce6")
ax.axhline(r.random_top1.iloc[0], ls="--", c="#c0392b", lw=1.2,
           label=f"random Top-1 ({r.random_top1.iloc[0]:.3f})")
ax.axhline(r.random_top3.iloc[0], ls=":", c="#c0392b", lw=1.2,
           label=f"random Top-3 ({r.random_top3.iloc[0]:.3f})")
for xi, (a, b) in enumerate(zip(r.top1, r.top3)):
    ax.text(xi - w/2, a + .015, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(xi + w/2, b + .015, f"{b:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(r.label, rotation=20)
ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy")
ax.set_title("Within-scaffold optimal-substituent retrieval  (mean 6.0 cand/group, n=720)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(BASE / "figures" / "fig1_topk_bar.png", dpi=300)
plt.close(fig)

# ---- Fig 2: Top-1 by candidate-group size, per channel ----
fig, ax = plt.subplots(figsize=(9, 5))
pg["bin"] = pd.cut(pg.n_cand, [2, 3, 4, 6, 8, 12, 20],
                   labels=["3", "4", "5-6", "7-8", "9-12", "13+"])
for ch, g in pg.groupby("channel"):
    m = g.groupby("bin", observed=True).top1.mean()
    ax.plot(m.index.astype(str), m.values, marker="o", label=NICE[ch])
base = pg.groupby("bin", observed=True).n_cand.mean()
ax.plot(base.index.astype(str), 1 / base.values, "k--", lw=1.5, label="random 1/n")
ax.set_xlabel("candidate group size")
ax.set_ylabel("Top-1 accuracy")
ax.set_title("Retrieval difficulty vs candidate-group size")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(BASE / "figures" / "fig2_topk_by_size.png", dpi=300)
plt.close(fig)

# ---- Fig 3: Spearman ρ violin per channel ----
fig, ax = plt.subplots(figsize=(9, 5))
order = res.sort_values("spearman", ascending=False).channel.tolist()
data = [pg[(pg.channel == c) & pg.spearman.notna()].spearman.values for c in order]
parts = ax.violinplot(data, showmedians=True, widths=.8)
for pc in parts["bodies"]:
    pc.set_facecolor("#2c6fbb"); pc.set_alpha(.55)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels([NICE[c] for c in order], rotation=20)
ax.axhline(0, c="k", lw=.8)
ax.set_ylabel("within-group Spearman ρ")
ax.set_title("Rank correlation within scaffold groups  (n=595)")
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(BASE / "figures" / "fig3_spearman_violin.png", dpi=300)
plt.close(fig)

print("figures saved: fig1_topk_bar.png, fig2_topk_by_size.png, fig3_spearman_violin.png")
