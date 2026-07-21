"""SPEC_11 - side-by-side comparison of spec06 (28-d) and spec11 (33-d).

Both spec06 and spec11 use the identical 783-rxn cohort and outer_folds.json,
so we can directly align per-reaction OOF predictions and compute:

  - NMAE / RMSE bars per channel + barrier: arm-1 (base) side-by-side
    (xgb_28d vs xgb_33d) and arm-2 (base + delta) side-by-side.
  - Delta-NMAE bar: (spec11 - spec06) per channel + barrier with 95% CI,
    reaction-level bootstrap.
  - Parity grid: 2 rows x 6 cols (spec06 xgb_28d, spec11 xgb_33d).
  - Descriptor gain scatter: xgb_28d test residual vs xgb_33d test residual
    per channel, colored by family.

Outputs -> spec/spec11_electronic_33d/figures/compare_28d_33d/
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
S06 = REPO / "spec/spec06_2step_xgb28_delta"
S11 = REPO / "spec/spec11_electronic_33d"
OUT = S11 / "figures/compare_28d_33d"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
COLOR_28 = "#4b779a"
COLOR_33 = "#a83232"
COLOR_DELTA = "#2e8b57"
B_BOOT = 1000
SEED = 42


# ---------- metrics ----------
def nmae(yt, yp, mad):
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def slope(yt, yp):
    a = yt - yt.mean(); b = yp - yp.mean(); d = float(np.sum(a * a))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss / (tot + 1e-12))


def bootstrap_ci(yt, yp, mad, metric="NMAE", B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if metric == "NMAE": stats.append(nmae(yt[idx], yp[idx], mad))
        elif metric == "RMSE": stats.append(rmse(yt[idx], yp[idx]))
    stats = np.sort(stats)
    lo, hi = float(stats[int(0.025 * B)]), float(stats[int(0.975 * B) - 1])
    point = nmae(yt, yp, mad) if metric == "NMAE" else rmse(yt, yp)
    return point, lo, hi


def bootstrap_pairwise_nmae(yt, yp1, yp2, mad, B=B_BOOT, seed=SEED):
    """NMAE(yp1) - NMAE(yp2), reaction-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        stats.append(nmae(yt[idx], yp1[idx], mad) - nmae(yt[idx], yp2[idx], mad))
    stats = np.sort(stats)
    lo, hi = float(stats[int(0.025 * B)]), float(stats[int(0.975 * B) - 1])
    point = nmae(yt, yp1, mad) - nmae(yt, yp2, mad)
    return point, lo, hi


# ---------- loaders ----------
def load_oof(path):
    """Return (rids, y_true, y_pred) with rids sorted for stable alignment."""
    df = pd.read_parquet(path).sort_values("reaction_id").reset_index(drop=True)
    rids = df["reaction_id"].tolist()
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    return rids, yt, yp


def align_two(a, b):
    """Given two (rids, yt, yp) tuples, return them aligned to intersection."""
    rids_a, yt_a, yp_a = a
    rids_b, yt_b, yp_b = b
    common = sorted(set(rids_a) & set(rids_b))
    idx_a = np.array([rids_a.index(r) for r in common])
    idx_b = np.array([rids_b.index(r) for r in common])
    return common, yt_a[idx_a], yp_a[idx_a], yt_b[idx_b], yp_b[idx_b]


# ---------- plots ----------
def bar_compare(mad_c, mad_bar, aligned, metric, ylabel, path, title):
    """4-arm bar plot: spec06 base, spec06 delta, spec11 base, spec11 delta."""
    channels = CHANNELS + ["barrier"]
    x = np.arange(len(channels))
    arms = [
        ("xgb_28d (28-d base)",     aligned["s06_base"],  "#4b779a", -1.5),
        ("xgb28 + delta",           aligned["s06_delta"], "#7ea6c6", -0.5),
        ("xgb_33d (33-d base)",     aligned["s11_base"],  "#a83232", +0.5),
        ("xgb33 + delta",           aligned["s11_delta"], "#d97070", +1.5),
    ]
    w = 0.22
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for label, (yt, yp), color, off in arms:
        pts, los, his = [], [], []
        for i, ch in enumerate(channels):
            if ch == "barrier":
                a = yt.sum(1); b = yp.sum(1); mad = mad_bar
            else:
                idx = CHANNELS.index(ch); a = yt[:, idx]; b = yp[:, idx]; mad = mad_c[idx]
            pt, lo, hi = bootstrap_ci(a, b, mad, metric=metric)
            pts.append(pt); los.append(max(pt - lo, 0)); his.append(max(hi - pt, 0))
        ax.bar(x + off * w, pts, w, yerr=[los, his], capsize=3,
               label=label, color=color, edgecolor="white", lw=0.4)
    if metric == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.6, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels, fontsize=11)
    ax.set_ylabel(f"{ylabel} (pooled OOF, 95% CI)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def head_to_head_bar(mad_c, mad_bar, aligned, path):
    """Delta NMAE = NMAE(33-d arm) - NMAE(28-d arm) with 95% CI."""
    channels = CHANNELS + ["barrier"]
    x = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(11, 5))

    def compute_delta(label28, label33, offset, color):
        yt28, yp28 = aligned[label28]; yt33, yp33 = aligned[label33]
        # yt28 == yt33 after align_common; use whichever
        pts, los, his = [], [], []
        for i, ch in enumerate(channels):
            if ch == "barrier":
                a = yt28.sum(1); p28 = yp28.sum(1); p33 = yp33.sum(1); mad = mad_bar
            else:
                idx = CHANNELS.index(ch)
                a = yt28[:, idx]; p28 = yp28[:, idx]; p33 = yp33[:, idx]; mad = mad_c[idx]
            pt, lo, hi = bootstrap_pairwise_nmae(a, p33, p28, mad)
            pts.append(pt); los.append(max(pt - lo, 0)); his.append(max(hi - pt, 0))
        ax.bar(x + offset, pts, 0.35, yerr=[los, his], capsize=3,
               color=color, edgecolor="white", lw=0.4,
               label=f"NMAE({label33}) - NMAE({label28})")

    compute_delta("s06_base",  "s11_base",  -0.19, COLOR_DELTA)
    compute_delta("s06_delta", "s11_delta", +0.19, "#7bbf7b")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(channels, fontsize=11)
    ax.set_ylabel("delta NMAE  (negative = 33-d better)", fontsize=11)
    ax.set_title("Head-to-head: spec11 (33-d) minus spec06 (28-d), pooled OOF", fontsize=12)
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parity_grid_compare(mad_c, mad_bar, aligned, path):
    """4 rows (arms) x 6 cols (channels+barrier) parity grid."""
    channels_plot = CHANNELS + ["barrier"]
    arms = [("xgb_28d",         "s06_base",  COLOR_28),
            ("xgb28 + delta",   "s06_delta", "#7ea6c6"),
            ("xgb_33d",         "s11_base",  COLOR_33),
            ("xgb33 + delta",   "s11_delta", "#d97070")]
    fig, axes = plt.subplots(len(arms), len(channels_plot),
                             figsize=(3.0 * len(channels_plot), 3.0 * len(arms)))
    for r_i, (label, key, color) in enumerate(arms):
        yt, yp = aligned[key]
        for c_i, ch in enumerate(channels_plot):
            ax = axes[r_i, c_i]
            if ch == "barrier":
                a = yt.sum(1); b = yp.sum(1); mad = mad_bar
            else:
                idx = CHANNELS.index(ch)
                a = yt[:, idx]; b = yp[:, idx]; mad = mad_c[idx]
            ax.scatter(a, b, s=6, c=color, alpha=0.55, edgecolor="none")
            lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.5)
            ax.text(0.03, 0.97,
                    f"NMAE={nmae(a, b, mad):.3f}\nR2={r2(a, b):.2f}\nslope={slope(a, b):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            if r_i == 0: ax.set_title(ch, fontsize=10)
            if c_i == 0: ax.set_ylabel(f"{label}\ny_pred", fontsize=9)
            if r_i == len(arms) - 1: ax.set_xlabel("y_true", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def residual_scatter(aligned, path):
    """5-channel scatter: xgb_28d abs residual vs xgb_33d abs residual per rxn.
    Points BELOW y=x mean 33-d is better on that rxn.
    """
    yt, yp28 = aligned["s06_base"]
    _,  yp33 = aligned["s11_base"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for c_i, ch in enumerate(CHANNELS):
        r28 = np.abs(yt[:, c_i] - yp28[:, c_i])
        r33 = np.abs(yt[:, c_i] - yp33[:, c_i])
        ax = axes[c_i]
        ax.scatter(r28, r33, s=8, alpha=0.5, color="#333")
        lo, hi = 0, float(max(r28.max(), r33.max()) * 1.05)
        ax.plot([lo, hi], [lo, hi], "--", color="red", lw=0.7)
        n_better = int((r33 < r28).sum())
        ax.text(0.03, 0.97,
                f"{ch}\n33-d better: {n_better}/{len(r28)}\n"
                f"MAE_28={r28.mean():.3f}\nMAE_33={r33.mean():.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel("|residual| xgb_28d")
        if c_i == 0:
            ax.set_ylabel("|residual| xgb_33d")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-reaction |residual| comparison (base only), below y=x means 33-d is better",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[compare] loading OOFs")
    s06_base_  = load_oof(S06 / "results/xgb_28d_oof.parquet")
    s06_delta_ = load_oof(S06 / "results/pooled_oof.parquet")
    s11_base_  = load_oof(S11 / "results/xgb_33d_oof.parquet")
    s11_delta_ = load_oof(S11 / "results/pooled_oof.parquet")

    # Align on common rids across all four
    common = sorted(set(s06_base_[0]) & set(s06_delta_[0]) &
                    set(s11_base_[0]) & set(s11_delta_[0]))
    def pick(t):
        rids, yt, yp = t
        idx = np.array([rids.index(r) for r in common])
        return yt[idx], yp[idx]
    aligned = {
        "s06_base":  pick(s06_base_),
        "s06_delta": pick(s06_delta_),
        "s11_base":  pick(s11_base_),
        "s11_delta": pick(s11_delta_),
    }
    yt_ref = aligned["s06_base"][0]  # all four y_true match
    mad_c = np.array([np.mean(np.abs(yt_ref[:, i] - yt_ref[:, i].mean())) for i in range(5)])
    mad_bar = float(np.mean(np.abs(yt_ref.sum(1) - yt_ref.sum(1).mean())))
    print(f"[compare] aligned on {len(common)} rxns")

    # 1) NMAE 4-arm bars
    bar_compare(mad_c, mad_bar, aligned, "NMAE", "NMAE",
                OUT / "nmae_bars_28d_vs_33d.png",
                "spec06 (28-d) vs spec11 (33-d), pooled OOF NMAE")
    # 2) RMSE 4-arm bars
    bar_compare(mad_c, mad_bar, aligned, "RMSE", "RMSE (kcal/mol)",
                OUT / "rmse_bars_28d_vs_33d.png",
                "spec06 (28-d) vs spec11 (33-d), pooled OOF RMSE")
    # 3) head-to-head delta bar
    head_to_head_bar(mad_c, mad_bar, aligned,
                     OUT / "head_to_head_28d_vs_33d.png")
    # 4) 4-arm parity grid
    parity_grid_compare(mad_c, mad_bar, aligned,
                        OUT / "parity_grid_28d_vs_33d.png")
    # 5) per-rxn residual scatter (base arms only)
    residual_scatter(aligned, OUT / "residual_scatter_base_28d_vs_33d.png")

    # print summary
    print("\n=== NMAE summary (pooled OOF) ===")
    print(f"{'channel':10s}  {'xgb_28d':>10s}  {'xgb_33d':>10s}  {'delta':>10s}")
    for i, ch in enumerate(CHANNELS + ["barrier"]):
        if ch == "barrier":
            a = yt_ref.sum(1); p28 = aligned['s06_base'][1].sum(1); p33 = aligned['s11_base'][1].sum(1); mad = mad_bar
        else:
            idx = CHANNELS.index(ch)
            a = yt_ref[:, idx]; p28 = aligned['s06_base'][1][:, idx]; p33 = aligned['s11_base'][1][:, idx]; mad = mad_c[idx]
        n28 = nmae(a, p28, mad); n33 = nmae(a, p33, mad)
        print(f"{ch:10s}  {n28:10.3f}  {n33:10.3f}  {n33-n28:+10.4f}")
    print(f"\n[compare] wrote 5 figures to {OUT}")


if __name__ == "__main__":
    main()
