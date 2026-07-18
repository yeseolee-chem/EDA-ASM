"""SPEC_09 — aggregate the 100-cell per-family OOF and compare against
family-restricted xgb_28d base-only. Two arms only:

  xgb_28d          : per-channel XGB on 28-d descriptors, trained WITHIN each
                     family using the same per-family 5-fold KFold split
                     (splits/family_folds/{fam}_outer_folds.json). Base only.
  xgb28_delta      : this spec — b (xgb_28d family cross-fit OOF) + δ
                     (ModelM1Delta), 5 members averaged per (family, fold, rxn).

Writes:
  results/family_restricted/
    pooled_oof.parquet          xgb28_delta per-rxn predictions (100-cell mean)
    xgb_28d_oof.parquet         xgb_28d family-restricted base pooled OOF
    metrics.csv                 per (arm, family, channel, metric) + 95% CI
    head_to_head.csv            NMAE(xgb28_delta) - NMAE(xgb_28d) per (family, channel)
    leaderboard.csv             wide NMAE
    summary.md
  figures/family_restricted/
    nmae_bars_all.png           4-panel (family × subplot)
    rmse_bars_all.png
    parity_grid_all.png         4 family × 6 channel
    {family}_nmae.png           per-family individual bars
    {family}_rmse.png
    {family}_parity.png         2 arm × 6 channel
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec09_per_family_xgb28_delta"
sys.path.insert(0, str(SPEC / "code"))
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
from descriptors28 import build_X28  # noqa: E402
from baselines import fit_xgb, predict_xgb  # noqa: E402

BUNDLE_PT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt")
FAM_SPLITS_DIR = SPEC / "splits/family_folds"
OOF_ROOT = SPEC / "oof"
OUT_RES = SPEC / "results/family_restricted"
OUT_FIG = SPEC / "figures/family_restricted"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

FAMILIES = ["dipolar", "qmrxn20_e2", "qmrxn20_sn2", "rgd1"]
CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARM_COLORS = {"xgb_28d": "#4b779a", "xgb28_delta": "#a83232"}
ARM_LABELS = {"xgb_28d": "xgb_28d (family base)", "xgb28_delta": "xgb28 + δ (family)"}
B_BOOT = 1000
SEED = 42


# ---------- metrics ----------

def nmae(yt, yp, mad):
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss / (tot + 1e-12))


def slope(yt, yp):
    a = yt - yt.mean(); b = yp - yp.mean(); d = float(np.sum(a * a))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def bootstrap_ci(yt, yp, mad, metric, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if metric == "NMAE": stats.append(nmae(yt[idx], yp[idx], mad))
        elif metric == "RMSE": stats.append(rmse(yt[idx], yp[idx]))
        else: stats.append(r2(yt[idx], yp[idx]))
    stats = np.sort(stats)
    lo, hi = float(stats[int(0.025 * B)]), float(stats[int(0.975 * B) - 1])
    if metric == "NMAE": pt = nmae(yt, yp, mad)
    elif metric == "RMSE": pt = rmse(yt, yp)
    else: pt = r2(yt, yp)
    return pt, lo, hi


def bootstrap_pairwise(yt, yp1, yp2, mad, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        stats.append(nmae(yt[idx], yp1[idx], mad) - nmae(yt[idx], yp2[idx], mad))
    stats = np.sort(stats)
    return (nmae(yt, yp1, mad) - nmae(yt, yp2, mad),
            float(stats[int(0.025 * B)]),
            float(stats[int(0.975 * B) - 1]))


# ---------- data loaders ----------

def load_delta_pooled():
    """Load the 100 cells; average across members per (family, fold, rxn)."""
    rows = []
    for family in FAMILIES:
        for f in (OOF_ROOT / family).glob("fold*/member*.json"):
            d = json.load(open(f))
            for i, r in enumerate(d["reaction_ids"]):
                row = {"family": d["family"], "fold": d["fold"],
                       "member": d["member"], "reaction_id": r}
                for c in CHANNELS:
                    row[f"y_true_{c}"] = float(d[f"y_true_{c}"][i])
                    row[f"y_pred_{c}"] = float(d[f"y_pred_{c}"][i])
                    row[f"b_test_{c}"] = float(d[f"b_test_{c}"][i])
                    row[f"delta_test_{c}"] = float(d[f"delta_test_{c}"][i])
                rows.append(row)
    df = pd.DataFrame(rows)
    df = df.groupby(["family", "fold", "reaction_id"], as_index=False).mean(numeric_only=True)
    df.to_parquet(OUT_RES / "pooled_oof.parquet", index=False)
    return df


def compute_family_base_oof():
    """For each family, run xgb_28d 5-fold OOF using the same
    splits/family_folds/{fam}_outer_folds.json. Fresh fit inside each family."""
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    all_rids = np.asarray(b["reaction_ids"])
    X24_all = b["descriptors"].numpy().astype(np.float64)
    Y_all = b["labels"].numpy().astype(np.float64)
    X28_all, _ok = build_X28(all_rids, X24_all)
    r2i = {r: i for i, r in enumerate(all_rids)}

    rows = []
    for family in FAMILIES:
        with open(FAM_SPLITS_DIR / f"{family}_outer_folds.json") as fh:
            fam_folds = json.load(fh)
        fam_rids = np.asarray(fam_folds["all_rids"])
        fam_pos = np.array([r2i[r] for r in fam_rids])
        X_fam = X28_all[fam_pos]
        Y_fam = Y_all[fam_pos]
        fam_r2i = {r: i for i, r in enumerate(fam_rids)}
        for fkey in sorted(fam_folds["folds"], key=int):
            tr_rids = fam_folds["folds"][fkey]["train"]
            te_rids = fam_folds["folds"][fkey]["test"]
            tr = np.array([fam_r2i[r] for r in tr_rids])
            te = np.array([fam_r2i[r] for r in te_rids])
            model = fit_xgb(X_fam[tr], Y_fam[tr])
            yp = predict_xgb(model, X_fam[te])
            for i_te, idx in enumerate(te):
                row = {"family": family, "fold": int(fkey),
                       "reaction_id": fam_rids[idx]}
                for i_c, c in enumerate(CHANNELS):
                    row[f"y_true_{c}"] = float(Y_fam[idx, i_c])
                    row[f"y_pred_{c}"] = float(yp[i_te, i_c])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_RES / "xgb_28d_oof.parquet", index=False)
    return df


# ---------- alignment ----------

def slice_and_stack(df, family, cols_pred):
    d = df[df.family == family].copy().sort_values("reaction_id")
    yt = d[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = d[cols_pred].to_numpy()
    rids = d["reaction_id"].tolist()
    return rids, yt, yp


def family_metric_rows(family, arm_name, yt, yp, mad_c, mad_bar):
    rows = []
    for i, ch in enumerate(CHANNELS):
        for metric in ["NMAE", "RMSE", "R2"]:
            pt, lo, hi = bootstrap_ci(yt[:, i], yp[:, i], mad_c[i], metric)
            rows.append({"arm": arm_name, "family": family, "channel": ch,
                         "metric": metric, "point": pt,
                         "ci_low": lo, "ci_high": hi})
        rows.append({"arm": arm_name, "family": family, "channel": ch,
                     "metric": "slope",
                     "point": slope(yt[:, i], yp[:, i]),
                     "ci_low": np.nan, "ci_high": np.nan})
    for metric in ["NMAE", "RMSE", "R2"]:
        pt, lo, hi = bootstrap_ci(yt.sum(1), yp.sum(1), mad_bar, metric)
        rows.append({"arm": arm_name, "family": family, "channel": "barrier",
                     "metric": metric, "point": pt,
                     "ci_low": lo, "ci_high": hi})
    rows.append({"arm": arm_name, "family": family, "channel": "barrier",
                 "metric": "slope", "point": slope(yt.sum(1), yp.sum(1)),
                 "ci_low": np.nan, "ci_high": np.nan})
    return rows


# ---------- plotting ----------

def single_family_bars(ax, family, arm_metrics, metric_key, mad_scale=None,
                       ylabel="", show_meanline=False, show_legend=False):
    channels_plot = CHANNELS + ["barrier"]
    x = np.arange(len(channels_plot)); w = 0.36
    for i, arm_name in enumerate(["xgb_28d", "xgb28_delta"]):
        pts, los, his = [], [], []
        for ch in channels_plot:
            pt, lo, hi = arm_metrics[arm_name][(ch, metric_key)]
            pts.append(pt)
            los.append(max(pt - lo, 0))
            his.append(max(hi - pt, 0))
        ax.bar(x + (i - 0.5) * w, pts, w, yerr=[los, his], capsize=3,
               label=ARM_LABELS[arm_name], color=ARM_COLORS[arm_name],
               edgecolor="white", lw=0.4)
    if show_meanline and metric_key == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(channels_plot, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if show_legend:
        ax.legend(fontsize=9, loc="upper left")


FAM_COLORS = {
    "dipolar":     "#4c72b0",
    "qmrxn20_e2":  "#dd8452",
    "qmrxn20_sn2": "#55a868",
    "rgd1":        "#c44e52",
}


def combined_bars(family_data, metric_key, ylabel, out_path):
    """Single-axis grouped bar: 6 channels × (4 families × 2 arms) = 48 bars.

    Family = colour; base = hollow (hatched, edge only), δ = solid fill.
    """
    channels_plot = CHANNELS + ["barrier"]
    n_ch = len(channels_plot); n_fam = len(FAMILIES); n_arm = 2
    group_w = 0.85
    bar_w = group_w / (n_fam * n_arm)
    x = np.arange(n_ch)

    fig, ax = plt.subplots(figsize=(15, 6))
    for fi, family in enumerate(FAMILIES):
        colr = FAM_COLORS[family]
        m = family_data[family]["metrics"]
        for ai, arm_name in enumerate(["xgb_28d", "xgb28_delta"]):
            offset = (fi * n_arm + ai - (n_fam * n_arm - 1) / 2) * bar_w
            pts, los, his = [], [], []
            for ch in channels_plot:
                pt, lo, hi = m[arm_name][(ch, metric_key)]
                pts.append(pt)
                los.append(max(pt - lo, 0))
                his.append(max(hi - pt, 0))
            if arm_name == "xgb_28d":
                ax.bar(x + offset, pts, bar_w, yerr=[los, his], capsize=1.5,
                       facecolor="none", edgecolor=colr, hatch="////",
                       linewidth=1.0)
            else:
                ax.bar(x + offset, pts, bar_w, yerr=[los, his], capsize=1.5,
                       facecolor=colr, edgecolor="white", linewidth=0.4)
    if metric_key == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)

    ax.set_xticks(x); ax.set_xticklabels(channels_plot, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # 2-tier legend: family colours + arm hatching
    from matplotlib.patches import Patch
    fam_handles = [Patch(facecolor=FAM_COLORS[f], edgecolor="white", label=f)
                   for f in FAMILIES]
    arm_handles = [
        Patch(facecolor="none", edgecolor="gray", hatch="////",
              label=ARM_LABELS["xgb_28d"]),
        Patch(facecolor="gray", edgecolor="white",
              label=ARM_LABELS["xgb28_delta"]),
    ]
    leg1 = ax.legend(handles=fam_handles, title="family",
                     loc="upper left", fontsize=9, title_fontsize=9,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=arm_handles, title="arm",
              loc="upper right", fontsize=9, title_fontsize=9,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_family_bar(family, arm_metrics, n, metric_key, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    single_family_bars(ax, family, arm_metrics, metric_key,
                       ylabel=ylabel, show_meanline=True, show_legend=True)
    ax.set_title(f"{family}  (n={n})   {metric_key}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parity_grid_all(family_data, out_path):
    channels_plot = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(len(FAMILIES), len(channels_plot),
                             figsize=(3.2 * len(channels_plot), 3.2 * len(FAMILIES)))
    for r_i, family in enumerate(FAMILIES):
        d = family_data[family]
        for c_i, ch in enumerate(channels_plot):
            ax = axes[r_i, c_i]
            for arm_name in ["xgb_28d", "xgb28_delta"]:
                yt = d["yt"]; yp = d["yp"][arm_name]
                if ch == "barrier":
                    a = yt.sum(1); b = yp.sum(1); mad = d["mad_bar"]
                else:
                    ic = CHANNELS.index(ch); a = yt[:, ic]; b = yp[:, ic]; mad = d["mad_c"][ic]
                ax.scatter(a, b, s=7, c=ARM_COLORS[arm_name],
                           alpha=0.55, edgecolor="none",
                           label=ARM_LABELS[arm_name] if (r_i == 0 and c_i == 0) else None)
            all_a = yt.sum(1) if ch == "barrier" else yt[:, CHANNELS.index(ch)]
            lo = float(all_a.min()); hi = float(all_a.max())
            ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.6)
            if r_i == 0:
                ax.set_title(ch, fontsize=10)
            if c_i == 0:
                ax.set_ylabel(f"{family}\ny_pred", fontsize=9)
            if r_i == len(FAMILIES) - 1:
                ax.set_xlabel("y_true", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_family_parity(family, d, out_path):
    channels_plot = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(2, len(channels_plot),
                             figsize=(3.2 * len(channels_plot), 6.4))
    for r_i, arm_name in enumerate(["xgb_28d", "xgb28_delta"]):
        for c_i, ch in enumerate(channels_plot):
            ax = axes[r_i, c_i]
            yt = d["yt"]; yp = d["yp"][arm_name]
            if ch == "barrier":
                a = yt.sum(1); b = yp.sum(1); mad = d["mad_bar"]
            else:
                ic = CHANNELS.index(ch); a = yt[:, ic]; b = yp[:, ic]; mad = d["mad_c"][ic]
            ax.scatter(a, b, s=8, c=ARM_COLORS[arm_name], alpha=0.6, edgecolor="none")
            lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.6)
            ax.text(0.03, 0.97,
                    f"NMAE={nmae(a, b, mad):.3f}\nR²={r2(a, b):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8)
            if r_i == 0:
                ax.set_title(ch, fontsize=10)
            if c_i == 0:
                ax.set_ylabel(f"{ARM_LABELS[arm_name]}\ny_pred", fontsize=9)
            if r_i == 1:
                ax.set_xlabel("y_true", fontsize=9)
    fig.suptitle(f"{family}  (n={d['n']})", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main():
    print("[spec09] pooling 100 delta cells (5 members averaged per fold)…", flush=True)
    delta_df = load_delta_pooled()
    print(f"  delta pooled: {len(delta_df)} rows (rxns × 1)", flush=True)

    print("[spec09] computing xgb_28d family-restricted base OOF…", flush=True)
    base_df = compute_family_base_oof()
    print(f"  base pooled:  {len(base_df)} rows", flush=True)

    all_metric_rows = []
    all_head_rows = []
    family_data = {}
    for family in FAMILIES:
        rids_d, yt_d, yp_d = slice_and_stack(
            delta_df, family, [f"y_pred_{c}" for c in CHANNELS])
        rids_b, yt_b, yp_b = slice_and_stack(
            base_df, family, [f"y_pred_{c}" for c in CHANNELS])
        assert rids_d == rids_b, f"{family}: rid ordering mismatch between arms"
        assert np.allclose(yt_d, yt_b), f"{family}: y_true mismatch between arms"
        yt = yt_d
        n = len(rids_d)
        mad_c = np.array([np.mean(np.abs(yt[:, i] - yt[:, i].mean())) for i in range(5)])
        mad_bar = float(np.mean(np.abs(yt.sum(1) - yt.sum(1).mean())))

        arm_metrics = {"xgb_28d": {}, "xgb28_delta": {}}
        for arm_name, yp in [("xgb_28d", yp_b), ("xgb28_delta", yp_d)]:
            rows = family_metric_rows(family, arm_name, yt, yp, mad_c, mad_bar)
            all_metric_rows.extend(rows)
            for r in rows:
                if r["metric"] in ("NMAE", "RMSE"):
                    arm_metrics[arm_name][(r["channel"], r["metric"])] = (
                        r["point"], r["ci_low"], r["ci_high"])

        # head-to-head Δ NMAE within family
        for i, ch in enumerate(CHANNELS):
            pt, lo, hi = bootstrap_pairwise(yt[:, i], yp_d[:, i], yp_b[:, i], mad_c[i])
            all_head_rows.append({"family": family, "channel": ch,
                                  "delta": "NMAE(xgb28_delta) - NMAE(xgb_28d)",
                                  "point": pt, "ci_low": lo, "ci_high": hi, "n": n})
        pt, lo, hi = bootstrap_pairwise(yt.sum(1), yp_d.sum(1), yp_b.sum(1), mad_bar)
        all_head_rows.append({"family": family, "channel": "barrier",
                              "delta": "NMAE(xgb28_delta) - NMAE(xgb_28d)",
                              "point": pt, "ci_low": lo, "ci_high": hi, "n": n})

        family_data[family] = {
            "n": n, "yt": yt,
            "yp": {"xgb_28d": yp_b, "xgb28_delta": yp_d},
            "mad_c": mad_c, "mad_bar": mad_bar,
            "metrics": arm_metrics,
        }

    df_m = pd.DataFrame(all_metric_rows); df_h = pd.DataFrame(all_head_rows)
    df_m.to_csv(OUT_RES / "metrics.csv", index=False)
    df_h.to_csv(OUT_RES / "head_to_head.csv", index=False)

    # leaderboard
    lb = []
    for family in FAMILIES:
        for arm in ["xgb_28d", "xgb28_delta"]:
            row = {"family": family, "arm": arm}
            for ch in CHANNELS + ["barrier"]:
                row[ch] = family_data[family]["metrics"][arm][(ch, "NMAE")][0]
            lb.append(row)
    pd.DataFrame(lb).to_csv(OUT_RES / "leaderboard.csv", index=False)

    # combined + per-family figures
    combined_bars(family_data, "NMAE",
                  "NMAE (family-local MAD, 95% CI)",
                  OUT_FIG / "nmae_bars_all.png")
    combined_bars(family_data, "RMSE",
                  "RMSE (kcal/mol, 95% CI)",
                  OUT_FIG / "rmse_bars_all.png")
    parity_grid_all(family_data, OUT_FIG / "parity_grid_all.png")
    for family in FAMILIES:
        d = family_data[family]
        per_family_bar(family, d["metrics"], d["n"], "NMAE",
                       "NMAE (family-local MAD, 95% CI)",
                       OUT_FIG / f"{family}_nmae.png")
        per_family_bar(family, d["metrics"], d["n"], "RMSE",
                       "RMSE (kcal/mol, 95% CI)",
                       OUT_FIG / f"{family}_rmse.png")
        per_family_parity(family, d, OUT_FIG / f"{family}_parity.png")

    # summary.md
    lines = [
        "# SPEC_09 — per-family 2-step (xgb_28d base + δ) comparison",
        "",
        "- Two arms trained **within** each family, both using the same",
        "  5-fold KFold split (splits/family_folds/{family}_outer_folds.json, seed=42).",
        "  - `xgb_28d` (base): per-channel XGB on 28-d descriptors, 5-fold pooled OOF.",
        "  - `xgb28_delta` (b + δ): b = xgb_28d cross-fit OOF; δ = ModelM1Delta on",
        "    residuals; ŷ = b_full(test) + δ(test). 5 members averaged per (family, fold, rxn).",
        f"- Bootstrap B={B_BOOT}, reaction-level resampling within each family.",
        "- NMAE normalizer = family-local MAD = mean|y − ȳ_family|.",
        "",
    ]
    for family in FAMILIES:
        d = family_data[family]; m = d["metrics"]
        lines += [f"## {family}   (n = {d['n']})", "",
                  "| channel | xgb_28d NMAE | xgb28+δ NMAE | Δ NMAE (95% CI) |",
                  "|---|---|---|---|"]
        for ch in CHANNELS + ["barrier"]:
            bp, blo, bhi = m["xgb_28d"][(ch, "NMAE")]
            dp, dlo, dhi = m["xgb28_delta"][(ch, "NMAE")]
            hh = df_h[(df_h.family == family) & (df_h.channel == ch)].iloc[0]
            crosses = (hh.ci_low < 0) and (hh.ci_high > 0)
            mark = "  " if crosses else (" ✓" if hh.point < 0 else " ✗")
            lines.append(
                f"| {ch} | {bp:.3f} [{blo:.3f}, {bhi:.3f}] "
                f"| {dp:.3f} [{dlo:.3f}, {dhi:.3f}] "
                f"| {hh.point:+.3f} [{hh.ci_low:+.3f}, {hh.ci_high:+.3f}]{mark} |"
            )
        lines += ["",
                  "| channel | xgb_28d RMSE (kcal/mol) | xgb28+δ RMSE |",
                  "|---|---|---|"]
        for ch in CHANNELS + ["barrier"]:
            bp, blo, bhi = m["xgb_28d"][(ch, "RMSE")]
            dp, dlo, dhi = m["xgb28_delta"][(ch, "RMSE")]
            lines.append(
                f"| {ch} | {bp:.3f} [{blo:.3f}, {bhi:.3f}] "
                f"| {dp:.3f} [{dlo:.3f}, {dhi:.3f}] |"
            )
        lines.append("")
    lines += ["## Files",
              "- pooled_oof.parquet (xgb28+δ), xgb_28d_oof.parquet (base)",
              "- metrics.csv, head_to_head.csv, leaderboard.csv",
              "- figures: nmae_bars_all.png, rmse_bars_all.png, parity_grid_all.png,",
              "  {family}_nmae.png, {family}_rmse.png, {family}_parity.png"]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
