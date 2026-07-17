"""SPEC_06 — pool the 25-cell OOF JSONs → 2-arm comparison vs xgb_28d base-only.

Two arms only (per user request):
  - xgb_28d          : per-channel XGB on the 28-d descriptor set (base, no δ)
                       computed in-place using the same outer_folds.json (5-fold
                       stratified) so both arms cover the identical 783-rxn OOF.
  - xgb28_delta      : this spec — b (xgb_28d cross-fit OOF) + δ (MACE-OFF23 CA),
                       averaged over 5 members per fold.

Writes:
  results/pooled_oof.parquet     (xgb28_delta pooled OOF, per rxn)
  results/xgb_28d_oof.parquet    (xgb_28d base-only pooled OOF, per rxn)
  results/metrics.csv            per (arm, channel, metric) with 95% CI
  results/head_to_head.csv       NMAE(xgb28_delta) − NMAE(xgb_28d)
  results/leaderboard.csv        wide NMAE table
  results/summary.md             report
  figures/nmae_bars.png          per-channel + barrier NMAE bars w/ 95% CI
  figures/rmse_bars.png          per-channel + barrier RMSE bars w/ 95% CI
  figures/parity_grid.png        per-channel + barrier scatter (2 rows)
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
SPEC = REPO / "spec/spec06_2step_xgb28_delta"
sys.path.insert(0, str(SPEC / "code"))
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
from descriptors28 import build_X28  # noqa: E402
from baselines import fit_xgb, predict_xgb  # noqa: E402

BUNDLE_PT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt")
FOLDS_JSON = SPEC / "splits/outer_folds.json"
OOF_ROOT = SPEC / "oof"
OUT_RES = SPEC / "results"
OUT_FIG = SPEC / "figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARM_COLORS = {"xgb_28d": "#4b779a", "xgb28_delta": "#a83232"}
ARM_LABELS = {"xgb_28d": "xgb_28d (base)", "xgb28_delta": "xgb28 + δ (this spec)"}
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


def bootstrap_ci(yt, yp, mad, metric="NMAE", B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if metric == "NMAE": stats.append(nmae(yt[idx], yp[idx], mad))
        elif metric == "RMSE": stats.append(rmse(yt[idx], yp[idx]))
        elif metric == "R2":   stats.append(r2(yt[idx], yp[idx]))
    stats = np.sort(stats)
    lo, hi = float(stats[int(0.025 * B)]), float(stats[int(0.975 * B) - 1])
    if metric == "NMAE": point = nmae(yt, yp, mad)
    elif metric == "RMSE": point = rmse(yt, yp)
    else: point = r2(yt, yp)
    return point, lo, hi


def bootstrap_pairwise(yt, yp1, yp2, mad, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        stats.append(nmae(yt[idx], yp1[idx], mad) - nmae(yt[idx], yp2[idx], mad))
    stats = np.sort(stats)
    lo, hi = float(stats[int(0.025 * B)]), float(stats[int(0.975 * B) - 1])
    point = nmae(yt, yp1, mad) - nmae(yt, yp2, mad)
    return point, lo, hi


# ---------- data loaders ----------

def load_xgb28_delta():
    """Pool spec06 OOF JSONs across (fold, member); average members per rxn."""
    subdir = OOF_ROOT / "xgb28_delta"
    rows = []
    for f in subdir.glob("fold*/member*.json"):
        d = json.load(open(f))
        for i, r in enumerate(d["reaction_ids"]):
            row = {"reaction_id": r, "fold": d["fold"], "member": d["member"]}
            for c in CHANNELS:
                row[f"y_true_{c}"] = float(d[f"y_true_{c}"][i])
                row[f"y_pred_{c}"] = float(d[f"y_pred_{c}"][i])
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.groupby(["fold", "reaction_id"], as_index=False).mean(numeric_only=True)
    rids = df["reaction_id"].tolist()
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    df.to_parquet(OUT_RES / "pooled_oof.parquet", index=False)
    return rids, yt, yp


def compute_xgb28_baseonly_oof():
    """Compute xgb_28d base-only pooled OOF over the SAME outer_folds.json so
    both arms are on identical 783-rxn coverage. Emits per-rxn parquet."""
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy().astype(np.float64)
    Y = b["labels"].numpy().astype(np.float64)
    X28, _ok = build_X28(rids, X24)
    r2i = {r: i for i, r in enumerate(rids)}
    folds = json.load(open(FOLDS_JSON))

    rows = []
    for fkey in sorted(folds, key=int):
        tr = np.array([r2i[r] for r in folds[fkey]["train"]])
        te = np.array([r2i[r] for r in folds[fkey]["test"]])
        m = fit_xgb(X28[tr], Y[tr])
        yp = predict_xgb(m, X28[te])
        for i_te, idx in enumerate(te):
            row = {"reaction_id": rids[idx], "fold": int(fkey)}
            for i_c, c in enumerate(CHANNELS):
                row[f"y_true_{c}"] = float(Y[idx, i_c])
                row[f"y_pred_{c}"] = float(yp[i_te, i_c])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_RES / "xgb_28d_oof.parquet", index=False)
    return (df["reaction_id"].tolist(),
            df[[f"y_true_{c}" for c in CHANNELS]].to_numpy(),
            df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy())


def align_two(reference, other):
    """Align 'other' arm to 'reference' rid ordering."""
    ref_rids, ref_yt, ref_yp = reference
    rids, yt, yp = other
    r2i = {r: i for i, r in enumerate(rids)}
    idx = np.array([r2i[r] for r in ref_rids if r in r2i])
    if len(idx) != len(ref_rids):
        raise RuntimeError(
            f"align: only {len(idx)}/{len(ref_rids)} rids match — cohort mismatch")
    return ref_yt, {"xgb28_delta": (ref_yt, ref_yp),
                    "xgb_28d":     (yt[idx], yp[idx])}


def make_metric_rows(arm_name, yt, yp, mad_c, mad_bar):
    rows = []
    for i, ch in enumerate(CHANNELS):
        for metric in ["NMAE", "RMSE", "R2"]:
            pt, lo, hi = bootstrap_ci(yt[:, i], yp[:, i], mad_c[i], metric=metric)
            rows.append({"arm": arm_name, "channel": ch, "metric": metric,
                         "point": pt, "ci_low": lo, "ci_high": hi})
        rows.append({"arm": arm_name, "channel": ch, "metric": "slope",
                     "point": slope(yt[:, i], yp[:, i]),
                     "ci_low": np.nan, "ci_high": np.nan})
    for metric in ["NMAE", "RMSE", "R2"]:
        pt, lo, hi = bootstrap_ci(yt.sum(1), yp.sum(1), mad_bar, metric=metric)
        rows.append({"arm": arm_name, "channel": "barrier", "metric": metric,
                     "point": pt, "ci_low": lo, "ci_high": hi})
    rows.append({"arm": arm_name, "channel": "barrier", "metric": "slope",
                 "point": slope(yt.sum(1), yp.sum(1)),
                 "ci_low": np.nan, "ci_high": np.nan})
    return rows


def bar_plot(df, metric, ylabel, path, arms_plot):
    channels_plot = CHANNELS + ["barrier"]
    x = np.arange(len(channels_plot)); w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, arm_name in enumerate(arms_plot):
        pts, los, his = [], [], []
        for ch in channels_plot:
            row = df[(df.arm == arm_name) & (df.channel == ch) & (df.metric == metric)].iloc[0]
            pts.append(row.point)
            los.append(max(row.point - row.ci_low, 0))
            his.append(max(row.ci_high - row.point, 0))
        ax.bar(x + (i - 0.5) * w, pts, w, yerr=[los, his], capsize=4,
               label=ARM_LABELS[arm_name], color=ARM_COLORS[arm_name],
               edgecolor="white", lw=0.5)
    if metric == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel(f"{ylabel} (pooled OOF, 95% CI)")
    ax.legend(fontsize=10, loc="best"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parity_grid(aligned, mad_c, mad_bar, path):
    arms_plot = ["xgb_28d", "xgb28_delta"]
    channels_plot = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(len(arms_plot), len(channels_plot),
                             figsize=(3.4 * len(channels_plot), 3.4 * len(arms_plot)))
    for r_i, arm_name in enumerate(arms_plot):
        yt, yp = aligned[arm_name]
        colr = ARM_COLORS[arm_name]
        for c_i, ch in enumerate(channels_plot):
            ax = axes[r_i, c_i]
            if ch == "barrier":
                a = yt.sum(1); b = yp.sum(1); mad = mad_bar
            else:
                i_ = CHANNELS.index(ch); a = yt[:, i_]; b = yp[:, i_]; mad = mad_c[i_]
            ax.scatter(a, b, s=8, c=colr, alpha=0.6, edgecolor="none")
            lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.6)
            ax.text(0.03, 0.97,
                    f"NMAE={nmae(a, b, mad):.3f}\nR²={r2(a, b):.2f}\nslope={slope(a, b):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8)
            if r_i == 0:
                ax.set_title(ch, fontsize=11)
            if c_i == 0:
                ax.set_ylabel(f"{ARM_LABELS[arm_name]}\ny_pred", fontsize=10)
            if r_i == len(arms_plot) - 1:
                ax.set_xlabel("y_true", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ours = load_xgb28_delta()
    if ours is None:
        raise SystemExit("no spec06 OOF JSONs found — nothing to aggregate")
    print(f"[spec06] pooled OOF: {len(ours[0])} rxns", flush=True)

    print("[spec06] computing xgb_28d base-only OOF over same folds…", flush=True)
    base = compute_xgb28_baseonly_oof()

    yt_ref, aligned = align_two(ours, base)
    mad_c = np.array([np.mean(np.abs(yt_ref[:, i] - yt_ref[:, i].mean())) for i in range(5)])
    mad_bar = float(np.mean(np.abs(yt_ref.sum(1) - yt_ref.sum(1).mean())))

    # per-arm metrics
    all_rows = []
    for arm_name, (yt_a, yp_a) in aligned.items():
        all_rows.extend(make_metric_rows(arm_name, yt_a, yp_a, mad_c, mad_bar))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_RES / "metrics.csv", index=False)

    # head-to-head: xgb28_delta vs xgb_28d
    yt_delta, yp_delta = aligned["xgb28_delta"]
    yt_base,  yp_base  = aligned["xgb_28d"]
    delta_rows = []
    for i, ch in enumerate(CHANNELS):
        pt, lo, hi = bootstrap_pairwise(yt_delta[:, i], yp_delta[:, i], yp_base[:, i], mad_c[i])
        delta_rows.append({"delta": "NMAE(xgb28_delta) - NMAE(xgb_28d)",
                           "channel": ch, "point": pt, "ci_low": lo, "ci_high": hi})
    pt, lo, hi = bootstrap_pairwise(yt_delta.sum(1), yp_delta.sum(1), yp_base.sum(1), mad_bar)
    delta_rows.append({"delta": "NMAE(xgb28_delta) - NMAE(xgb_28d)",
                       "channel": "barrier", "point": pt, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(delta_rows).to_csv(OUT_RES / "head_to_head.csv", index=False)

    # leaderboard (wide NMAE)
    lb_rows = []
    for name in ["xgb_28d", "xgb28_delta"]:
        row = {"arm": name}
        for ch in CHANNELS + ["barrier"]:
            m = df[(df.arm == name) & (df.channel == ch) & (df.metric == "NMAE")]
            row[ch] = float(m.iloc[0].point) if len(m) else np.nan
        lb_rows.append(row)
    pd.DataFrame(lb_rows).to_csv(OUT_RES / "leaderboard.csv", index=False)

    # plots
    arms_plot = ["xgb_28d", "xgb28_delta"]
    bar_plot(df, "NMAE", "NMAE", OUT_FIG / "nmae_bars.png", arms_plot)
    bar_plot(df, "RMSE", "RMSE (kcal/mol)", OUT_FIG / "rmse_bars.png", arms_plot)
    parity_grid(aligned, mad_c, mad_bar, OUT_FIG / "parity_grid.png")

    # summary.md
    def get(arm, ch, met):
        r = df[(df.arm == arm) & (df.channel == ch) & (df.metric == met)]
        return r.iloc[0] if len(r) else None
    lines = [
        "# SPEC_06 — 2-arm comparison: xgb_28d (base) vs xgb28 + δ",
        "",
        f"- Cohort: {len(yt_ref)} rxns (v9 in-distribution m3, family-stratified 5-fold, identical split for both arms)",
        f"- Bootstrap: B={B_BOOT}, seed={SEED}, reaction-level resampling",
        f"- Descriptor set: 28-d = m3 (d1..d24) ⊕ d25 ⊕ d26 ⊕ d27 ⊕ d28 (spec05 no_sum_28d)",
        f"- xgb28_delta: 5 members averaged per (fold, rxn); ModelM1Delta (MACE-OFF23 medium + 4-block CA + AttnPool + MLP)",
        "",
        "## Pooled OOF NMAE (95% CI)",
        "",
        "| channel | xgb_28d (base) | xgb28 + δ | Δ NMAE (δ − base) |",
        "|---|---|---|---|",
    ]
    ht = pd.read_csv(OUT_RES / "head_to_head.csv")
    for ch in CHANNELS + ["barrier"]:
        rb = get("xgb_28d", ch, "NMAE"); rd = get("xgb28_delta", ch, "NMAE")
        rr = ht[ht.channel == ch].iloc[0]
        crosses = (rr.ci_low < 0) and (rr.ci_high > 0)
        mark = "  " if crosses else (" ✓" if rr.point < 0 else " ✗")
        lines.append(
            f"| {ch} | {rb.point:.3f} [{rb.ci_low:.3f}, {rb.ci_high:.3f}] "
            f"| {rd.point:.3f} [{rd.ci_low:.3f}, {rd.ci_high:.3f}] "
            f"| {rr.point:+.3f} [{rr.ci_low:+.3f}, {rr.ci_high:+.3f}]{mark} |"
        )
    lines += [
        "",
        "## Pooled OOF RMSE (kcal/mol, 95% CI)",
        "",
        "| channel | xgb_28d (base) | xgb28 + δ |",
        "|---|---|---|",
    ]
    for ch in CHANNELS + ["barrier"]:
        rb = get("xgb_28d", ch, "RMSE"); rd = get("xgb28_delta", ch, "RMSE")
        lines.append(
            f"| {ch} | {rb.point:.3f} [{rb.ci_low:.3f}, {rb.ci_high:.3f}] "
            f"| {rd.point:.3f} [{rd.ci_low:.3f}, {rd.ci_high:.3f}] |"
        )
    lines += ["", "## Files",
              "- pooled_oof.parquet (xgb28_delta), xgb_28d_oof.parquet (base)",
              "- metrics.csv, head_to_head.csv, leaderboard.csv",
              "- figures/nmae_bars.png, figures/rmse_bars.png, figures/parity_grid.png"]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
