"""SPEC_11 - 2-arm aggregator (drop-in upgrade of spec06.aggregate).

Two arms only:
  - xgb_33d       : per-channel XGB on 33-d (arm-1). Reads b_test_* from the
                    same fold JSONs written by train_xgb33_delta.py (=b_val).
                    So identical 783-rxn OOF coverage, no recompute needed.
  - xgb33 + delta : b_val + delta = y_pred_* from same JSONs.

Writes:
  results/pooled_oof.parquet     (xgb33+delta pooled OOF, per rxn)
  results/xgb_33d_oof.parquet    (xgb_33d base-only pooled OOF, per rxn)
  results/metrics.csv            per (arm, channel, metric) with 95% CI
  results/head_to_head.csv       NMAE(xgb33+delta) - NMAE(xgb_33d)
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

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec11_electronic_33d"
OOF_ROOT = SPEC / "oof"
OUT_RES = SPEC / "results"
OUT_FIG = SPEC / "figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARM_COLORS = {"xgb_33d": "#4b779a", "xgb33_delta": "#a83232"}
ARM_LABELS = {"xgb_33d": "xgb_33d (base)", "xgb33_delta": "xgb33 + delta (this spec)"}
B_BOOT = 1000
SEED = 42


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


def load_both_arms():
    """Pool OOF JSONs; return (rids, y_true, y_pred_delta, y_pred_base).

    Both arms share the same rid list (both come out of the same JSONs).
    Members are averaged per (fold, rxn) as in spec06.
    """
    subdir = OOF_ROOT / "xgb33_delta"
    rows = []
    for f in subdir.glob("fold*/member*.json"):
        d = json.load(open(f))
        for i, r in enumerate(d["reaction_ids"]):
            row = {"reaction_id": r, "fold": d["fold"], "member": d["member"]}
            for c in CHANNELS:
                row[f"y_true_{c}"] = float(d[f"y_true_{c}"][i])
                row[f"y_pred_{c}"] = float(d[f"y_pred_{c}"][i])   # b + delta
                row[f"b_test_{c}"] = float(d[f"b_test_{c}"][i])   # b (xgb_33d)
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.groupby(["fold", "reaction_id"], as_index=False).mean(numeric_only=True)
    rids = df["reaction_id"].tolist()
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp_delta = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    yp_base  = df[[f"b_test_{c}" for c in CHANNELS]].to_numpy()

    # per-arm parquets
    df_delta = df[["fold", "reaction_id"] +
                  [f"y_true_{c}" for c in CHANNELS] +
                  [f"y_pred_{c}" for c in CHANNELS]]
    df_delta.to_parquet(OUT_RES / "pooled_oof.parquet", index=False)
    df_base = df[["fold", "reaction_id"] +
                 [f"y_true_{c}" for c in CHANNELS] +
                 [f"b_test_{c}" for c in CHANNELS]].rename(
                 columns={f"b_test_{c}": f"y_pred_{c}" for c in CHANNELS})
    df_base.to_parquet(OUT_RES / "xgb_33d_oof.parquet", index=False)
    return rids, yt, yp_delta, yp_base


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
    arms_plot = ["xgb_33d", "xgb33_delta"]
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
                    f"NMAE={nmae(a, b, mad):.3f}\nR2={r2(a, b):.2f}\nslope={slope(a, b):.2f}",
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
    loaded = load_both_arms()
    if loaded is None:
        raise SystemExit("no spec11 OOF JSONs found - nothing to aggregate")
    rids, yt_ref, yp_delta, yp_base = loaded
    print(f"[spec11] pooled OOF: {len(rids)} rxns", flush=True)

    mad_c = np.array([np.mean(np.abs(yt_ref[:, i] - yt_ref[:, i].mean())) for i in range(5)])
    mad_bar = float(np.mean(np.abs(yt_ref.sum(1) - yt_ref.sum(1).mean())))

    aligned = {"xgb_33d": (yt_ref, yp_base), "xgb33_delta": (yt_ref, yp_delta)}

    all_rows = []
    for arm_name, (yt_a, yp_a) in aligned.items():
        all_rows.extend(make_metric_rows(arm_name, yt_a, yp_a, mad_c, mad_bar))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_RES / "metrics.csv", index=False)

    yt_d, yp_d = aligned["xgb33_delta"]
    yt_b, yp_b = aligned["xgb_33d"]
    delta_rows = []
    for i, ch in enumerate(CHANNELS):
        pt, lo, hi = bootstrap_pairwise(yt_d[:, i], yp_d[:, i], yp_b[:, i], mad_c[i])
        delta_rows.append({"delta": "NMAE(xgb33+delta) - NMAE(xgb_33d)",
                           "channel": ch, "point": pt, "ci_low": lo, "ci_high": hi})
    pt, lo, hi = bootstrap_pairwise(yt_d.sum(1), yp_d.sum(1), yp_b.sum(1), mad_bar)
    delta_rows.append({"delta": "NMAE(xgb33+delta) - NMAE(xgb_33d)",
                       "channel": "barrier", "point": pt, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(delta_rows).to_csv(OUT_RES / "head_to_head.csv", index=False)

    lb_rows = []
    for name in ["xgb_33d", "xgb33_delta"]:
        row = {"arm": name}
        for ch in CHANNELS + ["barrier"]:
            m = df[(df.arm == name) & (df.channel == ch) & (df.metric == "NMAE")]
            row[ch] = float(m.iloc[0].point) if len(m) else np.nan
        lb_rows.append(row)
    pd.DataFrame(lb_rows).to_csv(OUT_RES / "leaderboard.csv", index=False)

    arms_plot = ["xgb_33d", "xgb33_delta"]
    bar_plot(df, "NMAE", "NMAE", OUT_FIG / "nmae_bars.png", arms_plot)
    bar_plot(df, "RMSE", "RMSE (kcal/mol)", OUT_FIG / "rmse_bars.png", arms_plot)
    parity_grid(aligned, mad_c, mad_bar, OUT_FIG / "parity_grid.png")

    def get(arm, ch, met):
        r = df[(df.arm == arm) & (df.channel == ch) & (df.metric == met)]
        return r.iloc[0] if len(r) else None
    lines = [
        "# SPEC_11 - 2-arm comparison: xgb_33d (base) vs xgb33 + delta",
        "",
        f"- Cohort: {len(yt_ref)} rxns (v9 in-distribution m3, family-stratified 5-fold, identical split for both arms)",
        f"- Bootstrap: B={B_BOOT}, seed={SEED}, reaction-level resampling",
        "- Descriptor set: 33-d = m3 (d1..d24) + d25 + d26 + d27 + d28 + d29 + d30 + d31 + d32 + d33",
        "- xgb33+delta: MACE-OFF23 medium + 4-block CA + AttnPool + MLP",
        "",
        "## Pooled OOF NMAE (95% CI)",
        "",
        "| channel | xgb_33d (base) | xgb33 + delta | delta NMAE (arm2 - arm1) |",
        "|---|---|---|---|",
    ]
    ht = pd.read_csv(OUT_RES / "head_to_head.csv")
    for ch in CHANNELS + ["barrier"]:
        rb = get("xgb_33d", ch, "NMAE"); rd = get("xgb33_delta", ch, "NMAE")
        rr = ht[ht.channel == ch].iloc[0]
        crosses = (rr.ci_low < 0) and (rr.ci_high > 0)
        mark = "  " if crosses else (" *" if rr.point < 0 else " !")
        lines.append(
            f"| {ch} | {rb.point:.3f} [{rb.ci_low:.3f}, {rb.ci_high:.3f}] "
            f"| {rd.point:.3f} [{rd.ci_low:.3f}, {rd.ci_high:.3f}] "
            f"| {rr.point:+.3f} [{rr.ci_low:+.3f}, {rr.ci_high:+.3f}]{mark} |"
        )
    lines += [
        "",
        "## Pooled OOF RMSE (kcal/mol, 95% CI)",
        "",
        "| channel | xgb_33d (base) | xgb33 + delta |",
        "|---|---|---|",
    ]
    for ch in CHANNELS + ["barrier"]:
        rb = get("xgb_33d", ch, "RMSE"); rd = get("xgb33_delta", ch, "RMSE")
        lines.append(
            f"| {ch} | {rb.point:.3f} [{rb.ci_low:.3f}, {rb.ci_high:.3f}] "
            f"| {rd.point:.3f} [{rd.ci_low:.3f}, {rd.ci_high:.3f}] |"
        )
    lines += ["", "## Files",
              "- pooled_oof.parquet (xgb33+delta), xgb_33d_oof.parquet (base)",
              "- metrics.csv, head_to_head.csv, leaderboard.csv",
              "- figures/nmae_bars.png, figures/rmse_bars.png, figures/parity_grid.png"]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
