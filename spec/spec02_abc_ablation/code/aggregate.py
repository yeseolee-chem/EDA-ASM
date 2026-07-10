"""T6-T8 - aggregate arm A/B/C OOF predictions to metrics + bootstrap CIs +
plots + REPORT.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
OOF_ROOT = REPO / "spec/spec02_abc_ablation/oof"
OUT_RES = REPO / "spec/spec02_abc_ablation/results"
OUT_FIG = REPO / "spec/spec02_abc_ablation/figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARM_COLORS = {"xgb_direct": "#4b779a", "ridge_delta": "#1f4e79", "xgb_delta": "#c05e2b"}
B_BOOT = 1000
SEED = 42


def nmae(yt, yp, mad):
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def slope(yt, yp):
    a = yt - yt.mean(); b = yp - yp.mean(); d = float(np.sum(a * a))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def cancellation(yt, yp):
    err = yp - yt
    bar_err = np.abs(err.sum(axis=1))
    abs_sum = np.sum(np.abs(err), axis=1)
    return float(np.mean(bar_err / np.maximum(abs_sum, 1e-12)))


def load_arm_A():
    p = OOF_ROOT / "oof_A.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    rids = df["reaction_id"].tolist()
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    return rids, yt, yp


def load_arm_BC(arm_name):
    subdir = OOF_ROOT / arm_name
    if not subdir.exists(): return None
    all_rows = []
    for f in subdir.glob("fold*/member*.json"):
        d = json.load(open(f))
        for i, r in enumerate(d["reaction_ids"]):
            row = {"reaction_id": r, "fold": d["fold"], "member": d["member"]}
            for c in CHANNELS:
                row[f"y_true_{c}"] = float(d[f"y_true_{c}"][i])
                row[f"y_pred_{c}"] = float(d[f"y_pred_{c}"][i])
            all_rows.append(row)
    if not all_rows: return None
    df = pd.DataFrame(all_rows)
    df = df.groupby(["fold", "reaction_id"], as_index=False).mean(numeric_only=True)
    rids = df["reaction_id"].tolist()
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    return rids, yt, yp


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


def align_arms(A, B, C):
    arms = {"xgb_direct": A, "ridge_delta": B, "xgb_delta": C}
    rid_sets = [set(v[0]) for v in arms.values() if v is not None]
    common = set.intersection(*rid_sets) if rid_sets else set()
    sorted_rids = sorted(common)
    aligned = {}
    for name, arm in arms.items():
        if arm is None:
            aligned[name] = None; continue
        rids, yt, yp = arm
        idx = np.array([rids.index(r) for r in sorted_rids])
        aligned[name] = {"rids": sorted_rids, "yt": yt[idx], "yp": yp[idx]}
    yt_common = next(v["yt"] for v in aligned.values() if v is not None)
    return yt_common, aligned


def main():
    A = load_arm_A(); B = load_arm_BC("ridge_delta"); C = load_arm_BC("xgb_delta")
    if not any([A, B, C]):
        raise SystemExit("no arm data")

    yt_com, aligned = align_arms(A, B, C)
    n_rxn = len(yt_com)
    print(f"aligned across arms: {n_rxn} rxns")
    mad_c = np.array([np.mean(np.abs(yt_com[:, i] - yt_com[:, i].mean())) for i in range(5)])
    mad_bar = np.mean(np.abs(yt_com.sum(1) - yt_com.sum(1).mean()))

    rows = []
    for arm_name, dat in aligned.items():
        if dat is None: continue
        yp = dat["yp"]
        for i, ch in enumerate(CHANNELS):
            for metric in ["NMAE", "RMSE", "R2"]:
                pt, lo, hi = bootstrap_ci(yt_com[:, i], yp[:, i], mad_c[i], metric=metric)
                rows.append({"arm": arm_name, "channel": ch, "metric": metric,
                             "point": pt, "ci_low": lo, "ci_high": hi})
            rows.append({"arm": arm_name, "channel": ch, "metric": "slope",
                         "point": slope(yt_com[:, i], yp[:, i]), "ci_low": np.nan, "ci_high": np.nan})
        for metric in ["NMAE", "RMSE", "R2"]:
            pt, lo, hi = bootstrap_ci(yt_com.sum(1), yp.sum(1), mad_bar, metric=metric)
            rows.append({"arm": arm_name, "channel": "barrier", "metric": metric,
                         "point": pt, "ci_low": lo, "ci_high": hi})
        rows.append({"arm": arm_name, "channel": "barrier", "metric": "slope",
                     "point": slope(yt_com.sum(1), yp.sum(1)),
                     "ci_low": np.nan, "ci_high": np.nan})
        rows.append({"arm": arm_name, "channel": "barrier", "metric": "rho_cancellation",
                     "point": cancellation(yt_com, yp), "ci_low": np.nan, "ci_high": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "abc_metrics.csv", index=False)

    delta_rows = []
    for pair in [("ridge_delta", "xgb_delta"), ("ridge_delta", "xgb_direct"),
                 ("xgb_delta", "xgb_direct")]:
        a1, a2 = pair
        if aligned[a1] is None or aligned[a2] is None: continue
        yp1 = aligned[a1]["yp"]; yp2 = aligned[a2]["yp"]
        for i, ch in enumerate(CHANNELS):
            pt, lo, hi = bootstrap_pairwise(yt_com[:, i], yp1[:, i], yp2[:, i], mad_c[i])
            delta_rows.append({"delta": f"NMAE({a1}) - NMAE({a2})", "channel": ch,
                               "point": pt, "ci_low": lo, "ci_high": hi})
        pt, lo, hi = bootstrap_pairwise(yt_com.sum(1), yp1.sum(1), yp2.sum(1), mad_bar)
        delta_rows.append({"delta": f"NMAE({a1}) - NMAE({a2})", "channel": "barrier",
                           "point": pt, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(delta_rows).to_csv(OUT_RES / "abc_deltas.csv", index=False)

    channels_plot = CHANNELS + ["barrier"]
    arms_plot = [n for n in ["xgb_direct", "ridge_delta", "xgb_delta"] if aligned.get(n)]
    x = np.arange(len(channels_plot)); w = 0.85 / max(len(arms_plot), 1)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, arm_name in enumerate(arms_plot):
        pts, los, his = [], [], []
        for ch in channels_plot:
            row = df[(df.arm == arm_name) & (df.channel == ch) & (df.metric == "NMAE")].iloc[0]
            pts.append(row.point); los.append(row.point - row.ci_low); his.append(row.ci_high - row.point)
        ax.bar(x + (i - (len(arms_plot) - 1) / 2) * w, pts, w,
               yerr=[los, his], capsize=3, label=arm_name,
               color=ARM_COLORS[arm_name], edgecolor="white", lw=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel("NMAE (pooled OOF, 95% CI)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax.set_title("SPEC_02 A/B/C ablation - m3 v7 776 rxns")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "abc_nmae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, arm_name in enumerate(arms_plot):
        pts, los, his = [], [], []
        for ch in channels_plot:
            row = df[(df.arm == arm_name) & (df.channel == ch) & (df.metric == "RMSE")].iloc[0]
            pts.append(row.point); los.append(row.point - row.ci_low); his.append(row.ci_high - row.point)
        ax.bar(x + (i - (len(arms_plot) - 1) / 2) * w, pts, w,
               yerr=[los, his], capsize=3, label=arm_name,
               color=ARM_COLORS[arm_name], edgecolor="white", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel("RMSE (kcal/mol, 95% CI)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax.set_title("SPEC_02 A/B/C ablation RMSE - m3 v7 776 rxns")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "abc_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(arms_plot), len(channels_plot),
                             figsize=(3.4 * len(channels_plot), 3.4 * len(arms_plot)))
    if len(arms_plot) == 1:
        axes = np.array([axes])
    for r_i, arm_name in enumerate(arms_plot):
        yp = aligned[arm_name]["yp"]
        for c_i, ch in enumerate(channels_plot):
            ax = axes[r_i, c_i] if len(arms_plot) > 1 else axes[c_i]
            if ch == "barrier":
                a = yt_com.sum(1); b = yp.sum(1)
            else:
                i_ = CHANNELS.index(ch); a = yt_com[:, i_]; b = yp[:, i_]
            ax.scatter(a, b, s=6, c=ARM_COLORS[arm_name], alpha=0.55, edgecolor="none")
            lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.6)
            ax.text(0.03, 0.97,
                    f"NMAE={nmae(a, b, mad_bar if ch=='barrier' else mad_c[CHANNELS.index(ch)]):.2f}\n"
                    f"R^2={r2(a, b):.2f}\nslope={slope(a, b):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            if r_i == 0: ax.set_title(ch, fontsize=10)
            if c_i == 0: ax.set_ylabel(f"{arm_name}\ny_pred", fontsize=9)
            if r_i == len(arms_plot) - 1: ax.set_xlabel("y_true", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "abc_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def get(arm, ch, met):
        row = df[(df.arm == arm) & (df.channel == ch) & (df.metric == met)]
        return row.iloc[0] if len(row) else None
    lines = ["# SPEC_02 A/B/C ablation - REPORT", "",
             f"- Cohort: {n_rxn} rxns (v7 in-distribution m3, 24-d, 5-fold family-stratified)",
             f"- Bootstrap: B={B_BOOT}, seed={SEED}, reaction-level resampling",
             "",
             "## Ensemble NMAE (pooled OOF, 95% CI)",
             "",
             "| channel | xgb_direct | ridge_delta | xgb_delta |",
             "|---|---|---|---|"]
    for ch in channels_plot:
        cells = []
        for arm_name in ["xgb_direct", "ridge_delta", "xgb_delta"]:
            row = get(arm_name, ch, "NMAE")
            if row is None: cells.append("n/a"); continue
            cells.append(f"{row.point:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]")
        lines.append(f"| {ch} | " + " | ".join(cells) + " |")
    lines += ["", "## Pairwise NMAE delta (95% CI over reactions)", "See abc_deltas.csv."]
    if aligned.get("ridge_delta") and aligned.get("xgb_delta"):
        row_bar = pd.read_csv(OUT_RES / "abc_deltas.csv")
        bc = row_bar[(row_bar.delta == "NMAE(ridge_delta) - NMAE(xgb_delta)") &
                     (row_bar.channel == "barrier")]
        if len(bc):
            d = bc.iloc[0]
            crosses_zero = (d.ci_low < 0) and (d.ci_high > 0)
            verdict = ("indistinguishable (CI crosses 0) - keep ridge for simplicity"
                       if crosses_zero else
                       ("xgb baseline is better" if d.point > 0 else "ridge baseline is better"))
            lines += ["", "## Barrier verdict (B - C)",
                      f"- delta NMAE = {d.point:+.3f}, 95% CI = [{d.ci_low:+.3f}, {d.ci_high:+.3f}]",
                      f"- verdict: **{verdict}**"]
    (OUT_RES / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'REPORT.md'}")


if __name__ == "__main__":
    main()
