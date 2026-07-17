"""SPEC_06 — pool the 5-fold OOF JSONs → metrics + bootstrap CIs + parity + summary.

Reads:  spec/spec06_2step_xgb28_delta/oof/xgb28_delta/fold*/member*.json

Writes:
  results/pooled_oof.parquet
  results/metrics.csv         per (arm, channel, metric)
  results/head_to_head.csv    NMAE deltas vs xgb_28d, xgb_24d+δ, ridge+δ,
                              m3-neural baselines (from spec02, spec03, spec05)
  results/leaderboard.csv     wide NMAE table across all pooled OOFs
  results/summary.md          human-readable summary
  figures/nmae_bars.png       per-channel + barrier NMAE bars with 95% CI
  figures/rmse_bars.png       per-channel + barrier RMSE bars with 95% CI
  figures/parity_grid.png     per-channel + barrier scatter
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
SPEC = REPO / "spec/spec06_2step_xgb28_delta"
OOF_ROOT = SPEC / "oof"
OUT_RES = SPEC / "results"
OUT_FIG = SPEC / "figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
ARM_COLORS = {
    "xgb28_delta": "#a83232",
    "xgb_28d":     "#4b779a",
    "xgb_24d":     "#7d9db8",
    "xgb_delta":   "#c05e2b",
    "ridge_delta": "#1f4e79",
    "m3_neural":   "#3e8548",
}
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


def cancellation(yt, yp):
    err = yp - yt
    bar_err = np.abs(err.sum(axis=1))
    abs_sum = np.sum(np.abs(err), axis=1)
    return float(np.mean(bar_err / np.maximum(abs_sum, 1e-12)))


def bootstrap_ci(yt, yp, mad, metric="NMAE", B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if metric == "NMAE":
            stats.append(nmae(yt[idx], yp[idx], mad))
        elif metric == "RMSE":
            stats.append(rmse(yt[idx], yp[idx]))
        elif metric == "R2":
            stats.append(r2(yt[idx], yp[idx]))
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


def load_spec02_arm(arm_name):
    """Load spec02 arm (ridge_delta, xgb_delta) for head-to-head."""
    subdir = REPO / "spec/spec02_abc_ablation/oof" / arm_name
    if not subdir.exists():
        return None
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
    return rids, yt, yp


def load_spec02_arm_A():
    """spec02 arm A: xgb_direct 24-d (single OOF parquet)."""
    p = REPO / "spec/spec02_abc_ablation/oof/oof_A.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    rids = df["reaction_id"].tolist()
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    return rids, yt, yp


# ---------- alignment + reporting ----------

def align(reference, others):
    """reference : (rids, yt, yp)   others : dict[name -> (rids,yt,yp)]

    Return yt_common (from reference) + dict[name -> (yt, yp)] aligned on
    reference rid ordering. Any 'others' missing a rid is dropped from that
    arm only.
    """
    ref_rids, ref_yt, ref_yp = reference
    out = {"__reference__": (ref_yt, ref_yp)}
    for name, arm in others.items():
        if arm is None:
            continue
        rids, yt, yp = arm
        r2i = {r: i for i, r in enumerate(rids)}
        idx = np.array([r2i[r] for r in ref_rids if r in r2i])
        if len(idx) != len(ref_rids):
            print(f"[align] {name}: only {len(idx)}/{len(ref_rids)} rids match, skipping")
            continue
        out[name] = (yt[idx], yp[idx])
    return ref_yt, out


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
    rows.append({"arm": arm_name, "channel": "barrier", "metric": "rho_cancellation",
                 "point": cancellation(yt, yp), "ci_low": np.nan, "ci_high": np.nan})
    return rows


def load_spec05_xgb28_fold0():
    """The base-only xgb_28d single-fold reference (fold0)."""
    p = REPO / "spec/spec05_d25_sum/results/no_sum_28d/xgb_28d_fold0.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {
        "strain": d["channels"]["strain"]["NMAE"],
        "Pauli":  d["channels"]["Pauli"]["NMAE"],
        "elst":   d["channels"]["elst"]["NMAE"],
        "oi":     d["channels"]["oi"]["NMAE"],
        "disp":   d["channels"]["disp"]["NMAE"],
        "barrier": d["barrier"]["NMAE"],
    }


def bar_plot(df, metric, ylabel, path, arms_plot):
    channels_plot = CHANNELS + ["barrier"]
    x = np.arange(len(channels_plot)); w = 0.85 / max(len(arms_plot), 1)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, arm_name in enumerate(arms_plot):
        pts, los, his = [], [], []
        for ch in channels_plot:
            row = df[(df.arm == arm_name) & (df.channel == ch) & (df.metric == metric)]
            if not len(row):
                pts.append(np.nan); los.append(0); his.append(0); continue
            row = row.iloc[0]
            pts.append(row.point)
            los.append(max(row.point - row.ci_low, 0))
            his.append(max(row.ci_high - row.point, 0))
        ax.bar(x + (i - (len(arms_plot) - 1) / 2) * w, pts, w,
               yerr=[los, his], capsize=3, label=arm_name,
               color=ARM_COLORS.get(arm_name, "#888"), edgecolor="white", lw=0.4)
    if metric == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel(f"{ylabel} (pooled OOF, 95% CI)")
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parity_grid(aligned, mad_c, mad_bar, path):
    arms_plot = [n for n in aligned if n != "__reference__"]
    arms_plot = ["__reference__"] + arms_plot  # keep our arm first
    # rename __reference__ back to xgb28_delta for the title
    rename = {"__reference__": "xgb28_delta"}
    channels_plot = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(len(arms_plot), len(channels_plot),
                             figsize=(3.4 * len(channels_plot), 3.4 * len(arms_plot)))
    if len(arms_plot) == 1:
        axes = np.array([axes])
    for r_i, arm_name in enumerate(arms_plot):
        yt, yp = aligned[arm_name]
        disp_name = rename.get(arm_name, arm_name)
        colr = ARM_COLORS.get(disp_name, "#888")
        for c_i, ch in enumerate(channels_plot):
            ax = axes[r_i, c_i] if len(arms_plot) > 1 else axes[c_i]
            if ch == "barrier":
                a = yt.sum(1); b = yp.sum(1); mad = mad_bar
            else:
                i_ = CHANNELS.index(ch); a = yt[:, i_]; b = yp[:, i_]; mad = mad_c[i_]
            ax.scatter(a, b, s=6, c=colr, alpha=0.55, edgecolor="none")
            lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.6)
            ax.text(0.03, 0.97,
                    f"NMAE={nmae(a, b, mad):.2f}\nR²={r2(a, b):.2f}\nslope={slope(a, b):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            if r_i == 0:
                ax.set_title(ch, fontsize=10)
            if c_i == 0:
                ax.set_ylabel(f"{disp_name}\ny_pred", fontsize=9)
            if r_i == len(arms_plot) - 1:
                ax.set_xlabel("y_true", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ours = load_xgb28_delta()
    if ours is None:
        raise SystemExit("no spec06 OOF JSONs found — nothing to aggregate")
    ref_rids, yt, yp = ours
    n_rxn = len(ref_rids)
    print(f"[spec06] pooled OOF: {n_rxn} rxns", flush=True)

    others = {
        "ridge_delta": load_spec02_arm("ridge_delta"),
        "xgb_delta":   load_spec02_arm("xgb_delta"),
        "xgb_24d":     load_spec02_arm_A(),
    }

    yt_ref, aligned = align((ref_rids, yt, yp), others)
    mad_c = np.array([np.mean(np.abs(yt_ref[:, i] - yt_ref[:, i].mean())) for i in range(5)])
    mad_bar = float(np.mean(np.abs(yt_ref.sum(1) - yt_ref.sum(1).mean())))

    # per-arm metric rows
    all_rows = []
    for name, (yt_a, yp_a) in aligned.items():
        arm_name = "xgb28_delta" if name == "__reference__" else name
        all_rows.extend(make_metric_rows(arm_name, yt_a, yp_a, mad_c, mad_bar))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_RES / "metrics.csv", index=False)

    # head-to-head deltas: reference vs each aligned other
    delta_rows = []
    ref_yt, ref_yp = aligned["__reference__"]
    for name, (yt_a, yp_a) in aligned.items():
        if name == "__reference__":
            continue
        for i, ch in enumerate(CHANNELS):
            pt, lo, hi = bootstrap_pairwise(ref_yt[:, i], ref_yp[:, i], yp_a[:, i], mad_c[i])
            delta_rows.append({"delta": f"NMAE(xgb28_delta) - NMAE({name})",
                               "channel": ch, "point": pt, "ci_low": lo, "ci_high": hi})
        pt, lo, hi = bootstrap_pairwise(ref_yt.sum(1), ref_yp.sum(1), yp_a.sum(1), mad_bar)
        delta_rows.append({"delta": f"NMAE(xgb28_delta) - NMAE({name})",
                           "channel": "barrier", "point": pt, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(delta_rows).to_csv(OUT_RES / "head_to_head.csv", index=False)

    # leaderboard (wide NMAE)
    lb_rows = []
    for name in ["xgb28_delta"] + [n for n in aligned if n != "__reference__"]:
        row = {"arm": name}
        for ch in CHANNELS + ["barrier"]:
            m = df[(df.arm == name) & (df.channel == ch) & (df.metric == "NMAE")]
            row[ch] = float(m.iloc[0].point) if len(m) else np.nan
        lb_rows.append(row)
    # append spec05 base-only fold0 reference (not aligned to pooled, but useful anchor)
    xgb28_ref = load_spec05_xgb28_fold0()
    if xgb28_ref is not None:
        lb_rows.append({"arm": "xgb_28d(base_fold0)", **xgb28_ref})
    pd.DataFrame(lb_rows).to_csv(OUT_RES / "leaderboard.csv", index=False)

    # plots
    arms_plot = ["xgb28_delta"] + [n for n in aligned if n != "__reference__"]
    bar_plot(df, "NMAE", "NMAE", OUT_FIG / "nmae_bars.png", arms_plot)
    bar_plot(df, "RMSE", "RMSE (kcal/mol)", OUT_FIG / "rmse_bars.png", arms_plot)
    parity_grid(aligned, mad_c, mad_bar, OUT_FIG / "parity_grid.png")

    # summary.md
    def get(arm, ch, met):
        r = df[(df.arm == arm) & (df.channel == ch) & (df.metric == met)]
        return r.iloc[0] if len(r) else None
    lines = [
        "# SPEC_06 — 2-step xgb28 + δ — summary",
        "",
        f"- Cohort: {n_rxn} rxns (v9 in-distribution m3, family-stratified 5-fold)",
        f"- Bootstrap: B={B_BOOT}, seed={SEED}, reaction-level resampling",
        f"- Descriptor set: 28-d = m3 (d1..d24) ⊕ d25 ⊕ d26 ⊕ d27 ⊕ d28 (spec05 no_sum_28d)",
        f"- Delta model: ModelM1Delta (MACE-OFF23 medium + 4-block CA + AttnPool + MLP)",
        f"- Fixed hp: LR={1e-5}, WD={1e-3}, EPOCHS_MAX={EPOCHS_MAX_STR}, PATIENCE={PATIENCE_STR}, batch=16",
        "",
        "## Pooled OOF NMAE (95% CI)",
        "",
    ]
    header = "| channel | " + " | ".join(arms_plot) + " |"
    sep = "|---" * (len(arms_plot) + 1) + "|"
    lines += [header, sep]
    for ch in CHANNELS + ["barrier"]:
        cells = []
        for arm_name in arms_plot:
            row = get(arm_name, ch, "NMAE")
            cells.append("n/a" if row is None
                         else f"{row.point:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]")
        lines.append(f"| {ch} | " + " | ".join(cells) + " |")

    lines += ["", "## Head-to-head vs other arms (NMAE delta, 95% CI)", ""]
    ht = pd.read_csv(OUT_RES / "head_to_head.csv")
    for other in [n for n in aligned if n != "__reference__"]:
        lines.append(f"### xgb28_delta − {other}")
        lines.append("")
        lines.append("| channel | Δ NMAE | 95% CI |")
        lines.append("|---|---|---|")
        for ch in CHANNELS + ["barrier"]:
            rr = ht[(ht.delta == f"NMAE(xgb28_delta) - NMAE({other})") & (ht.channel == ch)]
            if not len(rr):
                lines.append(f"| {ch} | n/a | n/a |"); continue
            d = rr.iloc[0]
            crosses = (d.ci_low < 0) and (d.ci_high > 0)
            mark = "" if crosses else ("✓ (better)" if d.point < 0 else "✗ (worse)")
            lines.append(f"| {ch} | {d.point:+.3f} | [{d.ci_low:+.3f}, {d.ci_high:+.3f}] {mark} |")
        lines.append("")

    if xgb28_ref is not None:
        lines += ["## Reference: spec05 xgb_28d base-only (fold0)", ""]
        lines += ["| channel | NMAE (base fold0) |", "|---|---|"]
        for ch in CHANNELS + ["barrier"]:
            lines.append(f"| {ch} | {xgb28_ref[ch]:.3f} |")

    lines += ["",
              "## Files",
              "- pooled_oof.parquet, metrics.csv, head_to_head.csv, leaderboard.csv",
              "- figures/nmae_bars.png, figures/rmse_bars.png, figures/parity_grid.png"]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}", flush=True)


EPOCHS_MAX_STR = "100000"
PATIENCE_STR = "10000"


if __name__ == "__main__":
    main()
