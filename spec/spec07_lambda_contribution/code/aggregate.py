"""SPEC_07 — aggregate λ-sweep OOF JSONs → curve CSV + metrics + summary.md.

Reads:  spec/spec07_lambda_contribution/oof/lam*/fold*/member*.json

Writes:
  results/pooled_oof.parquet           (one row per (reaction, λ))
  results/lambda_curve.csv             long: (lambda, channel, metric, point, ci_lo, ci_hi)
  results/lambda_star.json             argmin λ per (channel + barrier) for NMAE
  results/summary.md                   human-readable table
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec07_lambda_contribution"
OOF_ROOT = SPEC / "oof"
OUT_RES = SPEC / "results"
OUT_RES.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
B_BOOT = 1000
SEED = 42


def nmae(yt, yp, mad):
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss / (tot + 1e-12))


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
    if metric == "NMAE":
        pt = nmae(yt, yp, mad)
    elif metric == "RMSE":
        pt = rmse(yt, yp)
    else:
        pt = r2(yt, yp)
    return pt, lo, hi


def load_lambda(lam_dir: Path):
    rows = []
    lam = None
    n_early_ok = n_early_fail = 0
    for f in sorted(lam_dir.glob("fold*/member*.json")):
        d = json.load(open(f))
        lam = float(d["lam"])
        if d.get("early_stopped", True):
            n_early_ok += 1
        else:
            n_early_fail += 1
        for i, r in enumerate(d["reaction_ids"]):
            row = {"reaction_id": r, "lam": lam, "fold": d["fold"], "member": d["member"]}
            for c in CHANNELS:
                row[f"y_true_{c}"] = float(d[f"y_true_{c}"][i])
                row[f"y_pred_{c}"] = float(d[f"y_pred_{c}"][i])
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.groupby(["fold", "reaction_id"], as_index=False).mean(numeric_only=True)
    yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
    return {
        "lam": lam,
        "df": df,
        "yt": yt,
        "yp": yp,
        "n": len(df),
        "early_ok": n_early_ok,
        "early_fail": n_early_fail,
    }


def main():
    lam_dirs = sorted([p for p in OOF_ROOT.iterdir()
                       if p.is_dir() and p.name.startswith("lam")])
    if not lam_dirs:
        raise SystemExit(f"no lam* dirs under {OOF_ROOT}")

    all_pooled = []
    lam_data = {}
    for d in lam_dirs:
        pl = load_lambda(d)
        if pl is None:
            print(f"[skip empty] {d.name}", flush=True); continue
        lam = pl["lam"]
        lam_data[lam] = pl
        pl["df"]["lam"] = lam
        all_pooled.append(pl["df"])
        print(f"[loaded] λ={lam:.2f}  n={pl['n']}  early_ok={pl['early_ok']}"
              f"  early_fail={pl['early_fail']}", flush=True)

    if not lam_data:
        raise SystemExit("nothing to aggregate")

    combined = pd.concat(all_pooled, ignore_index=True)
    combined.to_parquet(OUT_RES / "pooled_oof.parquet", index=False)

    # Use the intersection cohort across λ (rid must exist at every λ).
    rid_sets = [set(pl["df"]["reaction_id"]) for pl in lam_data.values()]
    common_rids = sorted(set.intersection(*rid_sets)) if rid_sets else []
    print(f"[align] common rids across {len(lam_data)} λ points: {len(common_rids)}",
          flush=True)

    # Anchor mad_c / mad_bar to the labels at the smallest λ (labels are identical
    # across λ but we need a stable indexing).
    ref_lam = min(lam_data.keys())
    ref_df = lam_data[ref_lam]["df"].set_index("reaction_id").loc[common_rids]
    yt_ref = ref_df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
    mad_c = np.array([np.mean(np.abs(yt_ref[:, i] - yt_ref[:, i].mean()))
                      for i in range(5)])
    bar = yt_ref.sum(axis=1)
    mad_bar = float(np.mean(np.abs(bar - bar.mean())))

    # Per-λ metrics on the common cohort
    curve_rows = []
    for lam in sorted(lam_data.keys()):
        pl = lam_data[lam]
        df = pl["df"].set_index("reaction_id").loc[common_rids]
        yt = df[[f"y_true_{c}" for c in CHANNELS]].to_numpy()
        yp = df[[f"y_pred_{c}" for c in CHANNELS]].to_numpy()
        for i, ch in enumerate(CHANNELS):
            for metric in ["NMAE", "RMSE", "R2"]:
                pt, lo, hi = bootstrap_ci(yt[:, i], yp[:, i], mad_c[i], metric=metric)
                curve_rows.append({"lambda": lam, "channel": ch, "metric": metric,
                                   "point": pt, "ci_lo": lo, "ci_hi": hi})
        for metric in ["NMAE", "RMSE", "R2"]:
            pt, lo, hi = bootstrap_ci(yt.sum(1), yp.sum(1), mad_bar, metric=metric)
            curve_rows.append({"lambda": lam, "channel": "barrier", "metric": metric,
                               "point": pt, "ci_lo": lo, "ci_hi": hi})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUT_RES / "lambda_curve.csv", index=False)

    # λ* per channel + barrier (argmin NMAE)
    lam_star = {}
    for ch in CHANNELS + ["barrier"]:
        sub = curve_df[(curve_df.channel == ch) & (curve_df.metric == "NMAE")]
        if len(sub):
            row = sub.loc[sub["point"].idxmin()]
            lam_star[ch] = {"lambda": float(row["lambda"]),
                            "nmae": float(row["point"]),
                            "ci_lo": float(row["ci_lo"]),
                            "ci_hi": float(row["ci_hi"])}
    (OUT_RES / "lambda_star.json").write_text(json.dumps(lam_star, indent=2))

    # summary.md
    lambdas_sorted = sorted(lam_data.keys())
    lines = [
        "# SPEC_07 — λ-contribution sweep — summary",
        "",
        f"- Cohort (common across all λ): {len(common_rids)} rxns",
        f"- λ grid: {[f'{x:.2f}' for x in lambdas_sorted]}",
        f"- Bootstrap: B={B_BOOT}, seed={SEED}, reaction-level resampling",
        f"- Model: xgb-28d base (b) + ModelM1Delta residual (δ), retrained per λ",
        f"- Blend: ŷ = (1−λ)·δ + λ·b   (λ=1 = base-only, δ untrained)",
        "",
        "## Per-channel + barrier NMAE vs λ (95% CI in brackets)",
        "",
    ]
    header = "| channel | " + " | ".join(f"λ={x:.2f}" for x in lambdas_sorted) + " |"
    sep = "|---" * (len(lambdas_sorted) + 1) + "|"
    lines += [header, sep]
    for ch in CHANNELS + ["barrier"]:
        cells = []
        for lam in lambdas_sorted:
            r = curve_df[(curve_df.channel == ch) & (curve_df.metric == "NMAE")
                         & (curve_df["lambda"] == lam)]
            if not len(r):
                cells.append("n/a"); continue
            r = r.iloc[0]
            cells.append(f"{r.point:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
        lines.append(f"| {ch} | " + " | ".join(cells) + " |")

    lines += ["", "## λ* (argmin NMAE) per channel", "",
              "| channel | λ* | NMAE(λ*) | 95% CI |", "|---|---|---|---|"]
    for ch in CHANNELS + ["barrier"]:
        if ch in lam_star:
            s = lam_star[ch]
            lines.append(f"| {ch} | {s['lambda']:.2f} | {s['nmae']:.3f} | "
                         f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] |")

    lines += ["", "## Cell-count check per λ",
              "", "| λ | pooled rxns | early-stopped folds | did NOT early-stop |",
              "|---|---|---|---|"]
    for lam in lambdas_sorted:
        pl = lam_data[lam]
        lines.append(f"| {lam:.2f} | {pl['n']} | {pl['early_ok']} | {pl['early_fail']} |")

    lines += ["",
              "## Files",
              "- pooled_oof.parquet, lambda_curve.csv, lambda_star.json",
              "- figures/lambda_nmae.png, figures/lambda_rmse.png,",
              "  figures/parity_at_lamstar.png"]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
