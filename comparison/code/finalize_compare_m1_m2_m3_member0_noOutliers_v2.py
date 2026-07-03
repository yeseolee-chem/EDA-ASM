"""3-way compare m1/m2/m3 (5 folds × member 0, no-OOD pool) with PARITY OUTLIERS removed.

The visual parity plots showed a few reactions with catastrophic residuals
(>1000 kcal/mol in Pauli/oi/V_elst) — primarily caused by xTB cache failures
(missing source_dir archives → mean-imputed descriptors → unphysical baseline)
or by pathological xTB SP results (e.g. dipolar_003220 with E_int = -764 kcal/mol).

Outlier detection (per-reaction, ANY-model, ANY-channel):
  modified Z (Iglewicz–Hoaglin) on the per-channel residual distribution
  pooled across the 3 models;  |Z*| > 5  ⇒  reaction excluded.

Outputs:
  results_compare_m1_m2_m3_member0_noOutliers/{model}/metrics_*.csv
  results_compare_m1_m2_m3_member0_noOutliers/excluded_rids.json
  figures_compare_m1_m2_m3_member0_noOutliers/{compare_*,*_parity,compare_parity_grid}.png
  REPORT_compare_m1_m2_m3_member0_noOutliers.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

HERE = Path(__file__).resolve().parent
OUT_RES = HERE / "results_compare_m1_m2_m3v2_member0_noOutliers"
OUT_FIG = HERE / "figures_compare_m1_m2_m3v2_member0_noOutliers"
REPORT_MD = HERE / "REPORT_compare_m1_m2_m3v2_member0_noOutliers.md"

ONLY_MEMBER = 0
FOLDS = [0, 1, 2, 3, 4]
ROBUST_Z_THRESHOLD = 5.0  # modified Z (median + MAD); >5 = catastrophic outlier

MODELS = [
    ("m1", "geom6",              "m1", "#1E2761"),
    ("m2", "xtb_geom6",          "m2", "#1C7293"),
    ("m3", "xtb_geom6_plus_v2",  "m3", "#C45A4D"),
]
MODEL_TO_BASELINE = {m[0]: m[1] for m in MODELS}
MODEL_LABEL = {m[0]: m[2] for m in MODELS}
MODEL_COLOR = {m[0]: m[3] for m in MODELS}

CHANS = ["strain", "Pauli", "Velst", "oi", "disp"]
DATA_TO_DISP = {"E_strain_kcal": "strain", "Pauli_kcal": "Pauli",
                "V_elst_kcal": "Velst", "E_orb_kcal": "oi", "E_disp_kcal": "disp"}


def channel_metrics(y_true, y_pred):
    e = y_pred - y_true
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    tail = rmse / mae if mae > 0 else np.nan
    ybar = float(np.mean(y_true))
    denom = float(np.mean(np.abs(y_true - ybar)))
    nmae = mae / denom if denom > 0 else np.nan
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - ybar) ** 2))
    r2_det = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "tail_ratio": tail,
            "NMAE": nmae, "R2_det": r2_det}


def derive_family(rid: str) -> str:
    if rid.startswith("dipolar"): return "dipolar"
    if rid.startswith("rgd1"): return "rgd1"
    if rid.startswith("qmrxn20_sn2"): return "qmrxn20_sn2"
    if rid.startswith("qmrxn20_e2"): return "qmrxn20_e2"
    return "?"


def load_cell(model_key: str, fold: int):
    bd = MODEL_TO_BASELINE[model_key]
    p = HERE / f"trackB_lowlr_no_ood_{bd}" / "m1_delta" / f"fold{fold}" / f"member{ONLY_MEMBER}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    OUT_RES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # ===== Pass 1: assemble per-(rid, channel, model) residuals =====
    components = None
    long_rows = []
    for model_key, _, _, _ in MODELS:
        for fold in FOLDS:
            d = load_cell(model_key, fold)
            if d is None:
                continue
            if components is None:
                components = d["components"]
                comp_idx = {DATA_TO_DISP[c]: i for i, c in enumerate(components) if c in DATA_TO_DISP}
            y_true = np.array(d["y_true"]); y_pred = np.array(d["y_pred"])
            for i, rid in enumerate(d["reaction_ids"]):
                for ch in CHANS:
                    ci = comp_idx[ch]
                    long_rows.append({
                        "rid": rid, "fold": fold, "model": model_key, "channel": ch,
                        "y_true": float(y_true[i, ci]),
                        "y_pred": float(y_pred[i, ci]),
                        "residual": float(y_pred[i, ci] - y_true[i, ci]),
                    })
    long_df = pd.DataFrame(long_rows)
    print(f"long_df rows: {len(long_df)}  (rid × model × channel × member 0)")

    # ===== Pass 2: per-channel robust Z (modified Z-score, Iglewicz–Hoaglin) =====
    # Pool residuals over all 3 models per channel; outlier threshold from MAD.
    flags = []
    print(f"\n=== Per-channel robust Z thresholds (|Z*| > {ROBUST_Z_THRESHOLD}) ===")
    for ch in CHANS:
        r = long_df.loc[long_df["channel"] == ch, "residual"].to_numpy()
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        scale = 1.4826 * mad
        if scale == 0:
            print(f"  {ch}: MAD=0 → cannot robust-z; skip")
            continue
        z = (long_df.loc[long_df["channel"] == ch, "residual"] - med) / scale
        long_df.loc[long_df["channel"] == ch, "z_robust"] = z
        flagged = long_df.loc[(long_df["channel"] == ch) & (z.abs() > ROBUST_Z_THRESHOLD)]
        print(f"  {ch:8s}  median={med:+.2f}  MAD={mad:.2f}  scale={scale:.2f}  → {len(flagged)} cell rows flagged")
        flags.append(flagged)

    flag_df = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()
    excluded_rids = sorted(set(flag_df["rid"].tolist()))
    print(f"\n=== Excluded reactions (any (channel, model) |Z*| > {ROBUST_Z_THRESHOLD}) : {len(excluded_rids)} unique rids ===")
    for rid in excluded_rids:
        rows = flag_df[flag_df["rid"] == rid]
        print(f"  {rid}  flagged in:")
        for _, rr in rows.iterrows():
            print(f"    fold{rr['fold']} {rr['model']} {rr['channel']:8s}  y_true={rr['y_true']:+.2f}  y_pred={rr['y_pred']:+.2f}  |Z*|={abs(rr['z_robust']):.2f}")
    (OUT_RES).mkdir(parents=True, exist_ok=True)
    json.dump({
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "excluded_rids": excluded_rids,
        "n_excluded": len(excluded_rids),
    }, open(OUT_RES / "excluded_rids.json", "w"), indent=2)

    # ===== Pass 3: aggregate with exclusion =====
    kept_mask = ~long_df["rid"].isin(excluded_rids)
    kept_df = long_df[kept_mask].copy()
    print(f"\nkept rows: {len(kept_df)} / {len(long_df)}  "
          f"(removed {len(long_df) - len(kept_df)} cell rows = {len(excluded_rids)} unique rids)")

    aggregated = {}
    for model_key, _, _, _ in MODELS:
        meta_rows = []
        rows = []
        pooled = {ch: {"y_true": [], "y_pred": [], "family": []} for ch in CHANS}
        # Barrier = Σ 5 channels = ΔE‡_strain + ΔE‡_Pauli + ΔV‡_elst + ΔE‡_oi + ΔE‡_disp
        barrier_pooled = {"y_true": [], "y_pred": [], "family": []}
        barrier_rows: list[dict] = []

        for fold in FOLDS:
            d = load_cell(model_key, fold)
            if d is None:
                continue
            kept_rids = set(kept_df.loc[(kept_df["model"] == model_key) & (kept_df["fold"] == fold), "rid"])
            n_orig = len(d["reaction_ids"])
            n_kept = len(kept_rids)
            y_true = np.array(d["y_true"]); y_pred = np.array(d["y_pred"])

            comp_idx = {DATA_TO_DISP[c]: i for i, c in enumerate(components) if c in DATA_TO_DISP}
            for ch in CHANS:
                ci = comp_idx[ch]
                yt, yp, fams = [], [], []
                for i, rid in enumerate(d["reaction_ids"]):
                    if rid in kept_rids:
                        yt.append(y_true[i, ci])
                        yp.append(y_pred[i, ci])
                        fams.append(derive_family(rid))
                yt = np.array(yt); yp = np.array(yp)
                met = channel_metrics(yt, yp)
                rows.append({"model": model_key, "fold": fold, "member": ONLY_MEMBER,
                             "channel": ch, "n_test": len(yt), **met})
                pooled[ch]["y_true"].extend(yt.tolist())
                pooled[ch]["y_pred"].extend(yp.tolist())
                pooled[ch]["family"].extend(fams)

            # Barrier metric: sum 5 channels per reaction, then compute RMSE/MAE
            barrier_true, barrier_pred, barrier_fams = [], [], []
            for i, rid in enumerate(d["reaction_ids"]):
                if rid in kept_rids:
                    barrier_true.append(float(y_true[i, :].sum()))
                    barrier_pred.append(float(y_pred[i, :].sum()))
                    barrier_fams.append(derive_family(rid))
            barrier_true = np.array(barrier_true); barrier_pred = np.array(barrier_pred)
            bmet = channel_metrics(barrier_true, barrier_pred)
            barrier_rows.append({"model": model_key, "fold": fold, "member": ONLY_MEMBER,
                                 "n_test": len(barrier_true), **bmet})
            barrier_pooled["y_true"].extend(barrier_true.tolist())
            barrier_pooled["y_pred"].extend(barrier_pred.tolist())
            barrier_pooled["family"].extend(barrier_fams)

            meta_rows.append({"model": model_key, "fold": fold, "seed": d["seed"],
                              "n_train": d["n_train"], "n_val": d["n_val"],
                              "n_test_original": n_orig, "n_test_kept": n_kept,
                              "best_epoch": d["best_epoch"], "final_epoch": d["final_epoch"],
                              "elapsed_s": d["elapsed_s"],
                              "test_mae_mean_original": d["test_mae_mean_kcal"]})
        if not rows:
            continue
        a = {"runs": pd.DataFrame(rows), "meta": pd.DataFrame(meta_rows),
             "pooled": pooled,
             "barrier_runs": pd.DataFrame(barrier_rows),
             "barrier_pooled": barrier_pooled}
        aggregated[model_key] = a
        bd = OUT_RES / model_key
        bd.mkdir(parents=True, exist_ok=True)
        a["runs"].to_csv(bd / "metrics_per_run.csv", index=False)
        sm = (a["runs"].groupby("channel")[["MAE","RMSE","tail_ratio","NMAE","R2_det"]]
                       .agg(["mean","std"]).round(4)).reindex(CHANS)
        sm.to_csv(bd / "metrics_summary.csv")
        a["summary"] = sm
        a["meta"].to_csv(bd / "cell_meta.csv", index=False)
        # Barrier metrics (sum of 5 channels)
        a["barrier_runs"].to_csv(bd / "barrier_metrics_per_run.csv", index=False)
        bs = a["barrier_runs"][["MAE","RMSE","tail_ratio","NMAE","R2_det"]].agg(["mean","std"]).round(4)
        bs.to_csv(bd / "barrier_metrics_summary.csv")
        a["barrier_summary"] = bs

    # ===== Compare CSV =====
    compare_rows = []
    for ch in CHANS:
        row = {"channel": ch}
        for k in aggregated:
            sm = aggregated[k]["summary"]
            row[f"{k}_NMAE"] = sm.loc[ch, ("NMAE", "mean")]
            row[f"{k}_NMAE_std"] = sm.loc[ch, ("NMAE", "std")]
            row[f"{k}_R2"] = sm.loc[ch, ("R2_det", "mean")]
            row[f"{k}_R2_std"] = sm.loc[ch, ("R2_det", "std")]
            row[f"{k}_MAE"] = sm.loc[ch, ("MAE", "mean")]
            row[f"{k}_MAE_std"] = sm.loc[ch, ("MAE", "std")]
            row[f"{k}_RMSE"] = sm.loc[ch, ("RMSE", "mean")]
            row[f"{k}_RMSE_std"] = sm.loc[ch, ("RMSE", "std")]
        if "m1" in aggregated and "m3" in aggregated:
            row["delta_R2_m3_m1"] = aggregated["m3"]["summary"].loc[ch, ("R2_det","mean")] - aggregated["m1"]["summary"].loc[ch, ("R2_det","mean")]
            row["delta_NMAE_m3_m1"] = aggregated["m3"]["summary"].loc[ch, ("NMAE","mean")] - aggregated["m1"]["summary"].loc[ch, ("NMAE","mean")]
        if "m2" in aggregated and "m3" in aggregated:
            row["delta_R2_m3_m2"] = aggregated["m3"]["summary"].loc[ch, ("R2_det","mean")] - aggregated["m2"]["summary"].loc[ch, ("R2_det","mean")]
            row["delta_NMAE_m3_m2"] = aggregated["m3"]["summary"].loc[ch, ("NMAE","mean")] - aggregated["m2"]["summary"].loc[ch, ("NMAE","mean")]
        compare_rows.append(row)
    cmp_df = pd.DataFrame(compare_rows)
    cmp_df.to_csv(OUT_RES / "metrics_compare.csv", index=False)
    print("\n=== 3-way comparison (member 0, outliers removed) ===")
    print(cmp_df.to_string(index=False))

    # ===== Barrier comparison table (sum of 5 channels) =====
    barrier_row = {}
    for k in aggregated:
        bs = aggregated[k]["barrier_summary"]
        for metric in ["MAE", "RMSE", "NMAE", "R2_det", "tail_ratio"]:
            barrier_row[f"{k}_{metric}"]     = float(bs.loc["mean", metric])
            barrier_row[f"{k}_{metric}_std"] = float(bs.loc["std",  metric])
    barrier_df = pd.DataFrame([barrier_row])
    barrier_df.to_csv(OUT_RES / "barrier_metrics_compare.csv", index=False)
    print("\n=== Barrier (Σ 5 channels) — 3-way ===")
    for metric in ["MAE", "RMSE", "NMAE", "R2_det"]:
        print(f"  {metric:8s}  " +
              "  ".join(f"{k}={aggregated[k]['barrier_summary'].loc['mean', metric]:.3f}"
                        f"±{aggregated[k]['barrier_summary'].loc['std', metric]:.3f}"
                        for k in aggregated))

    # ===== Barrier bar plot =====
    def barrier_bars(metric, hline=None, hl_label=None, ylabel=None, fname=None):
        keys = list(aggregated.keys())
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = np.arange(len(keys))
        vals = [aggregated[k]["barrier_summary"].loc["mean", metric] for k in keys]
        errs = [aggregated[k]["barrier_summary"].loc["std",  metric] for k in keys]
        colors = [MODEL_COLOR[k] for k in keys]
        ax.bar(x, vals, 0.6, yerr=errs, color=colors, capsize=4, edgecolor="black", linewidth=0.4)
        if hline is not None:
            ax.axhline(hline, ls="--", color="grey", label=hl_label)
            ax.legend(loc="best", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[k] for k in keys])
        ax.set_ylabel(ylabel or metric)
        ax.set_title(f"Barrier (Σ 5 channels) {metric} — m1 vs m2 vs m3 "
                     f"(5 folds × m0, no-OOD + {len(excluded_rids)} outliers removed)")
        ax.grid(True, alpha=0.15, linestyle="--", axis="y")
        fig.tight_layout(); fig.savefig(OUT_FIG / fname, dpi=150); plt.close(fig)

    barrier_bars("RMSE", ylabel="Barrier RMSE (kcal/mol)", fname="barrier_rmse.png")
    barrier_bars("MAE",  ylabel="Barrier MAE  (kcal/mol)", fname="barrier_mae.png")
    barrier_bars("NMAE", ylabel="Barrier NMAE",           fname="barrier_nmae.png")
    barrier_bars("R2_det", hline=0.0, hl_label="mean-predictor",
                 ylabel="Barrier R²_det", fname="barrier_r2_det.png")

    # ===== Barrier parity plot (1 fig, 3 panels for 3 models) =====
    keys = list(aggregated.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        bp = aggregated[k]["barrier_pooled"]
        yt = np.array(bp["y_true"]); yp = np.array(bp["y_pred"])
        ax.scatter(yt, yp, s=8, alpha=0.5, color=MODEL_COLOR[k], linewidths=0)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        pad = 0.04 * (hi - lo)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls="--", color="#888", lw=1, label="y = x")
        lr = sstats.linregress(yt, yp)
        xs = np.linspace(yt.min(), yt.max(), 100)
        ax.plot(xs, lr.slope * xs + lr.intercept, color="#C99A2E", lw=1.5, label="linear fit")
        met = channel_metrics(yt, yp)
        ax.text(0.04, 0.96,
                f"MAE={met['MAE']:.2f}\nRMSE={met['RMSE']:.2f}\nR²={met['R2_det']:.2f}\nslope={lr.slope:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#888", alpha=0.85, linewidth=0.5))
        ax.set_title(f"{MODEL_LABEL[k]} — Barrier"); ax.set_xlabel("y_true Σch (kcal/mol)")
        ax.grid(True, alpha=0.15, ls="--")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("y_pred Σch (kcal/mol)")
    fig.suptitle(f"Barrier parity (member 0 × 5 folds, no-OOD + {len(excluded_rids)} outliers removed)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "barrier_parity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ===== Bar plots (5 channels + barrier as 6th group) =====
    CHANS_WITH_BARRIER = CHANS + ["barrier"]
    def cmp_bars(metric, hline=None, hl_label=None, ylabel=None, fname=None):
        keys = list(aggregated.keys())
        n = len(keys)
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = np.arange(len(CHANS_WITH_BARRIER)); w = 0.8 / n
        for i, k in enumerate(keys):
            off = (i - (n - 1) / 2) * w
            sm = aggregated[k]["summary"]
            bs = aggregated[k]["barrier_summary"]
            vals = [sm.loc[c, (metric, "mean")] for c in CHANS] + [float(bs.loc["mean", metric])]
            errs = [sm.loc[c, (metric, "std")] for c in CHANS] + [float(bs.loc["std", metric])]
            ax.bar(x + off, vals, w, yerr=errs, label=MODEL_LABEL[k],
                   color=MODEL_COLOR[k], capsize=3, edgecolor="black", linewidth=0.4)
        if hline is not None:
            ax.axhline(hline, ls="--", color="grey", label=hl_label)
        ax.set_xticks(x); ax.set_xticklabels(CHANS_WITH_BARRIER)
        # Visually separate "barrier" from per-channel with a light divider
        ax.axvline(len(CHANS) - 0.5, ls=":", color="grey", lw=0.8, alpha=0.6)
        ax.set_ylabel(ylabel or metric)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.15, linestyle="--", axis="y")
        fig.tight_layout(); fig.savefig(OUT_FIG / fname, dpi=150); plt.close(fig)

    cmp_bars("NMAE", hline=1.0, hl_label="mean-predictor",
             ylabel="NMAE = MAE / MAD(y_true)", fname="compare_nmae.png")
    cmp_bars("R2_det", hline=0.0, hl_label="mean-predictor",
             ylabel="R²_det", fname="compare_r2_det.png")
    cmp_bars("MAE", ylabel="MAE (kcal/mol)", fname="compare_mae.png")
    cmp_bars("RMSE", ylabel="RMSE (kcal/mol)", fname="compare_rmse.png")
    cmp_bars("tail_ratio", hline=1.2533, hl_label="Gaussian",
             ylabel="RMSE / MAE", fname="compare_tail_ratio.png")

    # ===== Per-model parity =====
    for k in aggregated:
        pooled = aggregated[k]["pooled"]
        n_cells = len(aggregated[k]["meta"])
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        for col, ch in enumerate(CHANS):
            ax = axes[col]
            yt = np.array(pooled[ch]["y_true"]); yp = np.array(pooled[ch]["y_pred"])
            ax.scatter(yt, yp, s=7, alpha=0.5, color=MODEL_COLOR[k], linewidths=0)
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            pad = 0.04 * (hi - lo)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls="--", color="#888", lw=1, zorder=0)
            lr = sstats.linregress(yt, yp)
            xs = np.linspace(yt.min(), yt.max(), 100)
            ax.plot(xs, lr.slope*xs+lr.intercept, color="#C99A2E", lw=1.5)
            met = channel_metrics(yt, yp)
            ax.text(0.04, 0.96,
                    f"MAE={met['MAE']:.2f}\nNMAE={met['NMAE']:.2f}\nR²={met['R2_det']:.2f}\nslope={lr.slope:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#888", alpha=0.85, linewidth=0.5))
            ax.set_title(ch); ax.set_xlabel("y_true (kcal/mol)")
            ax.grid(True, alpha=0.15, ls="--")
        axes[0].set_ylabel("y_pred (kcal/mol)")
        fig.suptitle(f"Parity — {MODEL_LABEL[k]} (member 0 × 5 folds, no-OOD + {len(excluded_rids)} outliers removed)",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_FIG / f"{k}_parity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ===== Combined parity grid (5 channels + barrier as 6th column) =====
    keys = list(aggregated.keys())
    n_rows = len(keys)
    n_cols = len(CHANS) + 1  # +1 for barrier
    col_labels = CHANS + ["barrier"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows), squeeze=False)
    for r, k in enumerate(keys):
        pooled = aggregated[k]["pooled"]
        barrier_pooled = aggregated[k]["barrier_pooled"]
        for c, ch in enumerate(col_labels):
            ax = axes[r, c]
            if ch == "barrier":
                yt = np.array(barrier_pooled["y_true"])
                yp = np.array(barrier_pooled["y_pred"])
            else:
                yt = np.array(pooled[ch]["y_true"])
                yp = np.array(pooled[ch]["y_pred"])
            ax.scatter(yt, yp, s=7, alpha=0.5, color=MODEL_COLOR[k], linewidths=0)
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            pad = 0.04 * (hi - lo)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls="--", color="#888", lw=1, zorder=0,
                    label="y = x")
            lr = sstats.linregress(yt, yp)
            xs = np.linspace(yt.min(), yt.max(), 100)
            ax.plot(xs, lr.slope * xs + lr.intercept, color="#C99A2E", lw=1.5,
                    label="linear fit")
            met = channel_metrics(yt, yp)
            ax.text(0.04, 0.96,
                    f"MAE={met['MAE']:.2f}\nNMAE={met['NMAE']:.2f}\nR²={met['R2_det']:.2f}\nslope={lr.slope:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#888", alpha=0.85, linewidth=0.5))
            if r == 0:
                ax.set_title(ch, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABEL[k]}\ny_pred", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("y_true (kcal/mol)", fontsize=9)
            ax.grid(True, alpha=0.15, ls="--")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "compare_parity_grid.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # (Old combined 6-panel figure disabled — per-channel plots now include
    #  barrier as their 6th group / column, so a separate combined figure is
    #  redundant.)
    _COMBINED_DISABLED = """
    # ===== Combined 6-panel figure: (NMAE / RMSE / Parity) × (per-channel / barrier) =====
    keys = list(aggregated.keys())
    n_mod = len(keys)
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # ---- Row 0: NMAE ---------------------------------------------------------
    ax = axes[0, 0]  # per-channel NMAE bars
    x = np.arange(len(CHANS)); w = 0.8 / n_mod
    for i, k in enumerate(keys):
        off = (i - (n_mod - 1) / 2) * w
        sm = aggregated[k]["summary"]
        vals = [sm.loc[c, ("NMAE", "mean")] for c in CHANS]
        errs = [sm.loc[c, ("NMAE", "std")] for c in CHANS]
        ax.bar(x + off, vals, w, yerr=errs, label=MODEL_LABEL[k],
               color=MODEL_COLOR[k], capsize=3, edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, ls="--", color="grey", label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(CHANS)
    ax.set_ylabel("NMAE (per-channel)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.15, ls="--", axis="y")

    ax = axes[0, 1]  # barrier NMAE
    xb = np.arange(n_mod)
    vals = [aggregated[k]["barrier_summary"].loc["mean", "NMAE"] for k in keys]
    errs = [aggregated[k]["barrier_summary"].loc["std",  "NMAE"] for k in keys]
    ax.bar(xb, vals, 0.6, yerr=errs,
           color=[MODEL_COLOR[k] for k in keys], capsize=4, edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, ls="--", color="grey", label="mean-predictor")
    ax.set_xticks(xb); ax.set_xticklabels([MODEL_LABEL[k] for k in keys])
    ax.set_ylabel("NMAE (barrier)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.15, ls="--", axis="y")

    # ---- Row 1: RMSE ---------------------------------------------------------
    ax = axes[1, 0]  # per-channel RMSE
    for i, k in enumerate(keys):
        off = (i - (n_mod - 1) / 2) * w
        sm = aggregated[k]["summary"]
        vals = [sm.loc[c, ("RMSE", "mean")] for c in CHANS]
        errs = [sm.loc[c, ("RMSE", "std")] for c in CHANS]
        ax.bar(x + off, vals, w, yerr=errs, label=MODEL_LABEL[k],
               color=MODEL_COLOR[k], capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(CHANS)
    ax.set_ylabel("RMSE per-channel (kcal/mol)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.15, ls="--", axis="y")

    ax = axes[1, 1]  # barrier RMSE
    vals = [aggregated[k]["barrier_summary"].loc["mean", "RMSE"] for k in keys]
    errs = [aggregated[k]["barrier_summary"].loc["std",  "RMSE"] for k in keys]
    ax.bar(xb, vals, 0.6, yerr=errs,
           color=[MODEL_COLOR[k] for k in keys], capsize=4, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xb); ax.set_xticklabels([MODEL_LABEL[k] for k in keys])
    ax.set_ylabel("RMSE (barrier, kcal/mol)")
    ax.grid(True, alpha=0.15, ls="--", axis="y")

    # ---- Row 2: Parity (m1/m2/m3 overlaid on the same axes) ------------------
    # Left cell: per-channel parity, z-scored per channel per model so scales
    # don't distort — dimensionless. Right cell: barrier parity in raw units.
    ax = axes[2, 0]
    for k in keys:
        pooled = aggregated[k]["pooled"]
        yt_all, yp_all = [], []
        for ch in CHANS:
            yt = np.array(pooled[ch]["y_true"]); yp = np.array(pooled[ch]["y_pred"])
            mu, sd = float(yt.mean()), float(yt.std())
            if sd > 0:
                yt_all.extend(((yt - mu) / sd).tolist())
                yp_all.extend(((yp - mu) / sd).tolist())
        yt_arr = np.array(yt_all); yp_arr = np.array(yp_all)
        ax.scatter(yt_arr, yp_arr, s=4, alpha=0.25, color=MODEL_COLOR[k],
                   linewidths=0, label=MODEL_LABEL[k])
    lim = 8
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#666", lw=1, label="y = x")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("y_true (z-scored per channel)")
    ax.set_ylabel("y_pred (z-scored per channel)")
    ax.legend(loc="best", fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.15, ls="--")

    ax = axes[2, 1]  # barrier parity (raw kcal/mol), 3 models overlaid
    lo_all, hi_all = np.inf, -np.inf
    for k in keys:
        bp = aggregated[k]["barrier_pooled"]
        yt = np.array(bp["y_true"]); yp = np.array(bp["y_pred"])
        ax.scatter(yt, yp, s=8, alpha=0.4, color=MODEL_COLOR[k],
                   linewidths=0, label=MODEL_LABEL[k])
        lo_all = min(lo_all, float(min(yt.min(), yp.min())))
        hi_all = max(hi_all, float(max(yt.max(), yp.max())))
    pad = 0.04 * (hi_all - lo_all)
    ax.plot([lo_all-pad, hi_all+pad], [lo_all-pad, hi_all+pad],
            ls="--", color="#666", lw=1, label="y = x")
    ax.set_xlim(lo_all - pad, hi_all + pad); ax.set_ylim(lo_all - pad, hi_all + pad)
    ax.set_xlabel("Σ channels — y_true (kcal/mol)")
    ax.set_ylabel("Σ channels — y_pred (kcal/mol)")
    ax.legend(loc="best", fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.15, ls="--")

    fig.tight_layout()
    fig.savefig(OUT_FIG / "compare_6panel_nmae_rmse_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    """

    # remove the old combined file if it exists (superseded)
    old = OUT_FIG / "compare_6panel_nmae_rmse_parity.png"
    if old.exists():
        old.unlink()

    # ===== Markdown =====
    md = ["# m1 vs m2 vs m3 — 5 folds × member 0, no-OOD + parity outliers removed\n"]
    md.append("Same setup as the previous member-0 comparison; HP/seed/split identical across m1/m2/m3.")
    md.append("")
    md.append(f"**Parity-outlier removal**: pooled per-channel residuals; modified Z (median+MAD) > {ROBUST_Z_THRESHOLD} ⇒ exclude.")
    md.append(f"**Excluded reactions**: {len(excluded_rids)} unique rids (applied uniformly to m1, m2, m3 so the test set stays matched).")
    md.append("")
    md.append("Most catastrophic causes:")
    md.append("- `dipolar_003220` — m2 xTB cache returned unphysical E_int = −764.6 kcal/mol; m3 xtb_extra cache missing (archive path gone).")
    md.append("- Several `qmrxn20_*` reactions whose source_dir under `archive/al_v*_baseline_*` is no longer on disk → m3's 6 extras were NaN-imputed.")
    md.append("")
    md.append("## Per-channel metrics (mean ± std across 5 folds, outliers removed)\n")
    md.append("| channel | metric | " + " | ".join(MODEL_LABEL[k] for k in aggregated) + " |")
    md.append("|---|---|" + "|".join(["---"] * len(aggregated)) + "|")
    for ch in CHANS:
        for metric, fmt in [("R2_det", ".3f"), ("NMAE", ".3f"),
                            ("MAE", ".2f"), ("RMSE", ".2f"), ("tail_ratio", ".3f")]:
            row = f"| {ch} | {metric} |"
            for k in aggregated:
                sm = aggregated[k]["summary"]
                row += f" {sm.loc[ch, (metric,'mean')]:{fmt}} ± {sm.loc[ch, (metric,'std')]:{fmt}} |"
            md.append(row)

    md.append("\n## Per-fold test sizes (after outlier removal)\n")
    md.append("| fold | n_test_original | n_test_kept |")
    md.append("|---|---|---|")
    if "m1" in aggregated:
        for _, r in aggregated["m1"]["meta"].iterrows():
            md.append(f"| {int(r['fold'])} | {int(r['n_test_original'])} | {int(r['n_test_kept'])} |")

    md.append("\n## Excluded reaction IDs")
    md.append(f"`results_compare_m1_m2_m3_member0_noOutliers/excluded_rids.json` ({len(excluded_rids)} total)")
    for rid in excluded_rids:
        md.append(f"- `{rid}` ({derive_family(rid)})")

    md.append(f"\n## Figures (`figures_compare_m1_m2_m3_member0_noOutliers/`)")
    md.append("- `compare_nmae.png`, `compare_r2_det.png`, `compare_mae.png`, `compare_tail_ratio.png`")
    md.append("- `<model>_parity.png`, `compare_parity_grid.png`")
    REPORT_MD.write_text("\n".join(md))
    print(f"\n✓ Results : {OUT_RES}")
    print(f"✓ Figures : {OUT_FIG}")
    print(f"✓ Report  : {REPORT_MD}")


if __name__ == "__main__":
    main()
