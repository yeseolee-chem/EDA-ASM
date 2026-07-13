"""Head-to-head: base 24d (SPEC 3 XGB) vs all extras (d25+d26+d27+d28) — 2 variants only.

Trains both variants on same 5-fold split, saves per-rxn OOF predictions,
outputs NMAE / RMSE bar charts + parity grid.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt")
SPLIT_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9")
D25_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_d25_refR.parquet"
D26_28_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet"
OUT_RES = REPO / "spec/spec05_d25_sum/results"
OUT_FIG = REPO / "spec/spec05_d25_sum/figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CH = ["strain","Pauli","elst","oi","disp"]
COLORS = {"base_24d": "#4a4a4a", "all_extras": "#d62728"}
XGB_HP = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
              subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
              reg_alpha=0.0, reg_lambda=1.0, tree_method="hist", n_jobs=4, random_state=42)

def load():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = list(b["reaction_ids"])
    X24 = b["descriptors"].numpy()  # 24d
    Y = b["labels"].numpy()
    folds = []
    for i in range(5):
        fd = SPLIT_ROOT / f"fold{i}"
        te = set(json.load(open(fd/"test_rids.json")))
        tf = sorted(fd.glob("size_*.json"))[-1]
        tr = set(json.load(open(tf)))
        tr_i = [j for j,r in enumerate(rids) if r in tr]
        te_i = [j for j,r in enumerate(rids) if r in te]
        folds.append((tr_i, te_i))
    return rids, X24, Y, folds

def attach_extras(rids, X24):
    """Concat X24 + d25 + d26,d27,d28 (fill NaN with column median)."""
    d25 = pd.read_parquet(D25_PQ).set_index("reaction_id")["d25"].reindex(rids).values.astype(np.float32)
    p = pd.read_parquet(D26_28_PQ).set_index("reaction_id")[["d26","d27","d28"]].reindex(rids).astype(np.float32)
    d25 = np.where(np.isnan(d25), np.nanmedian(d25), d25).reshape(-1,1)
    for c in p.columns:
        v = p[c].values
        p[c] = np.where(np.isnan(v), np.nanmedian(v), v)
    return np.hstack([X24, d25, p.values])  # 28-d

def run(X, Y, folds, name):
    """Train XGB per-channel per-fold, return pooled OOF."""
    n, k = Y.shape
    oof = np.full_like(Y, np.nan)
    for f, (tr, te) in enumerate(folds):
        for c in range(k):
            m = XGBRegressor(**XGB_HP)
            m.fit(X[tr], Y[tr, c])
            oof[te, c] = m.predict(X[te])
        print(f"  [{name}] fold{f} done")
    return oof

def mad(y): return float(np.mean(np.abs(y - np.mean(y))))
def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def main():
    rids, X24, Y, folds = load()
    print(f"n={len(rids)} X24={X24.shape} Y={Y.shape}")
    X28 = attach_extras(rids, X24)
    print(f"X28={X28.shape}")

    print("=== base_24d ===")
    oof_base = run(X24, Y, folds, "base")
    print("=== all_extras (28d) ===")
    oof_all = run(X28, Y, folds, "all")

    # metrics
    rows=[]
    for var, oof in [("base_24d", oof_base), ("all_extras", oof_all)]:
        for c, ch in enumerate(CH):
            m = mae(Y[:,c], oof[:,c])
            rows.append(dict(variant=var, channel=ch,
                             NMAE=m/mad(Y[:,c]), RMSE=rmse(Y[:,c], oof[:,c])))
        # barrier (sum-of-channels)
        b_true = Y.sum(1); b_pred = oof.sum(1)
        m = mae(b_true, b_pred)
        rows.append(dict(variant=var, channel="barrier",
                         NMAE=m/mad(b_true), RMSE=rmse(b_true, b_pred)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES/"compare_2way_metrics.csv", index=False)
    print(df.pivot(index="channel", columns="variant", values="NMAE"))

    # save OOF for parity
    np.savez(OUT_RES/"compare_2way_oof.npz",
             rids=np.array(rids), y_true=Y,
             base=oof_base, all_extras=oof_all)

    # ---- bar charts ----
    CHo = CH + ["barrier"]
    x = np.arange(len(CHo)); width = 0.35
    fig, ax = plt.subplots(figsize=(11,5))
    for i, var in enumerate(["base_24d", "all_extras"]):
        vals = [df[(df.variant==var)&(df.channel==c)]["NMAE"].values[0] for c in CHo]
        ax.bar(x + (i-0.5)*width, vals, width, label=var, color=COLORS[var],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CHo)
    ax.set_ylabel("NMAE"); ax.set_title("SPEC 5 - base 24d vs +d25/26/27/28 (v9, 783 rxns)")
    ax.axhline(1.0, ls="--", c="gray", alpha=0.5, label="mean-predictor")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG/"compare_2way_NMAE.png", dpi=140)

    fig, ax = plt.subplots(figsize=(11,5))
    for i, var in enumerate(["base_24d", "all_extras"]):
        vals = [df[(df.variant==var)&(df.channel==c)]["RMSE"].values[0] for c in CHo]
        ax.bar(x + (i-0.5)*width, vals, width, label=var, color=COLORS[var],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CHo)
    ax.set_ylabel("RMSE (kcal/mol)"); ax.set_title("SPEC 5 - base 24d vs +d25/26/27/28 (v9, 783 rxns)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG/"compare_2way_RMSE.png", dpi=140)

    # ---- parity grid ----
    fig, axes = plt.subplots(2, len(CHo), figsize=(3.2*len(CHo), 6.5))
    for row, (var, oof) in enumerate([("base_24d", oof_base), ("all_extras", oof_all)]):
        for col, ch in enumerate(CHo):
            ax = axes[row, col]
            if ch == "barrier":
                yt = Y.sum(1); yp = oof.sum(1)
            else:
                yt = Y[:, CH.index(ch)]; yp = oof[:, CH.index(ch)]
            ax.scatter(yt, yp, s=6, alpha=0.5, color=COLORS[var], edgecolors="none")
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo,hi], [lo,hi], "--", color="gray", lw=0.8)
            try:
                m, b = np.polyfit(yt, yp, 1)
                xx = np.array([lo,hi]); ax.plot(xx, m*xx+b, "-", color="orange", lw=1.2)
                r2 = 1 - np.sum((yt-yp)**2) / np.sum((yt - yt.mean())**2)
                stat = f"NMAE={mae(yt,yp)/mad(yt):.2f}\nR²={r2:.2f}\nslope={m:.2f}"
            except Exception:
                stat = ""
            ax.text(0.03, 0.97, stat, transform=ax.transAxes, fontsize=8, va="top",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
            if row == 0: ax.set_title(ch, fontsize=11)
            if col == 0: ax.set_ylabel(f"{var}\ny_pred", fontsize=10)
            if row == 1: ax.set_xlabel("y_true (kcal/mol)", fontsize=9)
            ax.grid(alpha=0.3, lw=0.4)
    fig.suptitle("SPEC 5 - Parity: base 24d vs all extras (v9, 783 rxns, 5-fold pooled OOF)", y=1.005)
    fig.tight_layout(); fig.savefig(OUT_FIG/"compare_2way_parity.png", dpi=130, bbox_inches="tight")
    print(f"wrote figures: compare_2way_NMAE.png, _RMSE.png, _parity.png")

if __name__ == "__main__":
    main()
