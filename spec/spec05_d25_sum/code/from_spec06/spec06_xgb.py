"""SPEC_06 T3 - XGB per-channel NMAE with channel-matched proxies added.

Uses m3 v9 24-d bundle + adds d26/d27/d28 (SPEC_06 T1) + optional d25 (SPEC_05).
Variants:
  base_24d           : m3 24-d (SPEC_03 xgb reference)
  base_25d           : 24-d + d25 (SPEC_05 M1)
  base_24d_d26       : 24-d + d26 (elst-matched)
  base_24d_d27       : 24-d + d27 (Pauli-matched)
  base_24d_d28       : 24-d + d28 (oi-matched)
  base_24d_d26_27_28 : 24-d + d26 + d27 + d28 (three-channel proxies)
  base_25d_d26_27_28 : 24-d + d25 + d26 + d27 + d28 (all four proxies, 28-d)

Pooled 5-fold OOF NMAE per channel + barrier(sum).
"""
from __future__ import annotations
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path(os.environ.get("BUNDLE_PT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
SPLIT_ROOT = Path(os.environ.get("SPLIT_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9"))
D25_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_d25_refR.parquet"
D26_28_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet"
OUT_RES = REPO / "spec/spec05_d25_sum/results"
OUT_FIG = REPO / "spec/spec05_d25_sum/figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
NAVY = "#1f4e79"
SEED = 42


def nmae(yt, yp):
    mad = np.mean(np.abs(yt - yt.mean()))
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss / (tot + 1e-12))


def slope(yt, yp):
    a = yt - yt.mean(); b = yp - yp.mean(); d = float(np.sum(a * a))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def xgb_predict(X_tr, y_tr, X_te, seed):
    est = XGBRegressor(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        random_state=seed, n_jobs=4, objective="reg:squarederror",
        verbosity=0,
    )
    est.fit(X_tr, y_tr)
    return est.predict(X_te)


def load_data():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy()
    Y = b["labels"].numpy()
    r2i = {r: i for i, r in enumerate(rids)}
    folds = []
    for i in range(5):
        fd = SPLIT_ROOT / f"fold{i}"
        te = json.load(open(fd / "test_rids.json"))
        tf = sorted(fd.glob("size_*.json"),
                    key=lambda p: int(p.stem.split("_")[1]), reverse=True)[0]
        tr = json.load(open(tf))
        folds.append((np.array([r2i[r] for r in tr if r in r2i]),
                      np.array([r2i[r] for r in te if r in r2i])))
    return rids, X24, Y, folds


def attach_col(rids, parquet_path, col):
    """Return (col_values, ok_mask); NaN fills for failed SCF."""
    df = pd.read_parquet(parquet_path).set_index("reaction_id")
    vals, ok = [], []
    for r in rids:
        if r in df.index and bool(df.loc[r, "scf_ok"]):
            v = df.loc[r, col]
            if pd.isna(v):
                vals.append(0.0); ok.append(False)
            else:
                vals.append(float(v)); ok.append(True)
        else:
            vals.append(0.0); ok.append(False)
    return np.array(vals), np.array(ok, dtype=bool)


def evaluate(X, Y, folds, ok_mask, tag):
    """5-fold pooled OOF NMAE per channel."""
    pooled_yt, pooled_yp = [], []
    for f_i, (tr, te) in enumerate(folds):
        tr = tr[ok_mask[tr]]; te = te[ok_mask[te]]
        preds = np.zeros((len(te), 5))
        for c in range(5):
            preds[:, c] = xgb_predict(X[tr], Y[tr, c], X[te], seed=SEED + c)
        pooled_yt.append(Y[te]); pooled_yp.append(preds)
    yt = np.concatenate(pooled_yt); yp = np.concatenate(pooled_yp)
    rows = []
    for c_i, ch in enumerate(CHANNELS):
        rows.append({"variant": tag, "channel": ch,
                     "NMAE": nmae(yt[:, c_i], yp[:, c_i]),
                     "RMSE": rmse(yt[:, c_i], yp[:, c_i]),
                     "R2":   r2(yt[:, c_i], yp[:, c_i]),
                     "slope": slope(yt[:, c_i], yp[:, c_i])})
    bt = yt.sum(axis=1); bp = yp.sum(axis=1)
    rows.append({"variant": tag, "channel": "barrier",
                 "NMAE": nmae(bt, bp), "RMSE": rmse(bt, bp),
                 "R2": r2(bt, bp), "slope": slope(bt, bp)})
    return rows


def main():
    rids, X24, Y, folds = load_data()
    n = len(rids)
    print(f"[m3 v9] N={n}, D=24")

    d25, ok25 = attach_col(rids, D25_PQ, "d25")
    d26, ok26 = attach_col(rids, D26_28_PQ, "d26")
    d27, ok27 = attach_col(rids, D26_28_PQ, "d27")
    d28, ok28 = attach_col(rids, D26_28_PQ, "d28")
    print(f"d25 ok:{ok25.sum()}/{n}  d26 ok:{ok26.sum()}/{n}  "
          f"d27 ok:{ok27.sum()}/{n}  d28 ok:{ok28.sum()}/{n}")

    variants = {
        "base_24d":           (X24,                                  np.ones(n, dtype=bool)),
        "base_25d":           (np.hstack([X24, d25[:, None]]),        ok25),
        "base_24d_d26":       (np.hstack([X24, d26[:, None]]),        ok26),
        "base_24d_d27":       (np.hstack([X24, d27[:, None]]),        ok27),
        "base_24d_d28":       (np.hstack([X24, d28[:, None]]),        ok28),
        "base_24d_d26_27_28": (np.hstack([X24, d26[:, None], d27[:, None], d28[:, None]]),
                               ok26 & ok27 & ok28),
        "base_25d_d26_27_28": (np.hstack([X24, d25[:, None], d26[:, None], d27[:, None], d28[:, None]]),
                               ok25 & ok26 & ok27 & ok28),
    }

    all_rows = []
    for tag, (X, ok) in variants.items():
        print(f"\n== {tag}  D={X.shape[1]}  n_ok={ok.sum()} ==", flush=True)
        rows = evaluate(X, Y, folds, ok, tag)
        for r in rows:
            r["n_used"] = int(ok.sum())
            print(f"   {r['channel']}: NMAE={r['NMAE']:.3f}  RMSE={r['RMSE']:.2f}", flush=True)
        all_rows += rows

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_RES / "channel_proxy_metrics.csv", index=False)

    # ============ Per-channel ablation deltas vs base_24d ============
    def get(tag, ch, met="NMAE"):
        return float(df[(df.variant == tag) & (df.channel == ch)][met].iloc[0])

    channels_plot = CHANNELS + ["barrier"]
    ablation = []
    ref = "base_24d"
    for ch in channels_plot:
        row = {"channel": ch, f"{ref}_NMAE": get(ref, ch)}
        for tag in variants:
            if tag == ref: continue
            row[f"{tag}_NMAE"] = get(tag, ch)
            row[f"delta_{tag}"] = get(tag, ch) - get(ref, ch)
        ablation.append(row)
    pd.DataFrame(ablation).to_csv(OUT_RES / "ablation_deltas.csv", index=False)

    # ============ Figure: grouped bar per channel ============
    tags_plot = list(variants.keys())
    x = np.arange(len(channels_plot)); w = 0.85 / len(tags_plot)
    colors = ["#7a7a7a", "#c05e2b", "#4ba36c", "#4b779a", "#8b3a62", "#d4a017", "#1f4e79"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, tag in enumerate(tags_plot):
        vals = [get(tag, ch) for ch in channels_plot]
        ax.bar(x + (i - len(tags_plot) / 2 + 0.5) * w, vals, w, label=tag,
               color=colors[i % len(colors)], edgecolor="white", lw=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel("NMAE (5-fold pooled OOF)")
    ax.legend(fontsize=8, loc="upper right", ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "channel_proxy_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ Delta bar (vs base_24d) ============
    fig, ax = plt.subplots(figsize=(14, 5.5))
    delta_tags = [t for t in tags_plot if t != ref]
    w2 = 0.85 / len(delta_tags)
    for i, tag in enumerate(delta_tags):
        vals = [get(tag, ch) - get(ref, ch) for ch in channels_plot]
        ax.bar(x + (i - len(delta_tags) / 2 + 0.5) * w2, vals, w2, label=tag,
               color=colors[(i + 1) % len(colors)], edgecolor="white", lw=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel("delta NMAE vs base_24d (negative = better)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "channel_proxy_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ REPORT ============
    labels = pd.read_parquet(REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet")
    prox = pd.read_parquet(D26_28_PQ)
    m = prox[prox["scf_ok"]].merge(labels[["reaction_id", "V_elst_kcal",
                                            "Pauli_kcal", "E_orb_kcal"]], on="reaction_id")
    corrs = ""
    if len(m) > 10:
        corrs = (f"- pearson(d26, V_elst) = {m.d26.corr(m.V_elst_kcal):+.3f}\n"
                 f"- pearson(d27, Pauli)  = {m.d27.corr(m.Pauli_kcal):+.3f}\n"
                 f"- pearson(d28, E_orb)  = {m.d28.corr(m.E_orb_kcal):+.3f}\n")
    lines = ["# SPEC_06 - channel-matched proxies (m3 v9, 783 rxns)", "",
             f"- N cohort: {n}",
             f"- d26/d27/d28 SCF ok: {ok26.sum()} / {ok27.sum()} / {ok28.sum()}",
             "", "## Physics sanity (pearson vs channel label)", corrs,
             "", "## NMAE per channel per variant (5-fold pooled OOF)", "",
             "| channel | " + " | ".join(tags_plot) + " |",
             "|" + "---|" * (len(tags_plot) + 1)]
    for ch in channels_plot:
        lines.append("| " + ch + " | " + " | ".join(f"{get(tag, ch):.3f}" for tag in tags_plot) + " |")
    lines += ["", "## Ablation deltas vs base_24d",
              "See ablation_deltas.csv (smaller / more negative = better)."]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}")


if __name__ == "__main__":
    main()
