"""SPEC_03 - Maximum classical physics-only baseline (m3 24-d, no MACE/delta).

Benchmarks per channel + barrier on the same 5-fold split:
  ridge, lasso, enet, xgb

Barrier via two routes:
  (a) sum of per-channel predictions
  (b) direct model trained on Σy

Head-to-head vs current M_bdelta pulled from m3/code/trackB_v7_lowlr_no_ood_geom6.

Outputs to spec/spec03_bmax/{results, figures}.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v7/features_v7_delta_m3.pt")
SPLIT_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v7/trackB_no_ood")
OUT_RES = REPO / "spec/spec03_bmax/results"
OUT_FIG = REPO / "spec/spec03_bmax/figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
NAVY = "#1f4e79"
CV_INNER = 5
SEED = 42

# SPEC_01 alpha* placeholder - actual value read from spec01 output if available
def spec01_alpha_star():
    p = REPO / "spec/spec01_alpha/results/alpha_selection.csv"
    if not p.exists():
        return {ch: 1.0 for ch in CHANNELS + ["barrier"]}
    df = pd.read_csv(p)
    out = {}
    for ch in CHANNELS + ["barrier"]:
        sub = df[(df.channel == ch) & (df.rule == "alpha_star_cv")]
        out[ch] = float(sub.alpha.iloc[0]) if len(sub) else 1.0
    return out


def load_data():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = b["reaction_ids"]
    X = b["descriptors"].numpy()
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
    return X, Y, folds, rids


def nmae(yt, yp):
    mad = np.mean(np.abs(yt - yt.mean()))
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def slope(yt, yp):
    a = yt - yt.mean(); b = yp - yp.mean()
    d = float(np.sum(a * a))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def make_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def fit_ridge(Xtr, ytr, alpha_star):
    return make_pipeline(Ridge(alpha=alpha_star)).fit(Xtr, ytr)


def fit_lasso(Xtr, ytr):
    return make_pipeline(LassoCV(cv=CV_INNER, random_state=SEED, max_iter=20000)).fit(Xtr, ytr)


def fit_enet(Xtr, ytr):
    return make_pipeline(ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                      cv=CV_INNER, random_state=SEED,
                                      max_iter=20000)).fit(Xtr, ytr)


def fit_xgb(Xtr, ytr):
    grid = {"model__n_estimators": [200, 500, 1000],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__reg_lambda": [1.0, 5.0]}
    est = make_pipeline(XGBRegressor(random_state=SEED, tree_method="hist",
                                     objective="reg:squarederror", verbosity=0,
                                     n_jobs=1))
    # Randomized subset to keep runtime bounded
    from sklearn.model_selection import RandomizedSearchCV
    gs = RandomizedSearchCV(est, grid, n_iter=24, cv=CV_INNER,
                            scoring="neg_mean_absolute_error",
                            random_state=SEED, n_jobs=-1)
    gs.fit(Xtr, ytr); return gs.best_estimator_


METHODS = {
    "ridge": lambda Xt, yt, a: fit_ridge(Xt, yt, a),
    "lasso": lambda Xt, yt, a: fit_lasso(Xt, yt),
    "enet":  lambda Xt, yt, a: fit_enet(Xt, yt),
    "xgb":   lambda Xt, yt, a: fit_xgb(Xt, yt),
}


def load_neural_m3():
    """Pull m3 v7 test-pred and y_true across all completed member*.json."""
    root = REPO / "m3/code/trackB_v7_lowlr_no_ood_geom6/m1_delta"
    if not root.exists(): return None
    per_ch = {c: {"yt": [], "yp": []} for c in CHANNELS}
    barrier = {"yt": [], "yp": []}
    for f in root.glob("fold*/member*.json"):
        c = json.load(open(f))
        yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
        for i, ch in enumerate(CHANNELS):
            per_ch[ch]["yt"].append(yt[:, i]); per_ch[ch]["yp"].append(yp[:, i])
        barrier["yt"].append(yt.sum(axis=1))
        barrier["yp"].append(yp.sum(axis=1))
    if not any(per_ch[c]["yt"] for c in CHANNELS): return None
    rows = []
    for ch in CHANNELS:
        yt = np.concatenate(per_ch[ch]["yt"]); yp = np.concatenate(per_ch[ch]["yp"])
        rows.append({"model": "M_bdelta_v7_neural", "channel": ch,
                     "NMAE": nmae(yt, yp), "RMSE": rmse(yt, yp),
                     "R2": r2(yt, yp), "slope": slope(yt, yp)})
    yt = np.concatenate(barrier["yt"]); yp = np.concatenate(barrier["yp"])
    rows.append({"model": "M_bdelta_v7_neural", "channel": "barrier",
                 "NMAE": nmae(yt, yp), "RMSE": rmse(yt, yp),
                 "R2": r2(yt, yp), "slope": slope(yt, yp)})
    return pd.DataFrame(rows)


def main():
    X, Y, folds, rids = load_data()
    n, D = X.shape
    print(f"[m3] N={n}, D={D}")
    alpha_star = spec01_alpha_star()
    print(f"SPEC_01 alpha_star: {alpha_star}")

    Y_ch = {c: Y[:, i] for i, c in enumerate(CHANNELS)}
    Y_ch["barrier_direct"] = Y.sum(axis=1)

    # Task 3.1 + 3.2: run each method per channel/barrier, folds -> avg metrics
    rows = []
    all_preds = {}  # method -> channel -> [(y_true, y_pred)] concatenated
    for method in METHODS:
        all_preds[method] = {}
        for ch, y in Y_ch.items():
            fold_preds_yt, fold_preds_yp = [], []
            for tr, te in folds:
                a = alpha_star.get(ch if ch != "barrier_direct" else "barrier", 1.0)
                est = METHODS[method](X[tr], y[tr], a)
                yp = est.predict(X[te])
                fold_preds_yt.append(y[te]); fold_preds_yp.append(yp)
            yt = np.concatenate(fold_preds_yt); yp = np.concatenate(fold_preds_yp)
            all_preds[method][ch] = (yt, yp)
            rows.append({"model": method, "channel": ch,
                         "NMAE": nmae(yt, yp), "RMSE": rmse(yt, yp),
                         "R2": r2(yt, yp), "slope": slope(yt, yp)})
            print(f"  {method} {ch}: NMAE={rows[-1]['NMAE']:.3f} RMSE={rows[-1]['RMSE']:.2f}")

    # Sum-of-channels barrier per method
    for method in METHODS:
        yt = np.zeros_like(all_preds[method][CHANNELS[0]][0])
        yp = np.zeros_like(all_preds[method][CHANNELS[0]][1])
        for ch in CHANNELS:
            yt_ch, yp_ch = all_preds[method][ch]
            yt = yt + yt_ch; yp = yp + yp_ch
        rows.append({"model": method, "channel": "barrier_sum",
                     "NMAE": nmae(yt, yp), "RMSE": rmse(yt, yp),
                     "R2": r2(yt, yp), "slope": slope(yt, yp)})
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT_RES / "baseline_leaderboard.csv", index=False)

    # Task 3.2 comparison table
    barrier_rows = lb[lb.channel.isin(["barrier_direct", "barrier_sum"])].copy()
    barrier_rows.to_csv(OUT_RES / "barrier_routes.csv", index=False)

    # Task 3.4 vs neural
    neu = load_neural_m3()
    if neu is not None:
        best_by_ch = {}
        for ch in CHANNELS + ["barrier_direct", "barrier_sum"]:
            sub = lb[lb.channel == ch].sort_values("NMAE")
            best_by_ch[ch] = sub.iloc[0].to_dict()
        # Compare vs neural (which reports "barrier" = sum)
        comp_rows = []
        for ch in CHANNELS + ["barrier"]:
            nrow = neu[neu.channel == ch].iloc[0]
            if ch == "barrier":
                bs = best_by_ch.get("barrier_sum", best_by_ch.get("barrier_direct"))
            else:
                bs = best_by_ch[ch]
            comp_rows.append({
                "channel": ch,
                "best_classical_model": bs["model"],
                "classical_NMAE": bs["NMAE"],
                "neural_NMAE": nrow["NMAE"],
                "delta_NMAE (classical - neural)": bs["NMAE"] - nrow["NMAE"],
            })
        pd.DataFrame(comp_rows).to_csv(OUT_RES / "best_vs_neural.csv", index=False)

    # ============================ Figure: NMAE bars ============================
    channels_plot = CHANNELS + ["barrier_sum"]
    methods_plot = list(METHODS.keys())
    x = np.arange(len(channels_plot))
    width = 0.14
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, m_ in enumerate(methods_plot):
        vals = [lb[(lb.model == m_) & (lb.channel == ch)].NMAE.iloc[0] for ch in channels_plot]
        ax.bar(x + (i - len(methods_plot) / 2) * width, vals, width, label=m_,
               edgecolor="white", lw=0.4)
    if neu is not None:
        vals = []
        for ch in channels_plot:
            k = "barrier" if ch == "barrier_sum" else ch
            row_ = neu[neu.channel == k]
            vals.append(row_.NMAE.iloc[0] if len(row_) else np.nan)
        ax.plot(x, vals, "k*", ms=12, label="M_bdelta (neural)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_xticks(x); ax.set_xticklabels(channels_plot)
    ax.set_ylabel("NMAE (5-fold CV)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("SPEC_03 - Classical baselines vs neural M_bdelta (m3, 776 rxn v7)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "baseline_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # summary
    lines = ["# SPEC_03 summary", "",
             "## Best per channel (classical baselines)", "",
             "| channel | best method | NMAE | RMSE |", "|---|---|---|---|"]
    for ch in channels_plot:
        sub = lb[lb.channel == ch].sort_values("NMAE").iloc[0]
        lines.append(f"| {ch} | {sub.model} | {sub.NMAE:.3f} | {sub.RMSE:.2f} |")
    lines += ["", "## Barrier via sum-of-channels vs direct",
              "See barrier_routes.csv.", "",
              "## Head-to-head vs neural (v7 m3)",
              "See best_vs_neural.csv."]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}")


if __name__ == "__main__":
    main()
