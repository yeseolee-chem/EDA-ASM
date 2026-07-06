"""SPEC_03 — physics-only baseline maximization for m3.

4 classical predictors on the 24-d m3 descriptor matrix:
  - ridge   (α = 1)
  - lasso   (λ from 5-fold CV)
  - enet    (λ + l1_ratio from 5-fold CV)
  - xgb     (XGBoostRegressor with 5-fold-CV-tuned hp)

Saves fitted-model artefacts per method × fold × channel via joblib
under results/spec03_bmax/weights/, so the CSV numbers can be verified
by reloading + predicting without retraining.

Output layout (relative to repo root):

  results/spec03_bmax/
    baseline_leaderboard.csv
    baseline_bars.png           per-method NMAE bar
    barrier_routes.csv          Σ-of-channels vs direct
    best_vs_neural.png          best classical vs M_bδ
    summary.md
    weights/<method>/fold<F>/<channel>.joblib
    weights/<method>_scalers/fold<F>.joblib
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor  # relies on stage sbatch to ensure xgboost is installed

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt")
SPLITS = REPO / "pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood"
OUT = REPO / "results" / "spec03_bmax"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "weights").mkdir(exist_ok=True)

CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
SEED = 42
SIZE_FULL = 509
K = 5


def nmae(yt, yp):
    d = float(np.mean(np.abs(yt - yt.mean())))
    return float(np.mean(np.abs(yp - yt))) / d if d > 0 else np.nan
def rmse(yt, yp): return float(np.sqrt(np.mean((yp - yt) ** 2)))
def r2(yt, yp):
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - float(np.sum((yp - yt) ** 2)) / ss if ss > 0 else np.nan
def slope(yt, yp):
    x = yt - yt.mean(); y = yp - yp.mean(); d = float(np.sum(x ** 2))
    return float(np.sum(x * y) / d) if d > 0 else np.nan
def evalall(yt, yp): return {"NMAE": nmae(yt, yp), "RMSE": rmse(yt, yp),
                              "R2": r2(yt, yp), "slope": slope(yt, yp)}


def fit_ridge(X, y):
    return Ridge(alpha=1.0).fit(X, y)


def fit_lasso(X, y):
    return LassoCV(cv=K, random_state=SEED, max_iter=10000).fit(X, y)


def fit_enet(X, y):
    return ElasticNetCV(cv=K, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        random_state=SEED, max_iter=10000).fit(X, y)


def fit_xgb(X, y):
    grid = {
        "n_estimators": [200, 400, 800],
        "max_depth":    [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "subsample":    [0.8],
        "colsample_bytree": [0.8],
    }
    base = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                        random_state=SEED, verbosity=0, n_jobs=1)
    return GridSearchCV(base, grid, cv=KFold(K, shuffle=True, random_state=SEED),
                        scoring="neg_mean_absolute_error", n_jobs=-1).fit(X, y)


METHODS = {"ridge": fit_ridge, "lasso": fit_lasso, "enet": fit_enet, "xgb": fit_xgb}
METHOD_COLORS = {"ridge": "#1f4e79", "lasso": "#2b8a89",
                 "enet": "#d6a13b", "xgb": "#c25a5a"}


def m3_cells():
    return [json.load(open(f)) for f in
            sorted((REPO / "m3" / "results").glob("fold*/member*.json"))]


def main():
    b = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    D = b["descriptors"].numpy()
    Y = b["labels"].numpy()
    r2i = {r: i for i, r in enumerate(b["reaction_ids"])}
    print(f"loaded {D.shape[0]} × {D.shape[1]}", flush=True)

    rows = []
    barrier_routes = []
    for f in range(5):
        fdir = SPLITS / f"fold{f}"
        te = np.array([r2i[r] for r in json.load(open(fdir / "test_rids.json"))
                       if r in r2i])
        tr = np.array([r2i[r] for r in json.load(open(fdir / f"size_{SIZE_FULL}.json"))
                       if r in r2i])
        sc = StandardScaler().fit(D[tr])
        Xtr = sc.transform(D[tr]); Xte = sc.transform(D[te])

        # Persist the scaler once per fold so predictions are byte-reproducible.
        joblib.dump(sc, OUT / "weights" / f"scaler_fold{f}.joblib")

        for method_name, fit_fn in METHODS.items():
            print(f"  fold{f} {method_name}", flush=True)
            mdir = OUT / "weights" / method_name / f"fold{f}"
            mdir.mkdir(parents=True, exist_ok=True)
            per = np.zeros_like(Y[te])
            for c in range(5):
                m = fit_fn(Xtr, Y[tr, c])
                per[:, c] = m.predict(Xte)
                joblib.dump(m, mdir / f"{CH[c]}.joblib")
                mm = evalall(Y[te, c], per[:, c])
                rows.append({"fold": f, "method": method_name, "channel": CH[c], **mm})
            bs = per.sum(axis=1)
            bt = Y[te].sum(axis=1)
            rows.append({"fold": f, "method": method_name, "channel": "barrier_sum",
                         **evalall(bt, bs)})
            # direct barrier prediction (single scalar target)
            m_b = fit_fn(Xtr, Y[tr].sum(axis=1))
            joblib.dump(m_b, mdir / "barrier_direct.joblib")
            direct = m_b.predict(Xte)
            rows.append({"fold": f, "method": method_name, "channel": "barrier_direct",
                         **evalall(bt, direct)})
            barrier_routes.append({"fold": f, "method": method_name,
                                    "sum_NMAE": nmae(bt, bs),
                                    "direct_NMAE": nmae(bt, direct)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_leaderboard.csv", index=False)
    pd.DataFrame(barrier_routes).to_csv(OUT / "barrier_routes.csv", index=False)

    # ==== Bars ====
    channels_bar = CH + ["barrier_sum", "barrier_direct"]
    x = np.arange(len(channels_bar))
    n_meth = len(METHODS)
    width = 0.8 / n_meth
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, name in enumerate(METHODS.keys()):
        color = METHOD_COLORS[name]
        means = [df[(df.method == name) & (df.channel == ch)].NMAE.mean() for ch in channels_bar]
        stds = [df[(df.method == name) & (df.channel == ch)].NMAE.std() for ch in channels_bar]
        ax.bar(x + (i - (n_meth - 1) / 2) * width, means, width, yerr=stds,
               label=name, color=color, capsize=2, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_ylabel("NMAE")
    ax.set_xticks(x); ax.set_xticklabels(channels_bar, rotation=15)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_bars.png", dpi=150)
    plt.close(fig)

    # ==== best classical vs M_bδ ====
    m3c = m3_cells()
    m3n = {ch: [] for ch in CH + ["barrier"]}
    for c in m3c:
        yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
        for i, ch in enumerate(CH):
            m3n[ch].append(nmae(yt[:, i], yp[:, i]))
        m3n["barrier"].append(nmae(yt.sum(axis=1), yp.sum(axis=1)))
    best = {}
    for ch in CH + ["barrier"]:
        key = "barrier_sum" if ch == "barrier" else ch
        s = df[df.channel == key].groupby("method").NMAE.mean()
        best[ch] = {"method": s.idxmin(), "NMAE_mean": float(s.min())}

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CH) + 1)
    m3m = [np.mean(m3n[c]) for c in CH + ["barrier"]]
    bm = [best[c]["NMAE_mean"] for c in CH + ["barrier"]]
    ax.bar(xs - 0.2, bm, 0.4, color="#1f4e79", label="best classical")
    ax.bar(xs + 0.2, m3m, 0.4, color="#c25a5a", label="M_bδ (m3)")
    for i, (b_, m_, ch_) in enumerate(zip(bm, m3m, CH + ["barrier"])):
        ax.text(i, max(b_, m_) + 0.02,
                f'{best[ch_]["method"]}\nΔ={(b_ - m_) * 100:+.1f}pp',
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(CH + ["barrier"])
    ax.set_ylabel("NMAE")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "best_vs_neural.png", dpi=150)
    plt.close(fig)

    # ==== summary.md ====
    lines = ["# SPEC_03 — classical b-only maximization (m3, 787-rxn cohort)",
             "",
             "4 methods on 24-d descriptors: **ridge, lasso, enet, xgb**.",
             "All tuned by 5-fold CV on each fold's train split.",
             "",
             "| channel | best method | best NMAE | M_bδ NMAE | Δ (pp) |",
             "|---|---|---|---|---|"]
    for ch in CH + ["barrier"]:
        b_ = best[ch]; m3_ = float(np.mean(m3n[ch]))
        lines.append(f"| {ch} | {b_['method']} | {b_['NMAE_mean']:.3f} "
                     f"| {m3_:.3f} | {(b_['NMAE_mean'] - m3_) * 100:+.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("SPEC_03 done")


if __name__ == "__main__":
    main()
