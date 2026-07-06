"""SPEC_03 — maximizing the physics-only baseline for m3 (24-d)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt")
SPLITS = REPO / "pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood"
OUT = REPO / "results" / "spec03_bmax"; OUT.mkdir(parents=True, exist_ok=True)

CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
SEED = 42; SIZE_FULL = 509; K = 5


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
def evalall(yt, yp): return {"NMAE": nmae(yt, yp), "RMSE": rmse(yt, yp), "R2": r2(yt, yp), "slope": slope(yt, yp)}


def fit_ridge(X, y): return Ridge(alpha=1.0).fit(X, y)
def fit_lasso(X, y): return LassoCV(cv=K, random_state=SEED, max_iter=10000).fit(X, y)
def fit_enet(X, y): return ElasticNetCV(cv=K, l1_ratio=[.1,.3,.5,.7,.9], random_state=SEED, max_iter=10000).fit(X, y)
def fit_krr(X, y):
    return GridSearchCV(KernelRidge(kernel="rbf"),
                        {"alpha": np.logspace(-3, 2, 6), "gamma": np.logspace(-3, 1, 5)},
                        cv=KFold(K, shuffle=True, random_state=SEED),
                        scoring="neg_mean_absolute_error", n_jobs=-1).fit(X, y)
def fit_gbm(X, y):
    return GridSearchCV(GradientBoostingRegressor(random_state=SEED),
                        {"n_estimators": [100, 300, 600], "max_depth": [2, 3, 5],
                         "learning_rate": [0.03, 0.1], "subsample": [0.8]},
                        cv=KFold(K, shuffle=True, random_state=SEED),
                        scoring="neg_mean_absolute_error", n_jobs=-1).fit(X, y)
def fit_rf(X, y):
    return GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
                        {"n_estimators": [200, 500], "max_depth": [None, 8, 16]},
                        cv=KFold(K, shuffle=True, random_state=SEED),
                        scoring="neg_mean_absolute_error", n_jobs=-1).fit(X, y)

METHODS = {"ridge": fit_ridge, "lasso": fit_lasso, "enet": fit_enet,
           "krr": fit_krr, "gbm": fit_gbm, "rf": fit_rf}


def m3_cells():
    return [json.load(open(f)) for f in sorted((REPO / "m3" / "results").glob("fold*/member*.json"))]


def main():
    b = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    D = b["descriptors"].numpy(); Y = b["labels"].numpy()
    r2i = {r: i for i, r in enumerate(b["reaction_ids"])}
    print(f"loaded {D.shape[0]}×{D.shape[1]}", flush=True)

    rows, barrier_routes = [], []
    for f in range(5):
        fdir = SPLITS / f"fold{f}"
        te = np.array([r2i[r] for r in json.load(open(fdir / "test_rids.json")) if r in r2i])
        tr = np.array([r2i[r] for r in json.load(open(fdir / f"size_{SIZE_FULL}.json")) if r in r2i])
        sc = StandardScaler().fit(D[tr])
        Xtr = sc.transform(D[tr]); Xte = sc.transform(D[te])
        for name, fn in METHODS.items():
            print(f"  fold{f} {name}", flush=True)
            per = np.zeros_like(Y[te])
            for c in range(5):
                m = fn(Xtr, Y[tr, c]); per[:, c] = m.predict(Xte)
                mm = evalall(Y[te, c], per[:, c])
                rows.append({"fold": f, "method": name, "channel": CH[c], **mm})
            bs = per.sum(axis=1); bt = Y[te].sum(axis=1)
            rows.append({"fold": f, "method": name, "channel": "barrier_sum", **evalall(bt, bs)})
            m = fn(Xtr, Y[tr].sum(axis=1)); direct = m.predict(Xte)
            rows.append({"fold": f, "method": name, "channel": "barrier_direct", **evalall(bt, direct)})
            barrier_routes.append({"fold": f, "method": name,
                                    "sum_NMAE": nmae(bt, bs), "direct_NMAE": nmae(bt, direct)})

    df = pd.DataFrame(rows); df.to_csv(OUT / "baseline_leaderboard.csv", index=False)
    pd.DataFrame(barrier_routes).to_csv(OUT / "barrier_routes.csv", index=False)

    channels_bar = CH + ["barrier_sum", "barrier_direct"]
    x = np.arange(len(channels_bar)); width = 0.8 / len(METHODS)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#1f4e79", "#2b8a89", "#c25a5a", "#8ca752", "#d6a13b", "#7d5ca0"]
    for i, (name, color) in enumerate(zip(METHODS.keys(), colors)):
        means = [df[(df.method == name) & (df.channel == ch)].NMAE.mean() for ch in channels_bar]
        stds = [df[(df.method == name) & (df.channel == ch)].NMAE.std() for ch in channels_bar]
        ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, means, width, yerr=stds,
               label=name, color=color, capsize=2, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_ylabel("NMAE"); ax.set_xticks(x); ax.set_xticklabels(channels_bar, rotation=15)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=9); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "baseline_bars.png", dpi=150); plt.close(fig)

    # Best classical vs M_bδ
    m3c = m3_cells()
    m3n = {ch: [] for ch in CH + ["barrier"]}
    for c in m3c:
        yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
        for i, ch in enumerate(CH): m3n[ch].append(nmae(yt[:, i], yp[:, i]))
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
        ax.text(i, max(b_, m_) + 0.02, f'{best[ch_]["method"]}\nΔ={(b_ - m_) * 100:+.1f}pp',
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(CH + ["barrier"]); ax.set_ylabel("NMAE")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "best_vs_neural.png", dpi=150); plt.close(fig)

    lines = ["# SPEC_03 — classical b-only maximization (m3, 787-rxn cohort)",
             "", "Per-method 5-fold-CV-tuned classical models on 24-d descriptors.", "",
             "| channel | best method | best NMAE | M_bδ NMAE | Δ (pp) |",
             "|---|---|---|---|---|"]
    for ch in CH + ["barrier"]:
        b_ = best[ch]; m3_ = float(np.mean(m3n[ch]))
        lines.append(f"| {ch} | {b_['method']} | {b_['NMAE_mean']:.3f} | {m3_:.3f} "
                     f"| {(b_['NMAE_mean'] - m3_) * 100:+.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("SPEC_03 done")


if __name__ == "__main__":
    main()
