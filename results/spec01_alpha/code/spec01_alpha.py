"""SPEC_01 — Ridge α optimization for the m3 physics baseline (24-d).

Per SPEC_01_ridge_alpha_optimization.md.

Outputs (results/spec01_alpha/):
  alpha_curves.png       6 panels (5 channels + barrier), NMAE vs α + GCV
  ridge_trace.png        coefficient paths per channel
  alpha_selection.csv    per-fold Test metrics at α ∈ {≈0, 1, α*_A}
  summary.md             α*_A / α*_B per channel + cond(XᵀX) + rank
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt")
SPLITS = REPO / "pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood"
OUT = REPO / "results" / "spec01_alpha"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "V_elst", "oi", "disp"]
NAVY = "#1f4e79"


def zscore(Xt, X):
    mu = Xt.mean(axis=0); sd = Xt.std(axis=0); sd[sd < 1e-8] = 1.0
    return mu, sd, (X - mu) / sd

def add_int(X): return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

def ridge_fit(X, y, alpha):
    D = X.shape[1]; reg = np.eye(D) * alpha; reg[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)

def gcv(X, y, alpha):
    D = X.shape[1]; reg = np.eye(D) * alpha; reg[-1, -1] = 0.0
    H = X @ np.linalg.solve(X.T @ X + reg, X.T)
    n = X.shape[0]; resid = y - H @ y
    num = float(np.mean(resid ** 2))
    den = (1.0 - float(np.trace(H) / n)) ** 2
    return num / den if den > 0 else np.nan

def nmae(yt, yp):
    denom = float(np.mean(np.abs(yt - yt.mean())))
    return float(np.mean(np.abs(yp - yt))) / denom if denom > 0 else np.nan

def rmse(yt, yp): return float(np.sqrt(np.mean((yp - yt) ** 2)))
def r2(yt, yp):
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - float(np.sum((yp - yt) ** 2)) / ss_tot if ss_tot > 0 else np.nan
def slope(yt, yp):
    x = yt - yt.mean(); y = yp - yp.mean(); d = float(np.sum(x ** 2))
    return float(np.sum(x * y) / d) if d > 0 else np.nan

def kfold_cv(X_tr, y_tr, alphas, K=5, seed=42):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y_tr))
    fold_idx = np.array_split(idx, K)
    scores = np.zeros((K, len(alphas)))
    for k in range(K):
        va = fold_idx[k]; tr = np.concatenate([f for i, f in enumerate(fold_idx) if i != k])
        mu, sd, Xtr = zscore(X_tr[tr], X_tr[tr]); Xva = (X_tr[va] - mu) / sd
        Xtri = add_int(Xtr); Xvai = add_int(Xva)
        for i, a in enumerate(alphas):
            W = ridge_fit(Xtri, y_tr[tr], a); scores[k, i] = nmae(y_tr[va], Xvai @ W)
    return scores.mean(axis=0)


def main():
    b = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    rids = b["reaction_ids"]; D = b["descriptors"].numpy(); Y = b["labels"].numpy()
    print(f"loaded {len(rids)} × {D.shape[1]}"); assert D.shape[1] == 24
    r2i = {r: i for i, r in enumerate(rids)}

    alphas = np.logspace(-6, 4, 61)
    all_ch = CHANNELS + ["barrier"]
    curves_cv = {c: [] for c in all_ch}
    curves_gcv = {c: [] for c in all_ch}
    cond_ls, rank_ls, metrics = [], [], []

    for f in range(5):
        fdir = SPLITS / f"fold{f}"
        te = np.array([r2i[r] for r in json.load(open(fdir / "test_rids.json")) if r in r2i])
        train_rids = json.load(open(fdir / "size_509.json"))
        tr = np.array([r2i[r] for r in train_rids if r in r2i])
        mu, sd, Xtr = zscore(D[tr], D[tr]); Xte = (D[te] - mu) / sd
        Xtri = add_int(Xtr); Xtei = add_int(Xte)
        cond_ls.append(float(np.linalg.cond(Xtri.T @ Xtri)))
        rank_ls.append(int(np.linalg.matrix_rank(Xtri)))

        for cidx, c in enumerate(CHANNELS):
            y_tr = Y[tr, cidx]; y_te = Y[te, cidx]
            curves_cv[c].append(kfold_cv(D[tr], y_tr, alphas))
            curves_gcv[c].append(np.array([gcv(Xtri, y_tr, a) for a in alphas]))
            for lbl, a in [("alpha_0", 1e-6), ("alpha_1", 1.0)]:
                W = ridge_fit(Xtri, y_tr, a); yp = Xtei @ W
                metrics.append({"fold": f, "channel": c, "alpha_label": lbl, "alpha": a,
                                "NMAE": nmae(y_te, yp), "RMSE": rmse(y_te, yp),
                                "R2": r2(y_te, yp), "slope": slope(y_te, yp)})
            a_star = float(alphas[np.argmin(curves_cv[c][-1])])
            W = ridge_fit(Xtri, y_tr, a_star); yp = Xtei @ W
            metrics.append({"fold": f, "channel": c, "alpha_label": "alpha_A", "alpha": a_star,
                            "NMAE": nmae(y_te, yp), "RMSE": rmse(y_te, yp),
                            "R2": r2(y_te, yp), "slope": slope(y_te, yp)})

        # barrier
        y_tr_b = Y[tr].sum(axis=1); y_te_b = Y[te].sum(axis=1)
        curves_cv["barrier"].append(kfold_cv(D[tr], y_tr_b, alphas))
        curves_gcv["barrier"].append(np.array([gcv(Xtri, y_tr_b, a) for a in alphas]))
        for lbl, a in [("alpha_0", 1e-6), ("alpha_1", 1.0)]:
            W = ridge_fit(Xtri, y_tr_b, a); yp = Xtei @ W
            metrics.append({"fold": f, "channel": "barrier", "alpha_label": lbl, "alpha": a,
                            "NMAE": nmae(y_te_b, yp), "RMSE": rmse(y_te_b, yp),
                            "R2": r2(y_te_b, yp), "slope": slope(y_te_b, yp)})
        a_star = float(alphas[np.argmin(curves_cv["barrier"][-1])])
        W = ridge_fit(Xtri, y_tr_b, a_star); yp = Xtei @ W
        metrics.append({"fold": f, "channel": "barrier", "alpha_label": "alpha_A", "alpha": a_star,
                        "NMAE": nmae(y_te_b, yp), "RMSE": rmse(y_te_b, yp),
                        "R2": r2(y_te_b, yp), "slope": slope(y_te_b, yp)})

    pd.DataFrame(metrics).to_csv(OUT / "alpha_selection.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    for k, c in enumerate(all_ch):
        ax = axes[k // 3, k % 3]
        cvm = np.mean(curves_cv[c], axis=0); gcvm = np.mean(curves_gcv[c], axis=0)
        gcvn = gcvm / np.nanmin(gcvm) * np.nanmin(cvm)
        a_A = alphas[np.argmin(cvm)]; a_B = alphas[np.nanargmin(gcvm)]
        ax.semilogx(alphas, cvm, color=NAVY, lw=1.6, label="k-fold CV")
        ax.semilogx(alphas, gcvn, color="#c25a5a", lw=1.2, ls="--", label="GCV (rescaled)")
        ax.axvline(1.0, color="#999", lw=0.7, ls=":")
        ax.axvline(a_A, color=NAVY, lw=0.7); ax.axvline(a_B, color="#c25a5a", lw=0.7)
        ax.set_title(f"{c}   α*_A={a_A:.1e}  α*_B={a_B:.1e}", fontsize=10)
        ax.set_xlabel("α"); ax.set_ylabel("NMAE"); ax.grid(alpha=0.3)
        if k == 0: ax.legend(fontsize=8)
    fig.savefig(OUT / "alpha_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    tr0 = np.array([r2i[r] for r in json.load(open(SPLITS / "fold0" / "size_509.json")) if r in r2i])
    mu, sd, Xtr = zscore(D[tr0], D[tr0]); Xtri = add_int(Xtr)
    for k, c in enumerate(all_ch):
        ax = axes[k // 3, k % 3]
        yv = Y[tr0].sum(axis=1) if c == "barrier" else Y[tr0, CHANNELS.index(c)]
        coefs = np.array([ridge_fit(Xtri, yv, a) for a in alphas])
        for j in range(coefs.shape[1] - 1):
            ax.semilogx(alphas, coefs[:, j], lw=0.6, alpha=0.7)
        ax.set_title(c, fontsize=10); ax.set_xlabel("α"); ax.set_ylabel("β̂"); ax.grid(alpha=0.3)
    fig.savefig(OUT / "ridge_trace.png", dpi=150); plt.close(fig)

    df = pd.DataFrame(metrics)
    sm = df.groupby(["channel", "alpha_label"])["NMAE"].agg(["mean", "std"]).reset_index()
    lines = ["# SPEC_01 — ridge α optimization for m3 (787-rxn cohort)",
             "", f"- descriptor width: **24** (asserted)",
             f"- cond(XᵀX) mean: {np.mean(cond_ls):.3e}",
             f"- rank(X) per fold: {rank_ls}", "",
             "## α*_A (CV) / α*_B (GCV) per channel", ""]
    for c in all_ch:
        cvm = np.mean(curves_cv[c], axis=0); gcvm = np.mean(curves_gcv[c], axis=0)
        lines.append(f"- **{c}**: α*_A = {float(alphas[np.argmin(cvm)]):.3e}, "
                     f"α*_B (GCV) = {float(alphas[np.nanargmin(gcvm)]):.3e}")
    lines += ["", "## Test NMAE at α ∈ {≈0, 1, α*_A} (mean ± std over 5 folds)",
              "", "| channel | α ≈ 0 | α = 1 | α = α* |", "|---|---|---|---|"]
    for c in all_ch:
        r0 = sm[(sm.channel == c) & (sm.alpha_label == "alpha_0")].iloc[0]
        r1 = sm[(sm.channel == c) & (sm.alpha_label == "alpha_1")].iloc[0]
        rA = sm[(sm.channel == c) & (sm.alpha_label == "alpha_A")].iloc[0]
        lines.append(f"| {c} | {r0['mean']:.3f} ± {r0['std']:.3f} "
                     f"| {r1['mean']:.3f} ± {r1['std']:.3f} "
                     f"| {rA['mean']:.3f} ± {rA['std']:.3f} |")
    lines += ["", "## Notes",
              "- Intercept NOT penalised.",
              "- σ_c-normalised loss + system-level α tuning is out of scope (SPEC_02)."]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("SPEC_01 done")

if __name__ == "__main__":
    main()
