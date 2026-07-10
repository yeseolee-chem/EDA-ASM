"""SPEC_01 - Ridge alpha optimization for the m3 24-d physics baseline.

Uses v7 bundle (776 rxns) and its 5-fold splits. For each of 5 channels + barrier:
  - grid alphas = np.logspace(-6, 4, 61)
  - Rule A: 5-fold CV within train (mean val NMAE)
  - Rule B: GCV (Golub-Heath-Wahba)
  - alpha=0 admissibility via cond(X.T@X) and rank(X)
  - Ridge trace + effective dof plot
  - Compare test NMAE at alpha in {0, 1, alpha*}

Outputs go to spec/spec01_alpha/{results, figures}.
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

REPO       = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT  = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v7/features_v7_delta_m3.pt")
SPLIT_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v7/trackB_no_ood")
OUT_RES    = REPO / "spec/spec01_alpha/results"
OUT_FIG    = REPO / "spec/spec01_alpha/figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
NAVY = "#1f4e79"


def zscore_fit(X):
    mu = X.mean(0); sig = X.std(0)
    sig = np.where(sig < 1e-9, 1.0, sig)
    return mu, sig


def add_intercept(X):
    return np.hstack([X, np.ones((len(X), 1))])


def ridge_solve(X_train, y_train, alpha):
    """Return (W, mu, sig) with unpenalized intercept."""
    mu, sig = zscore_fit(X_train)
    Xn = (X_train - mu) / sig
    Xa = add_intercept(Xn)
    D = Xa.shape[1]
    reg = alpha * np.eye(D); reg[-1, -1] = 0.0
    W = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y_train)
    return W, mu, sig


def ridge_pred(X, W, mu, sig):
    Xn = (X - mu) / sig
    return add_intercept(Xn) @ W


def nmae(y_true, y_pred):
    mad = np.mean(np.abs(y_true - y_true.mean()))
    return float(np.mean(np.abs(y_true - y_pred)) / (mad + 1e-12))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def slope(y_true, y_pred):
    a = y_true - y_true.mean(); b = y_pred - y_pred.mean()
    d = float(np.sum(a * a))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def gcv_score(X_train, y_train, alpha):
    """Golub-Heath-Wahba GCV. y_train shape (n,) OR (n, C) - averaged across C."""
    mu, sig = zscore_fit(X_train)
    Xn = (X_train - mu) / sig
    Xa = add_intercept(Xn); n, D = Xa.shape
    reg = alpha * np.eye(D); reg[-1, -1] = 0.0
    M = np.linalg.inv(Xa.T @ Xa + reg)
    H = Xa @ M @ Xa.T
    trace_I_H = np.trace(np.eye(n) - H)
    y_hat = H @ y_train
    res = y_train - y_hat
    if res.ndim == 1:
        num = np.mean(res ** 2)
    else:
        num = np.mean(res ** 2)  # avg over both axes
    return float(num / (trace_I_H / n) ** 2)


def cv_score(X_train, y_train, alpha, n_splits=5, seed=42):
    """5-fold CV within train, returns mean val NMAE."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_train)); rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    scores = []
    for k in range(n_splits):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        W, mu, sig = ridge_solve(X_train[train_idx], y_train[train_idx], alpha)
        yp = ridge_pred(X_train[val_idx], W, mu, sig)
        scores.append(nmae(y_train[val_idx], yp))
    return float(np.mean(scores))


def load_data():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = b["reaction_ids"]
    X = b["descriptors"].numpy()
    Y = b["labels"].numpy()  # order: strain, Pauli, Velst, orb, disp -> match CHANNELS
    rid_to_idx = {r: i for i, r in enumerate(rids)}
    folds = []
    for i in range(5):
        fd = SPLIT_ROOT / f"fold{i}"
        test_rids = json.load(open(fd / "test_rids.json"))
        train_files = sorted(fd.glob("size_*.json"),
                             key=lambda p: int(p.stem.split("_")[1]), reverse=True)
        train_rids = json.load(open(train_files[0]))
        train_idx = np.array([rid_to_idx[r] for r in train_rids if r in rid_to_idx])
        test_idx  = np.array([rid_to_idx[r] for r in test_rids  if r in rid_to_idx])
        folds.append((train_idx, test_idx))
    return X, Y, folds, rids


def main():
    X, Y, folds, rids = load_data()
    n, D = X.shape
    print(f"[m3] N={n}, D={D}, channels={len(CHANNELS)} (+ barrier)")
    assert D == 24, f"expected D=24 per m3 spec, got {D}"

    # Global rank/cond check on standardized X with intercept
    mu, sig = zscore_fit(X)
    Xn = (X - mu) / sig
    Xa = add_intercept(Xn)
    cond_XtX = float(np.linalg.cond(Xa.T @ Xa))
    rank_X = int(np.linalg.matrix_rank(Xa))
    print(f"cond(X.T@X) = {cond_XtX:.3e}   rank(X) = {rank_X}/{Xa.shape[1]}")
    alpha_zero_admissible = (cond_XtX < 1e6) and (rank_X == Xa.shape[1])

    alphas = np.logspace(-6, 4, 61)
    Y_targets = {c: Y[:, i] for i, c in enumerate(CHANNELS)}
    Y_targets["barrier"] = Y.sum(axis=1)

    # ============================ Task 3.1 grid search =========================
    curves = {}  # (ch): {"cv": [...], "gcv": [...]}
    for ch, y in Y_targets.items():
        cv_curve = np.zeros(len(alphas))
        gcv_curve = np.zeros(len(alphas))
        # CV/GCV averaged across folds' training splits
        for a_i, a in enumerate(alphas):
            cvs, gcvs = [], []
            for train_idx, _ in folds:
                cvs.append(cv_score(X[train_idx], y[train_idx], a))
                gcvs.append(gcv_score(X[train_idx], y[train_idx], a))
            cv_curve[a_i] = np.mean(cvs)
            gcv_curve[a_i] = np.mean(gcvs)
        curves[ch] = {"cv": cv_curve, "gcv": gcv_curve}
        print(f"  {ch}: cv min at alpha={alphas[cv_curve.argmin()]:.3e} NMAE={cv_curve.min():.3f}; "
              f"gcv min at alpha={alphas[gcv_curve.argmin()]:.3e}")

    # ============================ Task 3.4 test compare ========================
    def eval_alpha(alpha, y_all):
        """Test NMAE/RMSE/R2/slope averaged across 5 folds' test sets."""
        m = {"NMAE": [], "RMSE": [], "R2": [], "slope": []}
        for train_idx, test_idx in folds:
            W, mu_f, sig_f = ridge_solve(X[train_idx], y_all[train_idx], alpha)
            yp = ridge_pred(X[test_idx], W, mu_f, sig_f)
            yt = y_all[test_idx]
            m["NMAE"].append(nmae(yt, yp))
            m["RMSE"].append(rmse(yt, yp))
            m["R2"].append(r2(yt, yp))
            m["slope"].append(slope(yt, yp))
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in m.items()}

    rows = []
    for ch, y in Y_targets.items():
        a_star_cv = float(alphas[curves[ch]["cv"].argmin()])
        a_star_gcv = float(alphas[curves[ch]["gcv"].argmin()])
        tested = {"alpha1": 1.0, "alpha_star_cv": a_star_cv, "alpha_star_gcv": a_star_gcv}
        if alpha_zero_admissible:
            tested["alpha0"] = 0.0
        for tag, a in tested.items():
            m = eval_alpha(a, y)
            rows.append({
                "channel": ch, "rule": tag, "alpha": a,
                "NMAE_mean": m["NMAE"][0], "NMAE_std": m["NMAE"][1],
                "RMSE_mean": m["RMSE"][0], "RMSE_std": m["RMSE"][1],
                "R2_mean": m["R2"][0], "R2_std": m["R2"][1],
                "slope_mean": m["slope"][0], "slope_std": m["slope"][1],
                "cond_XtX": cond_XtX, "rank_X": rank_X,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "alpha_selection.csv", index=False)
    print(df.to_string())

    # ============================ Fig 1: alpha curves ==========================
    channels_plot = CHANNELS + ["barrier"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, ch in zip(axes.flat, channels_plot):
        cv_c = curves[ch]["cv"]; gcv_c = curves[ch]["gcv"]
        # Normalize GCV to match NMAE scale visually: rescale to [min, 1] mapping cv range
        gcv_norm = (gcv_c - gcv_c.min()) / (gcv_c.max() - gcv_c.min() + 1e-12)
        gcv_norm = gcv_norm * (cv_c.max() - cv_c.min()) + cv_c.min()
        ax.plot(alphas, cv_c, "-", color=NAVY, lw=1.6, label="5-fold CV NMAE")
        ax.plot(alphas, gcv_norm, "--", color="#c05e2b", lw=1.2, label="GCV (rescaled)")
        a_star = alphas[cv_c.argmin()]
        ax.axvline(a_star, color="green", lw=0.8, ls=":", label=f"a*_CV={a_star:.2e}")
        ax.axvline(1.0, color="gray", lw=0.6, ls=":", label="a=1")
        ax.set_xscale("log")
        ax.set_title(ch); ax.grid(alpha=0.3)
        ax.set_xlabel("alpha (log)")
        ax.set_ylabel("val NMAE")
        if ch == channels_plot[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("SPEC_01 - Ridge alpha curves (m3, 5-fold on 776 rxn v7)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "alpha_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============================ Fig 2: ridge trace + df(alpha) ==============
    # Use fold-0 train to build one ridge trace per channel
    train_idx0, _ = folds[0]
    Xt = X[train_idx0]
    mu_t, sig_t = zscore_fit(Xt); Xtn = (Xt - mu_t) / sig_t; Xta = add_intercept(Xtn)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    df_alpha = []
    for ax, ch in zip(axes.flat, channels_plot):
        y = Y_targets[ch][train_idx0]
        coefs = np.zeros((len(alphas), D))
        eff_df = np.zeros(len(alphas))
        for a_i, a in enumerate(alphas):
            reg = a * np.eye(Xta.shape[1]); reg[-1, -1] = 0.0
            M = np.linalg.inv(Xta.T @ Xta + reg)
            W = M @ Xta.T @ y
            coefs[a_i] = W[:-1]  # exclude intercept
            H = Xta @ M @ Xta.T
            eff_df[a_i] = np.trace(H)
        for k in range(D):
            ax.plot(alphas, coefs[:, k], lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(f"{ch} (df@a=1: {eff_df[np.abs(alphas-1).argmin()]:.1f})")
        ax.set_xlabel("alpha")
        ax.set_ylabel("standardized coef")
        ax.grid(alpha=0.3)
        df_alpha.append({"channel": ch, "df_at_a_1": float(eff_df[np.abs(alphas-1).argmin()]),
                         "df_at_a_star": float(eff_df[curves[ch]["cv"].argmin()])})
    fig.suptitle("SPEC_01 - Ridge trace (m3, fold-0 train)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "ridge_trace.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(df_alpha).to_csv(OUT_RES / "effective_dof.csv", index=False)

    # ============================ summary.md ==================================
    lines = ["# SPEC_01 summary", "",
             f"- N = {n}, D = {D} (m3 v7 776-rxn bundle)",
             f"- cond(X.T@X) = {cond_XtX:.3e}, rank = {rank_X}",
             f"- alpha=0 admissible: {alpha_zero_admissible}",
             "",
             "## Selected alpha per channel/barrier", "",
             "| channel | a*_CV | a*_GCV | NMAE@a=1 | NMAE@a* | delta (a=1 -> a*) |",
             "|---|---|---|---|---|---|"]
    for ch in channels_plot:
        sub = df[df.channel == ch]
        a1 = sub[sub.rule == "alpha1"].iloc[0]
        astar = sub[sub.rule == "alpha_star_cv"].iloc[0]
        d = a1.NMAE_mean - astar.NMAE_mean
        lines.append(f"| {ch} | {astar.alpha:.2e} | "
                     f"{sub[sub.rule=='alpha_star_gcv'].iloc[0].alpha:.2e} | "
                     f"{a1.NMAE_mean:.3f} | {astar.NMAE_mean:.3f} | {d:+.3f} |")
    lines += ["",
              "## Scope note (delta interaction)", "",
              "`b` is only the ridge baseline; the residual `delta = y - b` is what MLP head learns.",
              "The alpha that minimizes b-alone NMAE is NOT necessarily optimal for the b+delta system.",
              "System-level alpha selection belongs to SPEC_02."]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}")


if __name__ == "__main__":
    main()
