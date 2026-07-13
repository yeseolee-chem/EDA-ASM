"""SPEC_04 (XGB-based descriptor contribution) - m3 v9 783 rxn cohort.

Replaces the old OLS/VIF/lasso-path analysis with an XGB-native workflow:
  1. Per-channel + barrier XGB with fixed HPs (matches SPEC_03/SPEC_06 config).
  2. Feature-importance heatmap (gain) across (channel + barrier) x descriptor.
  3. Forward-selection saturation curves using XGB CV NMAE per channel.
  4. Reduced-set proposal (union of per-channel elbows) with full-vs-reduced test NMAE delta.
  5. Cross-descriptor VIF (numpy-only) for collinearity flags.

Outputs to spec/spec04_descriptors/{results, figures}.
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
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

REPO       = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path(os.environ.get("BUNDLE_PT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
SPLIT_ROOT = Path(os.environ.get("SPLIT_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9"))
D25_PQ     = REPO / "spec/spec05_d25_sum/data/descriptors_d25_refR.parquet"
D26_28_PQ  = REPO / "spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet"
OUT_RES    = REPO / "spec/spec04_descriptors/results"
OUT_FIG    = REPO / "spec/spec04_descriptors/figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
NAVY = "#1f4e79"
SEED = 42

XGB_HP = dict(
    n_estimators=800, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    objective="reg:squarederror", verbosity=0, n_jobs=4,
)


def make_xgb(seed):
    return XGBRegressor(random_state=seed, **XGB_HP)


def nmae(yt, yp):
    mad = np.mean(np.abs(yt - yt.mean()))
    return float(np.mean(np.abs(yt - yp)) / (mad + 1e-12))


def cv_nmae(X_tr, y_tr, X_te, y_te, seed):
    m = make_xgb(seed).fit(X_tr, y_tr)
    return nmae(y_te, m.predict(X_te))


def attach_col(rids, parquet_path, col):
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


def load_data():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = b["reaction_ids"]
    X24 = b["descriptors"].numpy()
    Y = b["labels"].numpy()
    # Attach the four SPEC_05 / SPEC_06 proxies to form the 28-d matrix
    d25, ok25 = attach_col(rids, D25_PQ, "d25")
    d26, ok26 = attach_col(rids, D26_28_PQ, "d26")
    d27, ok27 = attach_col(rids, D26_28_PQ, "d27")
    d28, ok28 = attach_col(rids, D26_28_PQ, "d28")
    X = np.hstack([X24, d25[:, None], d26[:, None], d27[:, None], d28[:, None]])
    ok_all = ok25 & ok26 & ok27 & ok28
    print(f"proxies scf_ok:  d25={ok25.sum()}  d26={ok26.sum()}  d27={ok27.sum()}  "
          f"d28={ok28.sum()}  intersection={ok_all.sum()}", flush=True)
    r2i = {r: i for i, r in enumerate(rids)}
    folds = []
    for i in range(5):
        fd = SPLIT_ROOT / f"fold{i}"
        te = json.load(open(fd / "test_rids.json"))
        tf = sorted(fd.glob("size_*.json"),
                    key=lambda p: int(p.stem.split("_")[1]), reverse=True)[0]
        tr = json.load(open(tf))
        # Restrict train/test to rxns with all four proxies available (fair 28-d comparison)
        tr_idx = np.array([r2i[r] for r in tr if r in r2i and ok_all[r2i[r]]])
        te_idx = np.array([r2i[r] for r in te if r in r2i and ok_all[r2i[r]]])
        folds.append((tr_idx, te_idx))
    return rids, X, Y, folds


def main():
    rids, X, Y, folds = load_data()
    n, D = X.shape
    # 24-d m3 + d25 (SPEC_05, strain proxy) + d26/d27/d28 (SPEC_06 elst/Pauli/oi)
    names = [f"d{k+1}" for k in range(24)] + ["d25", "d26", "d27", "d28"]
    assert D == 28, f"expected 28-d matrix (24-d m3 + d25 + d26 + d27 + d28), got D={D}"
    print(f"[m3 v9 + proxies] N={n}  D={D}")

    Y_all = {c: Y[:, i] for i, c in enumerate(CHANNELS)}
    Y_all["barrier"] = Y.sum(axis=1)
    channels_plot = CHANNELS + ["barrier"]

    # ============ VIF (numpy-only, no statsmodels) ============
    # Restrict VIF to rxns with valid proxies (union of test/train intersection)
    ok_all = np.zeros(n, dtype=bool)
    for tr, te in folds:
        ok_all[tr] = True; ok_all[te] = True
    X_valid = X[ok_all]
    mu = X_valid.mean(0); sig = X_valid.std(0); sig = np.where(sig < 1e-9, 1.0, sig)
    Xz = (X_valid - mu) / sig
    vif_rows = []
    for k in range(D):
        others = np.delete(Xz, k, axis=1)
        target = Xz[:, k]
        # OLS: beta = (X'X)^-1 X'y (no intercept because z-scored)
        try:
            beta, *_ = np.linalg.lstsq(others, target, rcond=None)
            resid = target - others @ beta
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum(target ** 2))
            r2k = 1 - ss_res / max(ss_tot, 1e-12)
            vif = 1.0 / max(1 - r2k, 1e-12)
        except Exception:
            r2k = float("nan"); vif = float("nan")
        vif_rows.append({"descriptor": names[k], "R2_on_others": r2k,
                         "VIF": vif,
                         "flag": "severe" if vif > 10 else ("moderate" if vif > 5 else "ok")})
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(OUT_RES / "vif.csv", index=False)
    cond_XtX = float(np.linalg.cond(Xz.T @ Xz))
    rank_X = int(np.linalg.matrix_rank(Xz))
    print(f"cond(X.T@X)={cond_XtX:.3e}  rank={rank_X}/{D}")

    # ============ XGB feature importance heatmap per channel ============
    imp_by_ch = {}
    for ch, y in Y_all.items():
        m = make_xgb(SEED).fit(X[ok_all], y[ok_all])
        imp_by_ch[ch] = m.feature_importances_.copy()
    imp_arr = np.stack([imp_by_ch[ch] for ch in channels_plot], axis=0)
    pd.DataFrame(imp_arr, index=channels_plot, columns=names).to_csv(
        OUT_RES / "importance_heatmap.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(imp_arr, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(channels_plot))); ax.set_yticklabels(channels_plot)
    ax.set_xticks(range(D)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "importance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ Forward-selection saturation curves ============
    # Use fold-0 train -> 3-fold inner CV to score each candidate.
    train_idx0 = folds[0][0]  # already filtered to ok_all
    Xt = X[train_idx0]
    fs_order = {}
    fs_curves = {}
    for ch in channels_plot:
        y_tr = Y_all[ch][train_idx0]
        print(f"  running FS for {ch} (D={D})", flush=True)
        selected = []; remaining = list(range(D))
        curve = []
        while remaining:
            best_next = None; best_score = None
            for k in remaining:
                cols = selected + [k]
                # 3-fold CV
                inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
                scores = []
                for i_tr, i_va in inner.split(Xt):
                    scores.append(cv_nmae(Xt[i_tr][:, cols], y_tr[i_tr],
                                          Xt[i_va][:, cols], y_tr[i_va],
                                          seed=SEED + len(selected)))
                s = float(np.mean(scores))
                if best_score is None or s < best_score:
                    best_score = s; best_next = k
            selected.append(best_next); remaining.remove(best_next)
            curve.append(best_score)
        fs_order[ch] = [names[k] for k in selected]
        fs_curves[ch] = curve
        print(f"  FS {ch}: top-5 {fs_order[ch][:5]}", flush=True)

    with open(OUT_RES / "forward_selection_order.json", "w") as f:
        json.dump(fs_order, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    core_by_ch = {}
    for ax, ch in zip(axes.flat, channels_plot):
        c = np.array(fs_curves[ch])
        ax.plot(range(1, D + 1), c, "-o", color=NAVY, ms=3)
        margins = np.abs(np.diff(c))
        elbow = D
        for i, dm in enumerate(margins):
            if dm < 0.005:
                elbow = i + 1
                break
        ax.axvline(elbow, color="green", ls="--", lw=0.8, label=f"elbow={elbow}")
        ax.axvspan(elbow, D, color="gray", alpha=0.2, label="removable")
        ax.set_title(ch)
        ax.set_xlabel("# descriptors added (XGB CV)")
        ax.set_ylabel("CV NMAE")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        core_by_ch[ch] = fs_order[ch][:elbow]
    fig.suptitle("SPEC_04 - XGB forward-selection saturation (m3 v9, fold-0 train)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "saturation_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ Reduced set proposal ============
    core = sorted(set().union(*core_by_ch.values()))
    core_idx = [names.index(c) for c in core]
    print(f"core set: {core} ({len(core)}/{D})", flush=True)

    reduced_rows = []
    for ch in channels_plot:
        y = Y_all[ch]
        full_nmae, red_nmae = [], []
        for tr, te in folds:
            m_full = make_xgb(SEED).fit(X[tr], y[tr])
            full_nmae.append(nmae(y[te], m_full.predict(X[te])))
            m_red = make_xgb(SEED).fit(X[tr][:, core_idx], y[tr])
            red_nmae.append(nmae(y[te], m_red.predict(X[te][:, core_idx])))
        reduced_rows.append({
            "channel": ch,
            "full_NMAE": float(np.mean(full_nmae)),
            "reduced_NMAE": float(np.mean(red_nmae)),
            "delta": float(np.mean(red_nmae) - np.mean(full_nmae)),
        })
    pd.DataFrame(reduced_rows).to_csv(OUT_RES / "reduced_set_delta.csv", index=False)
    (OUT_RES / "reduced_set_proposal.md").write_text(
        f"# Reduced set proposal (XGB forward-selection elbow union)\n\n"
        f"Core ({len(core)} descriptors): {core}\n\n"
        f"Full m3 has D=24; reduced set drops {D - len(core)} descriptors.\n"
        f"See reduced_set_delta.csv for NMAE(full) vs NMAE(core) per channel.\n"
    )

    # ============ Cross-descriptor importance ranking summary ============
    # Union top-k across all channels
    top_k = 8
    universal = set()
    for ch in channels_plot:
        top_k_names = fs_order[ch][:top_k]
        universal.update(top_k_names)
    universal_sorted = sorted(universal, key=lambda n: int(n[1:]))

    # ============ summary.md ============
    lines = ["# SPEC_04 (XGB-based descriptor contribution) - summary", "",
             f"- Cohort: {n} rxns (m3 v9 783), D = {D}",
             f"- Method: per-channel XGB (fixed HP matching SPEC_03/SPEC_06),",
             "  forward-selection with 3-fold inner CV NMAE, gain-based importance heatmap.",
             f"- Collinearity: cond(X.T@X) = {cond_XtX:.2e}, rank = {rank_X}/{D}",
             "",
             "## Top-5 forward-selection order per channel", "",
             "| channel | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
    for ch in channels_plot:
        lines.append(f"| {ch} | " + " | ".join(fs_order[ch][:5]) + " |")
    lines += ["",
              f"## Union of top-{top_k} across channels ({len(universal_sorted)} descriptors)",
              f"- {universal_sorted}",
              "",
              "## VIF flags (severe = VIF>10, moderate = VIF>5)",
              "See vif.csv.",
              "",
              "## Reduced set (union of per-channel elbows)",
              f"- Core ({len(core)}): {core}",
              "- See reduced_set_delta.csv for NMAE (full) vs NMAE (reduced) per channel."]
    (OUT_RES / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT_RES / 'summary.md'}")


if __name__ == "__main__":
    main()
