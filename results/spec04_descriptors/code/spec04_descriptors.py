"""SPEC_04 — descriptor contribution + parsimony (m3, 24-d)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from scipy import stats as sstats
from sklearn.linear_model import lars_path, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _add_const(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


class _OLSFit:
    def __init__(self, X, y):
        self.X = X; self.y = y
        Xi = _add_const(X)
        self.n, self.k = Xi.shape                    # k = intercept + p
        self.beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        resid = y - Xi @ self.beta
        dof = self.n - self.k
        s2 = float(np.sum(resid ** 2)) / dof if dof > 0 else np.nan
        XtXi = np.linalg.pinv(Xi.T @ Xi)
        self.se = np.sqrt(np.diag(XtXi) * s2)
        self.t = self.beta / np.where(self.se > 0, self.se, np.nan)
        self.pvalues = 2.0 * (1.0 - sstats.t.cdf(np.abs(self.t), df=dof))
        crit = sstats.t.ppf(0.975, df=dof)
        self.ci_low = self.beta - crit * self.se
        self.ci_high = self.beta + crit * self.se
        ybar = y.mean()
        ss_tot = float(np.sum((y - ybar) ** 2))
        ss_res = float(np.sum(resid ** 2))
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    @property
    def params(self):    return self.beta
    @property
    def bse(self):       return self.se
    @property
    def tvalues(self):   return self.t
    def conf_int(self):
        return np.column_stack([self.ci_low, self.ci_high])

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt")
SPLITS = REPO / "pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood"
OUT = REPO / "results" / "spec04_descriptors"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ols_tables").mkdir(exist_ok=True)

CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
NAMES = [f"d{i}" for i in range(1, 25)]
SIZE_FULL = 509


def dedup(D, names, thr=0.98):
    C = np.corrcoef(D.T); keep = list(range(D.shape[1])); dropped = []
    for j in range(1, D.shape[1]):
        for i in range(j):
            if i not in keep: continue
            if abs(C[i, j]) >= thr:
                if j in keep:
                    keep.remove(j); dropped.append((names[j], names[i], float(C[i, j])))
    return D[:, keep], [names[i] for i in keep], dropped


def vif(X):
    v = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        y = X[:, j]; oth = np.delete(X, j, axis=1)
        m = _OLSFit(oth, y)
        v[j] = 1.0 / max(1e-12, 1.0 - m.rsquared)
    return v


def nmae(yt, yp):
    d = float(np.mean(np.abs(yt - yt.mean())))
    return float(np.mean(np.abs(yp - yt))) / d if d > 0 else np.nan


def forward_sel(Xtr, ytr, Xva, yva, names):
    remaining = list(range(Xtr.shape[1])); chosen = []; curve = []
    while remaining:
        best_j = None; best = np.inf
        for j in remaining:
            cols = chosen + [j]
            beta, *_ = np.linalg.lstsq(_add_const(Xtr[:, cols]), ytr, rcond=None)
            yp = _add_const(Xva[:, cols]) @ beta
            n = nmae(yva, yp)
            if n < best: best = n; best_j = j
        chosen.append(best_j); remaining.remove(best_j)
        curve.append({"n": len(chosen), "added": names[best_j], "val_NMAE": float(best)})
    return chosen, curve


def main():
    b = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    D = b["descriptors"].numpy(); Y = b["labels"].numpy()
    r2i = {r: i for i, r in enumerate(b["reaction_ids"])}
    print(f"loaded {D.shape[0]}×{D.shape[1]}", flush=True)

    Dd, names_d, dropped = dedup(D, NAMES)
    with open(OUT / "dedup_report.md", "w") as f:
        f.write(f"# SPEC_04 — dedup\n\n- true D: {D.shape[1]}\n- expected: 24\n"
                f"- MATCH: {D.shape[1] == 24}\n\n## Dropped duplicates (|r|≥0.98)\n\n")
        f.write("- none\n" if not dropped else "")
        for d_ in dropped:
            f.write(f"- {d_[0]} dropped (~ {d_[1]}, r={d_[2]:+.3f})\n")

    print(f"kept {len(names_d)}/{D.shape[1]}", flush=True)
    sc = StandardScaler().fit(Dd); Xz = sc.transform(Dd)
    v = vif(Xz)
    pd.DataFrame({"descriptor": names_d, "VIF": v, "severe": v > 10, "moderate": v > 5}).to_csv(OUT / "vif.csv", index=False)

    fdir = SPLITS / "fold0"
    tr = np.array([r2i[r] for r in json.load(open(fdir / f"size_{SIZE_FULL}.json")) if r in r2i])
    Xtrz = sc.transform(Dd[tr])
    beta_ch = {}
    for i, ch in enumerate(CH + ["barrier"]):
        y = Y[tr, i] if ch != "barrier" else Y[tr].sum(axis=1)
        m = _OLSFit(Xtrz, y)
        ci = m.conf_int()
        rows = [{"descriptor": n, "beta": m.params[j + 1], "SE": m.bse[j + 1],
                 "t": m.tvalues[j + 1], "p": m.pvalues[j + 1],
                 "CI_low": ci[j + 1, 0], "CI_high": ci[j + 1, 1]}
                for j, n in enumerate(names_d)]
        pd.DataFrame(rows).to_csv(OUT / "ols_tables" / f"{ch}.csv", index=False)
        beta_ch[ch] = np.abs(m.params[1:])

    kf = KFold(5, shuffle=True, random_state=42)
    tr_ix, va_ix = next(iter(kf.split(tr)))
    sat = {}
    for i, ch in enumerate(CH + ["barrier"]):
        y = Y[tr, i] if ch != "barrier" else Y[tr].sum(axis=1)
        _, curve = forward_sel(Xtrz[tr_ix], y[tr_ix], Xtrz[va_ix], y[va_ix], names_d)
        sat[ch] = curve

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    for k, ch in enumerate(CH + ["barrier"]):
        ax = axes[k // 3, k % 3]
        c = sat[ch]
        ax.plot([x["n"] for x in c], [x["val_NMAE"] for x in c], "o-", color="#1f4e79", ms=4)
        ax.set_title(ch); ax.set_xlabel("# descriptors"); ax.set_ylabel("val NMAE"); ax.grid(alpha=0.3)
    fig.savefig(OUT / "saturation_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    for k, ch in enumerate(CH + ["barrier"]):
        y = Y[tr, k] if ch != "barrier" else Y[tr].sum(axis=1)
        alphas_l, _, coefs = lars_path(Xtrz, y, method="lasso")
        ax = axes[k // 3, k % 3]
        for j in range(coefs.shape[0]):
            ax.plot(-np.log(np.maximum(alphas_l, 1e-12)), coefs[j], lw=0.7)
        ax.set_title(ch); ax.set_xlabel("−log α"); ax.set_ylabel("β̂"); ax.grid(alpha=0.3)
    fig.savefig(OUT / "lasso_paths.png", dpi=150); plt.close(fig)

    heat = np.stack([beta_ch[c] for c in CH + ["barrier"]], axis=0)
    heat = heat / (heat.max(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(len(names_d) * 0.5 + 2, 4))
    im = ax.imshow(heat, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(CH) + 1)); ax.set_yticklabels(CH + ["barrier"])
    ax.set_xticks(range(len(names_d))); ax.set_xticklabels(names_d, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="|β̂| normalized")
    fig.tight_layout(); fig.savefig(OUT / "importance_heatmap.png", dpi=150); plt.close(fig)

    core = set()
    for ch, curve in sat.items():
        core.add(curve[0]["added"])
        for i in range(1, len(curve)):
            gain = curve[i - 1]["val_NMAE"] - curve[i]["val_NMAE"]
            if gain < 0.005: break
            core.add(curve[i - 1]["added"])
    (OUT / "reduced_set_proposal.md").write_text(
        f"# reduced-set proposal (elbow ΔNMAE<0.005)\n\ncore = {sorted(core)}\n"
        f"|core| = {len(core)}\n")

    lines = ["# SPEC_04 — descriptor contribution & parsimony (m3, 787-rxn cohort)",
             "", f"- true D = {D.shape[1]} (expected 24)",
             f"- after dedup: {len(names_d)}",
             f"- VIF max: {v.max():.2f}", "",
             f"## Reduced core set: {sorted(core)}"]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("SPEC_04 done")


if __name__ == "__main__":
    main()
