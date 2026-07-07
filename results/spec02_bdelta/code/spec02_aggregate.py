"""SPEC_02 aggregate — M_b (ridge only) / M_δ (b≡0 retrained) / M_bδ (m3).

Outputs (results/spec02_bdelta/):
  decomposition_metrics.csv, decomposition_summary.csv,
  family_breakdown.csv, cancellation.csv, variance_decomposition.csv,
  contribution_bars.png, parity_3models.png, summary.md
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt")
SPLITS = REPO / "pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood"
OUT = REPO / "results" / "spec02_bdelta"
OUT.mkdir(parents=True, exist_ok=True)
FAMS = json.load(open(REPO / "pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.families.json"))

CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
SEED = 42; SIZE_FULL = 509
COLORS = {"M_b": "#1f4e79", "M_delta": "#2b8a89", "M_bdelta": "#c25a5a"}


def channel_metrics(yt, yp):
    e = yp - yt
    mae = float(np.mean(np.abs(e))); rmse = float(np.sqrt(np.mean(e ** 2)))
    ybar = float(yt.mean()); denom = float(np.mean(np.abs(yt - ybar)))
    nmae = mae / denom if denom > 0 else np.nan
    ss_tot = float(np.sum((yt - ybar) ** 2))
    r2 = 1 - float(np.sum(e ** 2)) / ss_tot if ss_tot > 0 else np.nan
    x = yt - ybar; yc = yp - float(yp.mean())
    slope = float(np.sum(x * yc) / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "NMAE": nmae, "R2": r2, "slope": slope}


def ridge_fit(X, y, alpha=1.0):
    D = X.shape[1]; reg = np.eye(D) * alpha; reg[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def build_Mb():
    b = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    D = b["descriptors"].numpy(); Y = b["labels"].numpy()
    r2i = {r: i for i, r in enumerate(b["reaction_ids"])}
    cells = []
    for f in range(5):
        fdir = SPLITS / f"fold{f}"
        te = np.array([r2i[r] for r in json.load(open(fdir / "test_rids.json")) if r in r2i])
        tr = np.array([r2i[r] for r in json.load(open(fdir / f"size_{SIZE_FULL}.json")) if r in r2i])
        mu = D[tr].mean(axis=0); sd = D[tr].std(axis=0); sd[sd < 1e-8] = 1.0
        Xtr = np.concatenate([(D[tr] - mu) / sd, np.ones((len(tr), 1))], axis=1)
        Xte = np.concatenate([(D[te] - mu) / sd, np.ones((len(te), 1))], axis=1)
        y_pred = np.zeros_like(Y[te])
        for c in range(5):
            W = ridge_fit(Xtr, Y[tr, c]); y_pred[:, c] = Xte @ W
        cells.append({"fold": f, "reaction_ids": [b["reaction_ids"][i] for i in te],
                      "y_true": Y[te].tolist(), "y_pred": y_pred.tolist()})
    return cells


def load_cells(root):
    return [json.load(open(f)) for f in sorted(root.glob("fold*/member*.json"))]


def rows_from(name, cells):
    rows = []
    for c in cells:
        yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
        for i, ch in enumerate(CH):
            m = channel_metrics(yt[:, i], yp[:, i])
            rows.append({"variant": name, "fold": c["fold"], "member": c.get("member", 0),
                         "channel": ch, **m})
        m = channel_metrics(yt.sum(axis=1), yp.sum(axis=1))
        rows.append({"variant": name, "fold": c["fold"], "member": c.get("member", 0),
                     "channel": "barrier", **m})
    return rows


def pool_m0(cells):
    m0 = [c for c in cells if c.get("member", 0) == 0]
    yt = np.concatenate([np.array(c["y_true"]) for c in m0])
    yp = np.concatenate([np.array(c["y_pred"]) for c in m0])
    return yt, yp


def main():
    mb = build_Mb()
    md = load_cells(OUT / "m_delta")
    mbd = load_cells(REPO / "m3" / "results")

    rows = rows_from("M_b", mb) + rows_from("M_delta", md) + rows_from("M_bdelta", mbd)
    df = pd.DataFrame(rows); df.to_csv(OUT / "decomposition_metrics.csv", index=False)
    summ = df.groupby(["variant", "channel"])[["MAE", "RMSE", "NMAE", "R2", "slope"]].agg(["mean", "std"]).round(3)
    summ.to_csv(OUT / "decomposition_summary.csv")

    # family breakdown
    frows = []
    for name, cells in [("M_b", mb), ("M_delta", md), ("M_bdelta", mbd)]:
        for c in cells:
            yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
            fams = np.array([FAMS.get(r, "?") for r in c["reaction_ids"]])
            for fam in ("dipolar", "qmrxn20_e2", "qmrxn20_sn2", "rgd1"):
                mask = fams == fam
                if mask.sum() == 0: continue
                for i, ch in enumerate(CH):
                    m = channel_metrics(yt[mask, i], yp[mask, i])
                    frows.append({"variant": name, "family": fam, "fold": c["fold"],
                                  "channel": ch, "NMAE": m["NMAE"]})
                m = channel_metrics(yt[mask].sum(axis=1), yp[mask].sum(axis=1))
                frows.append({"variant": name, "family": fam, "fold": c["fold"],
                              "channel": "barrier", "NMAE": m["NMAE"]})
    pd.DataFrame(frows).to_csv(OUT / "family_breakdown.csv", index=False)

    # bars
    channels_bar = CH + ["barrier"]
    x = np.arange(len(channels_bar)); width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, name in enumerate(["M_b", "M_delta", "M_bdelta"]):
        means, stds = [], []
        for ch in channels_bar:
            sub = df[(df.variant == name) & (df.channel == ch)]
            means.append(sub.NMAE.mean() if len(sub) else np.nan)
            stds.append(sub.NMAE.std() if len(sub) else 0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=name,
               color=COLORS[name], capsize=3, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
    ax.set_ylabel("NMAE"); ax.set_xticks(x); ax.set_xticklabels(channels_bar)
    ax.legend(loc="upper right", framealpha=0.9); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "contribution_bars.png", dpi=150); plt.close(fig)

    # parity grid
    fig, axes = plt.subplots(3, len(channels_bar), figsize=(3.4 * len(channels_bar), 9))
    for r, name in enumerate(["M_b", "M_delta", "M_bdelta"]):
        cells = {"M_b": mb, "M_delta": md, "M_bdelta": mbd}[name]
        yt_a, yp_a = pool_m0(cells) if any(c.get("member", 0) == 0 for c in cells) else \
                     (np.concatenate([np.array(c["y_true"]) for c in cells]),
                      np.concatenate([np.array(c["y_pred"]) for c in cells]))
        for i, ch in enumerate(channels_bar):
            ax = axes[r, i]
            if ch == "barrier": yt = yt_a.sum(axis=1); yp = yp_a.sum(axis=1)
            else: yt = yt_a[:, CH.index(ch)]; yp = yp_a[:, CH.index(ch)]
            m = channel_metrics(yt, yp)
            ax.scatter(yt, yp, s=6, c=COLORS[name], alpha=0.6, edgecolor="none")
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=0.6)
            ax.text(0.03, 0.97, f"MAE={m['MAE']:.2f}\nNMAE={m['NMAE']:.2f}\nR²={m['R2']:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7)
            if r == 0: ax.set_title(ch, fontsize=10)
            if i == 0: ax.set_ylabel(f"{name}\ny_pred", fontsize=9)
            if r == 2: ax.set_xlabel("y_true", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "parity_3models.png", dpi=150); plt.close(fig)

    # cancellation
    canc = []
    for name, cells in [("M_b", mb), ("M_delta", md), ("M_bdelta", mbd)]:
        yt_a, yp_a = pool_m0(cells) if any(c.get("member", 0) == 0 for c in cells) else \
                     (np.concatenate([np.array(c["y_true"]) for c in cells]),
                      np.concatenate([np.array(c["y_pred"]) for c in cells]))
        e = yp_a - yt_a; e_bar = e.sum(axis=1)
        rho = float(np.mean(np.abs(e_bar)) / np.mean(np.sum(np.abs(e), axis=1)))
        cov = np.cov(e, rowvar=False)
        row = {"variant": name, "rho": rho}
        for i in range(5):
            for j in range(i, 5):
                row[f"cov_{CH[i]}_{CH[j]}"] = float(cov[i, j])
        canc.append(row)
    pd.DataFrame(canc).to_csv(OUT / "cancellation.csv", index=False)

    # variance decomposition (M_bδ)
    yt_a, yp_bd = pool_m0(mbd) if any(c.get("member", 0) == 0 for c in mbd) else \
                  (np.concatenate([np.array(c["y_true"]) for c in mbd]),
                   np.concatenate([np.array(c["y_pred"]) for c in mbd]))
    _, yp_b = pool_m0(mb) if any(c.get("member", 0) == 0 for c in mb) else \
              (np.concatenate([np.array(c["y_true"]) for c in mb]),
               np.concatenate([np.array(c["y_pred"]) for c in mb]))
    n = min(len(yp_bd), len(yp_b)); delta = yp_bd[:n] - yp_b[:n]
    vrow = []
    for i, ch in enumerate(CH):
        vb = float(np.var(yp_b[:n, i])); vd = float(np.var(delta[:, i]))
        vc = float(np.cov(yp_b[:n, i], delta[:, i])[0, 1]); vy = float(np.var(yp_bd[:n, i]))
        vrow.append({"channel": ch, "Var_b": vb, "Var_delta": vd, "Cov_b_delta": vc,
                     "Var_y_hat": vy, "b_fraction": vb / vy if vy > 0 else np.nan,
                     "delta_fraction": vd / vy if vy > 0 else np.nan})
    pd.DataFrame(vrow).to_csv(OUT / "variance_decomposition.csv", index=False)

    # summary
    lines = ["# SPEC_02 — b / δ decomposition (m3, 787-rxn cohort)", "",
             "3 variants under matched split + HP:",
             f"- M_b:      {len(mb)} fold-avg (analytic ridge α=1)",
             f"- M_δ:      {len(md)} cells",
             f"- M_bδ:     {len(mbd)} cells", "",
             "## Mean NMAE across cells", "",
             "| channel | M_b | M_δ | M_bδ |", "|---|---|---|---|"]
    for ch in channels_bar:
        row = f"| {ch} |"
        for name in ("M_b", "M_delta", "M_bdelta"):
            sub = df[(df.variant == name) & (df.channel == ch)]
            row += f" {sub.NMAE.mean():.3f} ± {sub.NMAE.std():.3f} |"
        lines.append(row)
    lines += ["", "## Cancellation ρ (barrier)", ""]
    for c in canc: lines.append(f"- {c['variant']}: ρ = {c['rho']:.3f}")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("SPEC_02 aggregate done")


if __name__ == "__main__":
    main()
