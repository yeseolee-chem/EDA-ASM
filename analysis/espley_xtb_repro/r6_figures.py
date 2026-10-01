#!/usr/bin/env python3
"""r6_figures.py — rev 6 Phase E figures (spec §10): 183 mm wide, 7 pt, PDF + PNG 300 dpi + the plotted values as CSV,
all from results_rev6/ files (B_*, A_*, R2_*, R5_report.json, B_bias_check.csv).

  fig1_primary      4 targets x 4 arms: representative model, 5-seed mean MAE bar + the 5 seed points; the primary
                    comparison (ESP-C vs OUR-O) annotated with Δ and Holm p
  fig2_models       4 targets: arm x model MAE heat maps (all 6 models, all 4 arms)
  fig3_parity       ESP-C rep / OUR-O rep x 4 targets, test predictions of the 5 seeds pooled
  fig4_seed_pairs   per target, the 5 paired seed MAEs ESP-C rep -> OUR-O rep
  fig5_channels     8 channels x 6 models MAE (OUR-O, Phase A main) with ESP-O reference bars
  fig6_sumchannels  Int, ΔE‡: OUR-O direct rep vs Σ-channel vs ESP-C rep (seed means + points)
  fig7_recovery     Phase R recovery per rev 4 failure type + the §2-2 bias check (D vs C, 95 % CI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev6_common as R6  # noqa: E402

OUT = R6.RES6
FIG = OUT / "figures"
TLAB = {"d1": "d1 (dipole strain)", "d2": "d2 (dipolarophile strain)", "Int": "Int (eint_spe)", "dEa": "ΔE‡",
        "elst": "elst", "pauli": "Pauli", "oi": "OI", "disp": "disp", "cpcm": "CPCM", "cds": "CDS",
        "c_ghost": "c_ghost"}
ARMS = ["ESP-C", "ESP-O", "OUR-C", "OUR-O"]
FONT = "Arial" if any("Arial" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": R6.FONT_PT, "axes.titlesize": R6.FONT_PT,
                     "axes.labelsize": R6.FONT_PT, "xtick.labelsize": R6.FONT_PT - 1, "ytick.labelsize": R6.FONT_PT - 1,
                     "legend.fontsize": R6.FONT_PT - 1, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def save(fig, name, data):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    R6.write_atomic(FIG / f"{name}.csv", lambda t: pd.DataFrame(data).to_csv(t, index=False))


def pfmt(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def fig1(ps, rep, prim):
    fig, axs = plt.subplots(1, 4, figsize=(R6.FIG_W, 2.2))
    rows = []
    for ax, t in zip(axs, R6.TARGETS_B):
        for k, a in enumerate(ARMS):
            m = rep[a][t]["model"]
            v = ps[(ps.arm == a) & (ps.model == m) & (ps.target == t) & (ps.rowset == "C")].sort_values("seed")
            ax.bar(k, v.mae.mean(), color=R6.ARM_COLORS[a], width=0.7)
            ax.scatter(np.full(len(v), k) + np.linspace(-0.2, 0.2, len(v)), v.mae, s=6, color="k", zorder=3)
            rows += [dict(target=t, arm=a, model=m, seed=int(s), mae=float(x)) for s, x in zip(v.seed, v.mae)]
        ax.set_xticks(range(4), [f"{a}\n{rep[a][t]['model']}" for a in ARMS])
        ax.set_title(TLAB[t])
        pr = prim[prim.target == t].iloc[0]
        ax.text(0.5, 0.98, f"ESP-C − OUR-O = {pr.d_mean:+.2f}\nHolm p = {pfmt(pr.p_holm)}", transform=ax.transAxes,
                ha="center", va="top")
    axs[0].set_ylabel("test MAE (kcal/mol)")
    fig.tight_layout()
    save(fig, "fig1_primary", rows)


def fig2(sm):
    fig, axs = plt.subplots(1, 4, figsize=(R6.FIG_W, 1.9))
    rows = []
    for ax, t in zip(axs, R6.TARGETS_B):
        x = sm[(sm.target == t) & (sm.rowset == "C") & sm.model.isin(R6.MODELS)]
        M = x.pivot(index="arm", columns="model", values="mae").reindex(index=ARMS, columns=R6.MODELS)
        im = ax.imshow(M.to_numpy(), cmap="viridis_r", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.iat[i, j]
                ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", color="w",
                        fontsize=R6.FONT_PT - 2)
                rows.append(dict(target=t, arm=M.index[i], model=M.columns[j], mae=v))
        ax.set_xticks(range(len(R6.MODELS)), R6.MODELS, rotation=45)
        ax.set_yticks(range(len(ARMS)), ARMS if ax is axs[0] else [""] * len(ARMS))
        ax.set_title(TLAB[t])
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.tight_layout()
    save(fig, "fig2_models", rows)


def fig3(pred, rep):
    fig, axs = plt.subplots(2, 4, figsize=(R6.FIG_W, 3.8))
    rows = []
    for j, t in enumerate(R6.TARGETS_B):
        for i, a in enumerate(("ESP-C", "OUR-O")):
            m = rep[a][t]["model"]
            q = pred[(pred.arm == a) & (pred.model == m) & (pred.target == t) & (pred.rowset == "C")]
            ax = axs[i, j]
            ax.scatter(q.y_true, q.y_pred, s=1, alpha=0.3, color=R6.ARM_COLORS[a], rasterized=True)
            lo, hi = np.nanmin([q.y_true.min(), q.y_pred.min()]), np.nanmax([q.y_true.max(), q.y_pred.max()])
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.5)
            mae = float(np.mean(np.abs(q.y_pred - q.y_true)))
            ax.set_title(f"{a} {m}: {TLAB[t]}\nMAE {mae:.2f} (5 seeds pooled)")
            ax.set_xlabel("DFT (kcal/mol)")
            if j == 0:
                ax.set_ylabel("predicted (kcal/mol)")
            rows += q[["seed", "rxn_id", "y_true", "y_pred"]].assign(arm=a, model=m, target=t).to_dict("records")
    fig.tight_layout()
    save(fig, "fig3_parity", rows)


def fig4(ps, rep, prim):
    fig, axs = plt.subplots(1, 4, figsize=(R6.FIG_W, 1.9))
    rows = []
    for ax, t in zip(axs, R6.TARGETS_B):
        a = ps[(ps.arm == "ESP-C") & (ps.model == rep["ESP-C"][t]["model"]) & (ps.target == t) & (ps.rowset == "C")]
        b = ps[(ps.arm == "OUR-O") & (ps.model == rep["OUR-O"][t]["model"]) & (ps.target == t) & (ps.rowset == "C")]
        a, b = a.set_index("seed").mae, b.set_index("seed").mae
        for s in R6.SEEDS:
            ax.plot([0, 1], [a[s], b[s]], "-o", color="0.4", ms=2.5, lw=0.7)
            rows.append(dict(target=t, seed=s, esp_c=a[s], our_o=b[s]))
        pr = prim[prim.target == t].iloc[0]
        ax.set_xticks([0, 1], [f"ESP-C\n{rep['ESP-C'][t]['model']}", f"OUR-O\n{rep['OUR-O'][t]['model']}"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{TLAB[t]}: OUR-O lower in {pr.seeds_b_lower}/5")
    axs[0].set_ylabel("test MAE (kcal/mol)")
    fig.tight_layout()
    save(fig, "fig4_seed_pairs", rows)


def fig5(smA):
    chs = list(R6.CHANNELS8)
    fig, axs = plt.subplots(2, 4, figsize=(R6.FIG_W, 3.6))
    rows = []
    for ax, ch in zip(axs.ravel(), chs):
        x = smA[(smA.target == ch) & (smA.rowset == "C") & smA.model.isin(R6.MODELS)]
        for k, (a, off) in enumerate((("OUR-O", -0.2), ("ESP-O", 0.2))):
            v = x[x.arm == a].set_index("model").reindex(R6.MODELS)
            ax.bar(np.arange(6) + off, v.mae, 0.4, yerr=v.mae_se, color=R6.ARM_COLORS[a], label=a,
                   error_kw=dict(lw=0.5, capsize=1))
            rows += [dict(channel=ch, arm=a, model=m, mae=r.mae, se=r.mae_se) for m, r in v.iterrows()]
        ax.set_xticks(range(6), R6.MODELS, rotation=45)
        ax.set_title(TLAB[ch])
    axs[0, 0].set_ylabel("test MAE (kcal/mol)")
    axs[1, 0].set_ylabel("test MAE (kcal/mol)")
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig5_channels", rows)


def fig6(ps, sig_ps, rep):
    fig, axs = plt.subplots(1, 2, figsize=(R6.FIG_W * 0.6, 2.0))
    rows = []
    for ax, t in zip(axs, ("Int", "dEa")):
        sets = [("OUR-O direct", ps[(ps.arm == "OUR-O") & (ps.model == rep["OUR-O"][t]["model"]) & (ps.target == t)
                                    & (ps.rowset == "C")], R6.ARM_COLORS["OUR-O"]),
                ("OUR-O Σ channels", sig_ps[sig_ps.target == t], "#8c564b"),
                ("ESP-C direct", ps[(ps.arm == "ESP-C") & (ps.model == rep["ESP-C"][t]["model"]) & (ps.target == t)
                                    & (ps.rowset == "C")], R6.ARM_COLORS["ESP-C"])]
        for k, (lab, v, c) in enumerate(sets):
            ax.bar(k, v.mae.mean(), color=c, width=0.7)
            ax.scatter(np.full(len(v), k) + np.linspace(-0.2, 0.2, len(v)), v.sort_values("seed").mae, s=6,
                       color="k", zorder=3)
            rows += [dict(target=t, set=lab, seed=int(s), mae=float(m)) for s, m in zip(v.seed, v.mae)]
        ax.set_xticks(range(3), [s[0].replace(" ", "\n", 1) for s in sets])
        ax.set_title(TLAB[t])
    axs[0].set_ylabel("test MAE (kcal/mol)")
    fig.tight_layout()
    save(fig, "fig6_sumchannels", rows)


def fig7():
    fig, axs = plt.subplots(1, 2, figsize=(R6.FIG_W, 2.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    rows = []
    tab = pd.read_csv(OUT / "R2_by_reason.csv")
    ax = axs[0]
    tab = tab.sort_values("n_rev4", ascending=False)
    y = np.arange(len(tab))
    ax.barh(y, tab.n_recovered, color=R6.ARM_COLORS["OUR-O"], label="recovered")
    ax.barh(y, tab.n_remaining, left=tab.n_recovered, color="0.75", label="remaining")
    ax.set_yticks(y, tab.category)
    ax.invert_yaxis()
    ax.set_xlabel("rev 4 G1 failures (reactions)")
    ax.legend(frameon=False)
    rows += tab[["category", "n_rev4", "n_recovered", "n_remaining"]].assign(panel="recovery").to_dict("records")
    b = pd.read_csv(OUT / "B_bias_check.csv")
    b = b[b.representative]
    ax = axs[1]
    for k, r in enumerate(b.itertuples()):
        ax.errorbar(r.diff_D_minus_C, k, xerr=[[r.diff_D_minus_C - r.ci_lo], [r.ci_hi - r.diff_D_minus_C]],
                    fmt="o", color=R6.ARM_COLORS["ESP-C"], ms=3, lw=0.8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(b)), [f"{TLAB[t]} ({m}, n_D={n})" for t, m, n in zip(b.target, b.model, b.n_D)])
    ax.set_xlabel("ESP-C mean |error|: dropped rows − C rows (kcal/mol)")
    rows += b.assign(panel="bias").to_dict("records")
    fig.tight_layout()
    save(fig, "fig7_recovery", rows)


def main():
    ps = pd.read_csv(OUT / "B_per_seed.csv")
    sm = pd.read_csv(OUT / "B_summary.csv")
    rep = json.loads((OUT / "B_representative.json").read_text())["C"]
    prim = pd.read_csv(OUT / "B_primary_tests.csv")
    pred = pd.read_parquet(OUT / "B_predictions.parquet")
    smA = pd.read_csv(OUT / "A_summary.csv")
    sig = pd.read_csv(OUT / "B_sigma_per_seed.csv")
    fig1(ps, rep, prim)
    fig2(sm)
    fig3(pred, rep)
    fig4(ps, rep, prim)
    fig5(smA)
    fig6(ps, sig, rep)
    fig7()
    R6.write_json(FIG / "figures.json", dict(font=FONT, width_mm=183, font_pt=R6.FONT_PT,
                                             files=sorted(p.name for p in FIG.iterdir())))
    print("font", FONT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
