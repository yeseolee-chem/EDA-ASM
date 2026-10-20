#!/usr/bin/env python3
"""d1_figures.py — D1PAIR Phase E figures (§10-2): 183 mm wide, 7 pt, PDF + PNG (300 dpi) + the plotted values as CSV.

  python d1_figures.py        # results/figures/fig{1..7}_*.{pdf,png,csv} + CAPTIONS.md

Reads only results/ files (E_metrics.csv, E_bends.csv, M_predictions.parquet via d1_eval.rep_table, S_folds.csv).
Colors: the reference categorical palette in fixed slot order (rev 5/6 used the same hexes): A slot 1, B slot 2, C slot 3,
P slot 4; null = muted gray; text in ink tokens; solid hairline grid; legends for >= 2 series.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402
import d1_eval as E  # noqa: E402
import hammett_eval as HE  # noqa: E402

FIG = C.RES / "figures"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ARM_COLOR = {"null": "#898781", "A": SLOT[0], "B": SLOT[1], "C": SLOT[2], "P": SLOT[3]}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TLAB = {"d1": "d1", "d2": "d2", "elst": "elst", "pauli": "Pauli", "oi": "OI", "disp": "disp", "cpcm": "CPCM", "cds": "CDS",
        "c_ghost": "c_ghost", "barrier": "barrier"}
CAPTIONS = {}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": C.FONT_PT, "axes.titlesize": C.FONT_PT, "axes.labelsize": C.FONT_PT,
                         "xtick.labelsize": C.FONT_PT - 1, "ytick.labelsize": C.FONT_PT - 1, "legend.fontsize": C.FONT_PT - 1,
                         "font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100})
    return plt


def save(fig, name, df, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        C.assert_write_ok(p)
        fig.savefig(p, bbox_inches="tight")
    C.write_atomic(FIG / f"{name}.csv", lambda t: df.to_csv(t, index=False))
    CAPTIONS[name] = caption


def fig1(plt, Mt):
    d = Mt[(Mt.split == "T1") & Mt.rep & (Mt.set == "P_main") & Mt.arm.isin(["null", "A", "B", "C", "P"])]
    d = d[["arm", "model", "target", "pos", "n", "mae"]]
    fig, axs = plt.subplots(2, 5, figsize=(C.FIG_W, C.FIG_W * 0.42), constrained_layout=True)
    pos = ["all"] + C.POSITIONS
    arms = ["null", "A", "B", "C", "P"]
    w = 0.16
    for ax, t in zip(axs.ravel(), C.TARGETS):
        q = d[d.target == t]
        for k, a in enumerate(arms):
            v = [q[(q.arm == a) & (q.pos == p)].mae.mean() for p in pos]
            ax.bar(np.arange(len(pos)) + (k - 2) * w, v, width=w * 0.85, color=ARM_COLOR[a], label=a, linewidth=0)
        ax.set_xticks(range(len(pos)), pos, rotation=0)
        ax.set_title(TLAB[t])
        ax.grid(axis="x", visible=False)
    axs[0, 0].set_ylabel("ΔΔ MAE (kcal/mol)"); axs[1, 0].set_ylabel("ΔΔ MAE (kcal/mol)")
    axs[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig1_mae", d, "Test-pair ΔΔ MAE (T1, P_main reference pairs, all outer folds pooled) of the representative "
         "model of each arm, overall and by X position. null = 0, A = f0 difference, B = f0 difference + g, C = g alone, "
         "P = one KRR on D0 + training-side D1 reactions.")
    plt.close(fig)


def fig2(plt, L):
    d = L[(L.split == "T1") & (L.arm == "B") & L.rep & L.in_P_main & (L.ref_kind == "reference") & (L.pos == "p")]
    d = d[["pair_id", "target", "model", "y", "yhat"]]
    fig, axs = plt.subplots(2, 5, figsize=(C.FIG_W, C.FIG_W * 0.42), constrained_layout=True)
    for ax, t in zip(axs.ravel(), C.TARGETS):
        q = d[d.target == t]
        lo, hi = np.nanmin(np.r_[q.y, q.yhat]), np.nanmax(np.r_[q.y, q.yhat])
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6)
        ax.scatter(q.y, q.yhat, s=4, color=SLOT[1], linewidths=0, alpha=0.7)
        mae = float(np.mean(np.abs(q.yhat - q.y))) if len(q) else np.nan
        ax.set_title(f"{TLAB[t]}  (B: {q.model.iloc[0] if len(q) else ''}, MAE {mae:.2f})")
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axs[1]:
        ax.set_xlabel("DFT ΔΔ (kcal/mol)")
    for ax in axs[:, 0]:
        ax.set_ylabel("predicted ΔΔ (kcal/mol)")
    save(fig, "fig2_parity", d, "Parity of the arm B representative model on the para test pairs (T1, P_main), one panel per "
         "target; the gray line is y = x.")
    plt.close(fig)


def hseries(q, col):
    return HE.family_series(E.hammett_frame(q, col), q.fam.iloc[0], col, C.MIN_X_FAM)


def fig3(plt, L):
    tg = ["barrier", "oi", "elst", "cpcm"]
    d = L[(L.split == "T1") & (L.arm == "B") & L.rep & L.in_P_main & (L.ref_kind == "reference") & (L.pos == "p")
          & L.sigma.notna() & L.fam.isin(C.FIG3_FAMILIES)]
    fig, axs = plt.subplots(len(tg), len(C.FIG3_FAMILIES), figsize=(C.FIG_W, C.FIG_W * 0.75), constrained_layout=True,
                            squeeze=False)
    rows = []
    for j, f in enumerate(C.FIG3_FAMILIES):
        for i, t in enumerate(tg):
            ax = axs[i, j]
            q = d[(d.fam == f) & (d.target == t)]
            if len(q) == 0 or hseries(q, "y") is None:
                ax.text(0.5, 0.5, "not in P_main test set", ha="center", va="center", transform=ax.transAxes, color=MUTED)
                continue
            x, y, nm = hseries(q, "y")
            _, yh, _ = hseries(q, "pred")
            o = np.argsort(x)
            ax.scatter(x, y, s=10, color=SLOT[0], linewidths=0, label="DFT", zorder=3)
            ax.plot(x[o], yh[o], color=SLOT[1], lw=1.0, marker="o", ms=2.5, label="B prediction", zorder=2)
            rows += [dict(fam=f, target=t, X=n, sigma_p=a, y_dft=b, y_pred=c) for n, a, b, c in zip(nm, x, y, yh)]
            if i == 0:
                ax.set_title(f.replace("/", " / ").replace(" /  / ", " / "), fontsize=C.FONT_PT - 1)
            if j == 0:
                ax.set_ylabel(f"{TLAB[t]}: −ΔΔ/1.3642")
            if i == len(tg) - 1:
                ax.set_xlabel("σp")
    axs[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig3_hammett", pd.DataFrame(rows),
         "Hammett plots (y = −ΔΔ/1.3642, log k units, H at the origin) of six pre-selected para families: DFT points and the "
         "arm B representative predictions (T1 test pairs). The first family (nitrile oxide + para-X aryl alkene, skeleton C) "
         "is the same reaction type as the experimental V-shaped Hammett plot of nitrile oxide + para-X-phenylacetylene "
         "(ρ = −0.34 / +0.61; J. Chem. Soc. B 1967, doi:10.1039/J29670000867); the dipole and the solvent differ, so it is not a "
         "direct comparison.")
    plt.close(fig)


def fig4(plt, Bd):
    rows = []
    for c, r in E.bend_rates(Bd, "T1").items():
        rows.append(dict(target=c, sens=r["sens"], sens_lo=r["sens_ci"][0], sens_hi=r["sens_ci"][1], n_bent=r["n_bent_dft"],
                         fpr=r["fpr"], fpr_lo=r["fpr_ci"][0], fpr_hi=r["fpr_ci"][1], n_straight=r["n_straight_dft"]))
    d = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(C.FIG_W, C.FIG_W * 0.3), constrained_layout=True)
    x = np.arange(len(d))
    for ax, k, lab, ref in ((axs[0], "sens", "bend-call sensitivity", C.STAGE2_SENS), (axs[1], "fpr", "false-positive rate",
                                                                                         C.STAGE2_FPR)):
        v = d[k].to_numpy(float)
        err = np.vstack([v - d[f"{k}_lo"], d[f"{k}_hi"] - v])
        ax.bar(x, v, width=0.5, color=SLOT[1], linewidth=0)
        ax.errorbar(x, v, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5)
        ax.axhline(ref, color=MUTED, lw=0.6)
        ax.set_xticks(x, [TLAB[t] for t in d.target], rotation=45, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
        for xi, n in zip(x, d["n_bent" if k == "sens" else "n_straight"]):
            ax.text(xi, 1.0, f"n={n}", ha="center", va="bottom", fontsize=C.FONT_PT - 2, color=INK2)
    save(fig, "fig4_bends", d, "Agreement of the hinge (bend) call of appendix D between DFT and the arm B representative "
         "predictions on the T1 test para families: sensitivity (DFT bent, prediction bent) and false-positive rate (DFT "
         "straight, prediction bent), Clopper-Pearson 95 % intervals; gray lines = stage-2 thresholds (0.7, 0.1).")
    plt.close(fig)


def fig5(plt, L):
    rows = []
    for arm, col in (("A", SLOT[0]), ("B", SLOT[1]), ("C", SLOT[2])):
        for t in C.TARGETS[:8] + ["barrier"]:
            q = L[(L.split == "T1") & (L.arm == arm) & L.rep & (L.target == t)]
            fe = E.family_eval(q)
            rows += [dict(arm=arm, target=t, fam=r.fam, r2_fam=r.r2_fam) for r in fe.itertuples()]
    d = pd.DataFrame(rows)
    tg = C.TARGETS[:8] + ["barrier"]
    fig, ax = plt.subplots(figsize=(C.FIG_W, C.FIG_W * 0.33), constrained_layout=True)
    for k, (arm, col) in enumerate((("A", SLOT[0]), ("B", SLOT[1]), ("C", SLOT[2]))):
        data = [d[(d.arm == arm) & (d.target == t)].r2_fam.clip(lower=-1).to_numpy() for t in tg]
        bp = ax.boxplot(data, positions=np.arange(len(tg)) + (k - 1) * 0.25, widths=0.18, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=0.8), whiskerprops=dict(color=INK2, lw=0.5),
                        capprops=dict(color=INK2, lw=0.5), boxprops=dict(lw=0))
        for b in bp["boxes"]:
            b.set_facecolor(col)
        ax.plot([], [], color=col, lw=4, label=arm)
    ax.axhline(C.STAGE1_R2, color=MUTED, lw=0.6)
    ax.set_xticks(range(len(tg)), [TLAB[t] for t in tg]); ax.set_ylim(-1.02, 1.02)
    ax.set_ylabel("R²_fam (clipped at −1)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncol=3)
    save(fig, "fig5_r2fam", d, "Within-family R² (R²_fam, H included, log units) of the para test families, T1, arms A, B, C "
         "representative models; gray line = stage-1 threshold 0.5. Values below −1 are drawn at −1 (the CSV keeps them).")
    plt.close(fig)


def fig6(plt, Mt):
    d = Mt[Mt.split.isin(["T1", "LODO", "LOSO"]) & (Mt.arm == "B") & Mt.rep & (Mt.set == "P_main") & (Mt.pos == "all")]
    d = d[["split", "target", "model", "n", "mae"]]
    fig, ax = plt.subplots(figsize=(C.FIG_W, C.FIG_W * 0.3), constrained_layout=True)
    x = np.arange(len(C.TARGETS))
    for k, (sp, col) in enumerate((("T1", SLOT[0]), ("LODO", SLOT[1]), ("LOSO", SLOT[2]))):
        v = [d[(d.split == sp) & (d.target == t)].mae.mean() for t in C.TARGETS]
        ax.bar(x + (k - 1) * 0.25, v, width=0.22, color=col, label=sp, linewidth=0)
    ax.set_xticks(x, [TLAB[t] for t in C.TARGETS]); ax.set_ylabel("ΔΔ MAE (kcal/mol)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "fig6_lodo", d, "Arm B representative MAE on P_main test pairs: core 5-fold (T1) vs leave-one-dipole-out (LODO) "
         "and leave-one-skeleton-out (LOSO).")
    plt.close(fig)


def fig7(plt, L, Bd):
    fams = sorted(C.T2_HIDDEN_V)
    rows = []
    fig, axs = plt.subplots(2, len(fams), figsize=(C.FIG_W, C.FIG_W * 0.42), constrained_layout=True, squeeze=False)
    for j, f in enumerate(fams):
        vch = Bd[(Bd.split == "T1") & (Bd.fam == f) & Bd.V_dft & (Bd.target != "barrier")].target.tolist()
        ch = vch[0] if vch else None
        for i, t in enumerate([ch, "barrier"]):
            ax = axs[i, j]
            if t is None:
                ax.text(0.5, 0.5, "no V channel in P_main", ha="center", va="center", transform=ax.transAxes, color=MUTED)
                continue
            for sp, col, lab in (("T1", SLOT[0], "T1 prediction"), ("T2", SLOT[1], "T2 prediction (no V seen)")):
                q = L[(L.split == sp) & (L.arm == "B") & L.rep & L.in_P_main & (L.ref_kind == "reference") & (L.pos == "p")
                      & L.sigma.notna() & (L.fam == f) & (L.target == t)]
                if len(q) == 0 or hseries(q, "y") is None:
                    continue
                x, y, nm = hseries(q, "y")
                _, yh, _ = hseries(q, "pred")
                o = np.argsort(x)
                if sp == "T1":
                    ax.scatter(x, y, s=10, color=INK, linewidths=0, label="DFT", zorder=3)
                ax.plot(x[o], yh[o], color=col, lw=1.0, marker="o", ms=2.2, label=lab, zorder=2)
                rows += [dict(fam=f, target=t, split=sp, X=n, sigma_p=a, y_dft=b, y_pred=c) for n, a, b, c in zip(nm, x, y, yh)]
            if i == 0:
                ax.set_title(f, fontsize=C.FONT_PT - 2)
            ax.set_ylabel(f"{TLAB[t]}: −ΔΔ/1.3642" if t else "")
            if i == 1:
                ax.set_xlabel("σp")
    axs[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig7_vblind", pd.DataFrame(rows), "Hidden-V families (barrier straight, a channel V-shaped in DFT): top, the V "
         "channel; bottom, the barrier. DFT points, arm B representative predictions with every family seen in training (T1) "
         "and with the 11 V families never in training (T2). Six families only (five are nitrones): case evidence.")
    plt.close(fig)


def main():
    plt = style()
    S, M, U, f0r, f0 = E.load()
    L, reps = E.rep_table(S, M, U, f0, f0r)
    Mt = pd.read_csv(C.RES / "E_metrics.csv")
    Bd = pd.read_csv(C.RES / "E_bends.csv")
    fig1(plt, Mt); fig2(plt, L); fig3(plt, L); fig4(plt, Bd); fig5(plt, L); fig6(plt, Mt)
    import d1_train as TR
    if "T2" in TR.skipped():                               # FU1 K-3: T2 stopped by the user (2026-10-20)
        FIG.mkdir(parents=True, exist_ok=True)
        C.write_atomic(FIG / "fig7_vblind_NOT_RUN.txt", "fig7 (hidden-V, T2) not drawn: T2 was stopped by user decision "
                       "2026-10-20 (D1PAIR_FU1 F1); finished T2 units are kept but not analysed.\n")
    else:
        fig7(plt, L, Bd)
    md = ["# D1PAIR figures — captions", ""] + [f"- **{k}**: {v}" for k, v in CAPTIONS.items()]
    C.write_atomic(FIG / "CAPTIONS.md", "\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
