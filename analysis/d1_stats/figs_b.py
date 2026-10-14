"""figs_b.py — Phase B 판정 그림 (SPEC_D1_ALL B1b · B2 · B3 · B4 · B5 · B6 · B7 · B8). 부록 figs.py(fig1–4, B1)를 이어 fig05 부터.

python figs_b.py RESULTS_DIR
  RESULTS_DIR = results_final (run_all.py · Phase B 스크립트 출력). 그림은 RESULTS_DIR/fig/figNN_*.png (dpi 200, figs.py 와 같은 모양).
  입력이 없는 그림은 건너뛰고 (예: σ 표가 없으면 B3 · B6 · B5 σ 비대칭), fig/figs_b_index.json 에 만든 것 · 건너뛴 것 · 실패한 것을 적는다.
  그림마다 그리기 전에 지난 PNG 를 지운다 (건너뛰거나 실패한 그림이 옛 PNG 로 남아 보고서에 들어가지 않게). 쓰기는 원자적.
  종료 코드 1 = 어떤 그림이 예외로 실패 (입력이 없어서 건너뛴 것은 실패가 아니다). 숫자는 파일에서 옮기기만 한다 (새 통계 없음;
  화면 출력만 있던 값 — dipole 별 r, regio 크기 — 은 pairs_H.csv · sel_regio.csv 에서 같은 정의로 다시 센다).
  CI 는 [lo, hi] 구간을 그대로 그리고 점추정은 따로 찍는다 (백분위 CI 가 점추정을 포함하지 않을 수 있다).
"""
import ast
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RD = Path(sys.argv[1])
FD = RD / "fig"
FD.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style (= figs.py)
SURF, T1, T2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8c8b85"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIPS = ["azide", "azomethine ylide", "carbonyl ylide", "nitrile oxide", "nitrone"]
DC = dict(zip(DIPS, S + ["#8a5cc2"]))                     # 5th colour for the 5th dipole
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2423"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.edgecolor": "#bdbcb6", "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2,
                     "text.color": T1, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
CHN = {"d1_kcal": "d1", "d2_kcal": "d2", "elst_dft": "elst", "pauli_dft": "Pauli", "oi_dft": "OI", "disp_dft": "disp",
       "cpcm_dft": "CPCM", "cds_dft": "CDS", "bsse_shift_kcal": "bsse", "barrier_kcal": "ΔE‡", "pauli": "Pauli",
       "strain": "strain", "Eint": "E_int", "nosolv": "ΔE‡ − solv"}
INT = ["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal"]
X_ORDER = ["NMe2", "NH2", "OMe", "OH", "SMe", "Me", "tBu", "Ph", "Bn", "H", "SiMe3", "F", "Cl", "Br", "Ts", "Ac", "CHO",
           "CO2Me", "CF3", "CN", "NO2"]                       # roughly EDG -> EWG (display order only)
SKN = {"A": "vinyl-X (alkene)", "B": "X-norbornene", "C": "ethynyl-X (alkyne)", "D": "X-cyclooctyne"}
VERD = ["barrierless", "no_peak_short_scan", "peak", "no_path", "unparsed", "unread", "not_scanned"]
VC = {"barrierless": "#104281", "no_peak_short_scan": "#3987e5", "peak": "#eb6834", "no_path": "#bdbcb6",
      "unparsed": "#8c8b85", "unread": "#9e2423", "not_scanned": "#e4e3df"}
EN = {"큰 X": "bulky X", "계열 전체": "series, all", "골격": "skeleton"}   # Hangul in data labels -> English (no CJK font)

index = dict(made=[], skipped={}, failed={})


def csv(name, **kw):
    """The CSV as a frame, or None if absent / empty (a 0- or 1-byte file from an empty frame)."""
    f = RD / name
    if not f.is_file():
        return None
    try:
        d = pd.read_csv(f, **kw)
    except pd.errors.EmptyDataError:
        return None
    return d if len(d.columns) else None


def need(*names):
    """Required inputs: a missing or empty one skips the figure (FileNotFoundError), it is not a failure."""
    miss = [n for n in names if csv(n) is None]
    if miss:
        raise FileNotFoundError(", ".join(miss))


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.8, zorder=0)


def save(fig, name):
    fig.tight_layout()
    tmp = FD / f".{name}.tmp.png"
    fig.savefig(tmp, dpi=200)
    os.replace(tmp, FD / name)
    plt.close(fig)
    index["made"].append(name)


def ivl(ax, pos, lo, hi, horiz=True, color=T1, lw=1.0, cap=2.5):
    """Draw the interval [lo, hi] itself (with caps) at each position; NaN = no bar. Points are plotted separately."""
    pos, lo, hi = (np.atleast_1d(np.asarray(z, dtype=float)) for z in (pos, lo, hi))
    ok = np.isfinite(pos) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return
    mid, half = (lo + hi) / 2, np.abs(hi - lo) / 2
    kw = dict(fmt="none", ecolor=color, elinewidth=lw, capsize=cap, zorder=2.5)
    if horiz:
        ax.errorbar(mid[ok], pos[ok], xerr=half[ok], **kw)
    else:
        ax.errorbar(pos[ok], mid[ok], yerr=half[ok], **kw)


def ci(s):
    """'[a, b]' (b5_gedt CSV list strings) -> (a, b); anything else -> (nan, nan)."""
    try:
        a, b = ast.literal_eval(s) if isinstance(s, str) else (np.nan, np.nan)
        return float(a), float(b)
    except (ValueError, SyntaxError, TypeError):
        return np.nan, np.nan


def en(s):
    for k, v in EN.items():
        s = s.replace(k, v)
    return s


def boolcol(s):
    return s.astype(str).str.lower().isin(["true", "1"])


FIGS = []


def figure(fn):
    FIGS.append(fn)
    return fn


# ---------------------------------------------------------------- B1b
@figure
def fig05_b1b_eint_by_dipole():
    need("q5_dipole_eint.csv", "pairs_H.csv")
    Q = csv("q5_dipole_eint.csv")
    P = csv("pairs_H.csv"); P = P[~boolcol(P.q_bad)].copy()
    P["Eint"] = P[INT].sum(axis=1)
    H = json.load(open(RD / "headline.json")) if (RD / "headline.json").is_file() else {}
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(2, 1, 1)
    dips = [d for d in DIPS if d in set(Q.dipole)]
    var = list(dict.fromkeys(Q.variant))
    off = dict(zip(var, np.linspace(-0.27, 0.27, len(var))))
    mk = dict(zip(var, ["o", "s", "^", "D"]))
    for v in var:
        q = Q[Q.variant == v].set_index("dipole").reindex(dips)
        y = np.arange(len(dips)) + off[v]
        base = v == var[0]
        for i, d in enumerate(dips):
            c = DC[d] if base else T2
            ivl(ax, y[i], q.lo.iloc[i], q.hi.iloc[i], True, c, lw=1.2 if base else 0.8)
            ax.plot(q.r.iloc[i], y[i], mk[v], color=c, ms=6 if base else 4, zorder=3, label=v if i == 0 else None)
    ax.axvline(0, color=REF, lw=1)
    if "r_eint_barrier" in H:
        ax.axvline(H["r_eint_barrier"], color=REF, lw=1, ls="--")
        ax.text(H["r_eint_barrier"], len(dips) - 0.4, f" pooled r = {H['r_eint_barrier']:+.3f}", fontsize=7, color=T2)
    ax.set_yticks(range(len(dips))); ax.set_yticklabels(dips); ax.invert_yaxis(); grid(ax)
    ax.set_xlabel("r(ΔΔE_int, ΔΔE‡) over child − H-parent pairs  (bars: core-cluster bootstrap 95 % CI)")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    b = Q[Q.variant == var[0]]
    split = bool((b.lo > 0).any() and (b.hi < 0).any())
    ax.set_title("B1b — r(ΔΔE_int, ΔΔE‡) by dipole" + (": the sign splits across dipoles (do not read the pooled r as ‘none’)"
                                                        if split else " (base variant; compare with the pooled r)"))
    for j, d in enumerate(dips):
        a = fig.add_subplot(2, len(dips), len(dips) + j + 1)
        g = P[P.dipole == d]
        if len(g):
            a.hexbin(g.Eint, g.barrier_kcal, gridsize=28, mincnt=1, cmap="Blues", linewidths=0)
        a.axhline(0, color=REF, lw=0.6); a.axvline(0, color=REF, lw=0.6)
        rr = np.corrcoef(g.Eint, g.barrier_kcal)[0, 1] if len(g) > 2 else np.nan
        a.set_title(f"{d}\nr = {rr:+.2f} (n {len(g)})", fontsize=8)
        a.set_xlabel("ΔΔE_int", fontsize=8)
        if j == 0:
            a.set_ylabel("ΔΔE‡ (kcal/mol)", fontsize=8)
    save(fig, "fig05_b1b_eint_by_dipole.png")


@figure
def fig06_b1b_lowrank_ci():
    need("ci_lowrank_boot.csv")
    C = csv("ci_lowrank_boot.csv")
    M = ["subst", "additive", "rank1", "rank2", "rank3"]
    lab = ["scalar\nper X", "X + context\n(additive)", "rank 1\n(σ·ρ)", "rank 2", "rank 3"]
    jobs = list(dict.fromkeys(zip(C.side, C.ch)))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    rows = []
    for k, (sd, ch) in enumerate(jobs):
        g = C[(C.side == sd) & (C.ch == ch)]
        p0, bs = g[g.boot == 0], g[g.boot >= 1]
        if not len(p0):
            continue
        y = p0[M].iloc[0].to_numpy(float)
        lo, hi = (np.nanpercentile(bs[M], [2.5, 97.5], axis=0) if len(bs) else (np.full(5, np.nan), np.full(5, np.nan)))
        col = S[k % 4] if sd == "dipole" else "#9e9d97"
        x = np.arange(5) + 0.06 * k
        name = f"{sd} {CHN.get(ch, ch)} (n_boot {len(bs)})"
        ivl(axs[0], x, lo, hi, False, col)
        axs[0].plot(x, y, "o-", color=col, lw=1.6, ms=4, label=name, zorder=3)
        for a_, b_, nm in (("rank1", "subst", "rank1 − scalar"), ("rank2", "rank1", "rank2 − rank1"), ("rank3", "rank2", "rank3 − rank2")):
            d0 = float(p0[a_].iloc[0] - p0[b_].iloc[0]); db = (bs[a_] - bs[b_]).to_numpy(float)
            rows.append(dict(step=nm, d=d0, lo=np.nanpercentile(db, 2.5) if len(db) else np.nan,
                             hi=np.nanpercentile(db, 97.5) if len(db) else np.nan,
                             p_le0=float(np.mean(db <= 0)) if len(db) else np.nan, col=col, k=k))
    axs[0].set_xticks(range(5)); axs[0].set_xticklabels(lab, fontsize=7); grid(axs[0], "y")
    axs[0].axhline(0, color=REF, lw=1); axs[0].set_ylabel("cell-grouped 5-fold CV R² (boot 0; bars 2.5–97.5 % over core resamples)")
    axs[0].legend(frameon=False, fontsize=7); axs[0].set_title("How many substituent parameters?")
    D = pd.DataFrame(rows)
    steps = list(dict.fromkeys(D.step)) if len(D) else []
    for _, r in D.iterrows():
        x = steps.index(r.step) + 0.18 * (r.k - 1)
        ivl(axs[1], x, r.lo, r.hi, False, r.col)
        axs[1].plot(x, r.d, "o", color=r.col, zorder=3)
        if np.isfinite(r.hi):
            axs[1].text(x + 0.04, r.hi, f"P≤0 {r.p_le0:.2f}", fontsize=6.5, color=T2, va="bottom")
    axs[1].axhline(0, color=REF, lw=1); axs[1].set_xticks(range(len(steps))); axs[1].set_xticklabels(steps)
    grid(axs[1], "y"); axs[1].set_ylabel("paired ΔR² (same core resample)")
    axs[1].set_title("Gain per extra parameter (95 % CI; grey = dipolarophile side)")
    save(fig, "fig06_b1b_lowrank_ci.png")


# ---------------------------------------------------------------- B2
@figure
def fig07_b2_status():
    need("status_by_dipole.csv")
    St = csv("status_by_dipole.csv", index_col=0)
    ok = csv("ok_rate_dipole_X.csv", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1, 1.5]))
    fr = St.div(St.sum(axis=1), axis=0)
    order = (["ok"] if "ok" in fr else []) + [c for c in fr.columns if c != "ok"]
    cmap = plt.get_cmap("tab20")
    left = np.zeros(len(fr))
    for j, c in enumerate(order):
        axs[0].barh(range(len(fr)), fr[c], left=left, color="#1baf7a" if c == "ok" else cmap(j % 20), label=c, zorder=3)
        left += fr[c].to_numpy()
    axs[0].set_yticks(range(len(fr))); axs[0].set_yticklabels([f"{d} (n {int(St.loc[d].sum())})" for d in fr.index])
    axs[0].invert_yaxis(); axs[0].set_xlim(0, 1); grid(axs[0])
    axs[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left", bbox_to_anchor=(0, -0.55))
    axs[0].set_title("B2 — outcome class per dipole (computed jobs)"); axs[0].set_xlabel("share of D1 jobs")
    if ok is not None:
        xs = [x for x in X_ORDER if x in ok.index] + [x for x in ok.index if x not in X_ORDER]
        Z = ok.loc[xs].T
        im = axs[1].imshow(Z.to_numpy(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        axs[1].set_xticks(range(len(xs))); axs[1].set_xticklabels(xs, rotation=60, fontsize=7)
        axs[1].set_yticks(range(len(Z))); axs[1].set_yticklabels(Z.index)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z.iat[i, j]
                if np.isfinite(v):
                    axs[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < 0.35 else T1)
        fig.colorbar(im, ax=axs[1], fraction=0.03, label="ok rate")
        axs[1].set_title("ok rate by dipole × X (R1 + R2 series and direct + aryl pooled; X ordered EDG → EWG)", fontsize=8.5)
        for s in axs[1].spines.values():
            s.set_visible(False)
    save(fig, "fig07_b2_status.png")


@figure
def fig08_b2_missing_cause():
    need("missing_cause.csv", "missing_cause_cells.csv", "missing_cause_by_dipole.csv")
    N = csv("missing_cause.csv", dtype={"path_detail": str})
    Cc = csv("missing_cause_cells.csv")
    B = csv("missing_cause_by_dipole.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1.1, 1]))
    agg = Cc[~Cc.cell.str.contains("X=")].reset_index(drop=True)
    acc = {"left": np.zeros(len(agg))}

    def seg(col, color, label, hatch=None):
        x = agg[col].fillna(0).to_numpy(float) if col in agg else np.zeros(len(agg))
        if x.any():
            axs[0].barh(range(len(agg)), x, left=acc["left"], color=color, hatch=hatch, label=label, zorder=3)
        acc["left"] = acc["left"] + x
    seg("class_ok", "#1baf7a", "ok (labelled)")
    for v in VERD:
        seg(f"no_ts_{v}", VC[v], f"no_ts: {v}")
    for c in ("class_held", "class_pending", "class_running"):
        seg(c, "#f0efec", f"{c[6:]} (not computed)", hatch="//")
    other = agg.n.to_numpy(float) - acc["left"]
    if (other > 0).any():
        axs[0].barh(range(len(agg)), np.clip(other, 0, None), left=acc["left"], color="#bdbcb6", label="other failure classes", zorder=3)
    axs[0].set_yticks(range(len(agg))); axs[0].set_yticklabels([en(c) for c in agg.cell], fontsize=7); axs[0].invert_yaxis()
    grid(axs[0]); axs[0].set_xlabel("jobs"); axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    axs[0].set_title("Named empty cells: what happened (B2)")
    pv = B.pivot_table(index=["dipole", "series"], columns="path_verdict", values="n", aggfunc="sum").fillna(0)
    left2 = np.zeros(len(pv))
    for v in [v for v in VERD if v in pv.columns]:
        axs[1].barh(range(len(pv)), pv[v], left=left2, color=VC[v], label=v, zorder=3); left2 += pv[v].to_numpy()
    axs[1].set_yticks(range(len(pv))); axs[1].set_yticklabels([f"{d} | {s}" for d, s in pv.index], fontsize=7)
    axs[1].invert_yaxis(); grid(axs[1]); axs[1].set_xlabel("no_ts jobs"); axs[1].legend(frameon=False, fontsize=6.5)
    axs[1].set_title("no_ts verdict by dipole × series")
    nb = N[N.path_verdict.isin(["barrierless", "no_peak_short_scan"])]
    dm = nb.dform_react_min_A.dropna()
    lo_e = min(np.floor(dm.min() * 20) / 20, 2.95) if len(dm) else 2.5
    hi_e = max(3.31, np.ceil(dm.max() * 40) / 40 + 0.025) if len(dm) else 3.31
    bins = np.arange(lo_e, hi_e, 0.025)
    for d in DIPS:
        x = nb.loc[nb.dipole == d, "dform_react_min_A"].dropna()
        if len(x):
            axs[2].hist(x, bins=bins, histtype="step", lw=1.4, color=DC[d], label=f"{d} ({len(x)})")
    axs[2].axvline(3.0, color=T1, lw=1, ls="--"); axs[2].text(3.003, axs[2].get_ylim()[1] * 0.92, "3.0 Å rule", fontsize=7)
    axs[2].set_xlabel("reactant-end longer forming bond d_react (Å) = 2 × product bond")
    axs[2].set_ylabel("no-peak no_ts jobs"); axs[2].legend(frameon=False, fontsize=6.5); grid(axs[2], "y")
    axs[2].set_title("No interior maximum: where autodE's scan ended")
    fig.suptitle("B2 — missing labels: barrierless (no maximum, d_react ≥ 3.0 Å) vs no-peak short scan (d_react = 2 × product bond < 3.0 Å) vs peak",
                 fontsize=9.5, fontweight="bold")
    save(fig, "fig08_b2_missing_cause.png")


@figure
def fig09_b2_missing_logit():
    need("missing_logit.csv")
    L = csv("missing_logit.csv", index_col=0)
    L = L.drop(index=[i for i in L.index if i == "Intercept"])
    clip = 8.0
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.17 * len(L))))
    y = np.arange(len(L))
    sep = (L.se > 20).to_numpy()                                         # unidentified: huge se (separation)
    far = (L.coef.abs() > clip).to_numpy() & ~sep                        # identified but beyond the axis: clipped arrow
    c = L.coef.clip(-clip, clip).to_numpy()
    okm = ~sep & ~far
    ivl(ax, y[okm], (L.coef - 1.96 * L.se).to_numpy()[okm], (L.coef + 1.96 * L.se).to_numpy()[okm], True, S[0])
    ax.plot(c[okm], y[okm], "o", color=S[0], ms=3.5, zorder=3)
    if far.any():
        ax.plot(c[far], y[far], ">", color=S[0], ms=5, zorder=3, label=f"|coef| > {clip:g} (clipped; well identified)")
    if sep.any():
        ax.plot(c[sep], y[sep], "x", color=S[1], ms=6, zorder=3, label="unidentified (se > 20: separation)")
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([i.replace("C(", "").replace(")[T.", " = ").rstrip("]") for i in L.index], fontsize=6.5)
    ax.invert_yaxis(); grid(ax); ax.set_xlim(-clip - 0.5, clip + 0.5)
    ax.set_xlabel("log-odds of getting a label (95 % Wald CI); refs: azide, skeleton A, X = Ac, direct, R1")
    if far.any() or sep.any():
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("B2 — what predicts a missing label (logit, computed jobs)")
    save(fig, "fig09_b2_missing_logit.png")


# ---------------------------------------------------------------- B3 (σ table)
@figure
def fig10_b3_hammett():
    need("hammett.csv")
    Hm = csv("hammett.csv"); Sv = csv("sigma_vs_rank2.csv")
    chs = list(dict.fromkeys(Hm.ch))
    combos = [(s, p) for s in ("dipole", "dipolarophile") for p in ("p", "m") if len(Hm[(Hm.side == s) & (Hm.pos == p)])]
    fig, axs = plt.subplots(1, len(combos) + 1, figsize=(3.3 * (len(combos) + 1), 3.9), squeeze=False)
    for a, (s, p) in zip(axs[0], combos):
        g = Hm[(Hm.side == s) & (Hm.pos == p)]
        for k, (col, nm) in enumerate((("R2_sigma", "σ alone"), ("R2_FR", "F + R"))):
            data = [g.loc[g.ch == c, col].dropna().to_numpy() for c in chs]
            pos = np.arange(len(chs)) + (k - 0.5) * 0.35
            keep = [i for i, dd in enumerate(data) if len(dd)]
            if keep:
                a.boxplot([data[i] for i in keep], positions=pos[keep], widths=0.3, patch_artist=True,
                          boxprops=dict(facecolor=S[k], alpha=0.6), medianprops=dict(color=T1), showfliers=False)
            a.plot([], [], color=S[k], lw=6, alpha=0.6, label=nm)
        a.set_xticks(range(len(chs))); a.set_xticklabels(chs, rotation=45, fontsize=7); a.set_ylim(0, 1); grid(a, "y")
        a.set_title(f"{s}, {p}-aryl (vs Ph)"); a.legend(frameon=False, fontsize=7)
        a.set_ylabel("in-sample R² per aryl context")
    a = axs[0][-1]
    if Sv is not None and len(Sv):
        x = np.arange(len(Sv))
        a.bar(x - 0.2, Sv.can_r1, 0.4, color=S[0], label="canonical r₁", zorder=3)
        a.bar(x + 0.2, Sv.can_r2, 0.4, color=S[1], label="canonical r₂", zorder=3)
        a.set_xticks(x); a.set_xticklabels([str(c) for c in Sv.ch], rotation=60, fontsize=6.5); a.set_ylim(0, 1); grid(a, "y")
        sides = list(Sv.side)
        for i in range(1, len(sides)):
            if sides[i] != sides[i - 1]:
                a.axvline(i - 0.5, color=REF, lw=1)
        for sd in dict.fromkeys(sides):
            xs = [i for i, s_ in enumerate(sides) if s_ == sd]
            a.text(np.mean(xs), 1.02, sd, ha="center", fontsize=7.5, color=T2, transform=a.get_xaxis_transform())
        a.legend(frameon=False, fontsize=7, loc="lower right"); a.set_title("data rank-2 X vectors vs (F, R)", pad=14)
    else:
        a.text(0.5, 0.5, "sigma_vs_rank2.csv empty", ha="center", transform=a.transAxes, color=T2); a.set_axis_off()
    fig.suptitle("B3 — literature σ (Hansch, Leo & Taft 1991): R² per channel and Swain–Lupton canonical correlation",
                 fontsize=10, fontweight="bold")
    save(fig, "fig10_b3_hammett.png")


# ---------------------------------------------------------------- B4
@figure
def fig11_b4_robustness():
    need("robustness_headline.csv")
    Rb = csv("robustness_headline.csv")
    var = [c for c in ["incl_stepwise", "alt_same", "lowest_sibling", "no_carbonyl_ylide"] if c in Rb]
    frac = Rb.metric.str.startswith(("regio_acc", "regio_a_pref", "exo_"))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, max(4.0, 0.3 * max(frac.sum(), (~frac).sum()) + 1.2)))
    for a, sub, ref, ttl in ((axs[0], Rb[~frac], 0.0, "correlations and covariance shares"),
                             (axs[1], Rb[frac], 0.5, "accuracies and preference fractions")):
        sub = sub.reset_index(drop=True); y = np.arange(len(sub))
        if "ci_lo" in sub:
            ivl(a, y, sub.ci_lo, sub.ci_hi, True, T1)
        a.plot(sub.base, y, "o", color=T1, ms=5, zorder=4, label="base (core-bootstrap 95 % CI)")
        for k, v in enumerate(var):
            vals = sub[v]
            if v in ("alt_same", "lowest_sibling"):                    # not recomputed for selectivity rows
                vals = vals.where(~sub.metric.str.startswith(("regio_", "exo_")))
            a.plot(vals, y + 0.12 * (k + 1), "|", ms=9, mew=2, color=(S + ["#8a5cc2"])[k % 5], zorder=3, label=v)
        flip = sub.get("sign_flip_variants")
        for i, f in enumerate(flip if flip is not None else []):
            if isinstance(f, str) and f:
                a.text(np.nanmax(sub.loc[i, ["base"] + var].to_numpy(float)), i, f"  sign flip: {f}",
                       fontsize=6.5, color=S[1], va="center")
        a.axvline(ref, color=REF, lw=1, ls="--" if ref else "-")
        a.set_yticks(y); a.set_yticklabels(sub.metric, fontsize=7); a.invert_yaxis(); grid(a)
        a.set_title(ttl); a.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("B4 — do the headline numbers survive the robustness variants? (gate B: sign flips are listed, not STOPs)",
                 fontsize=10, fontweight="bold")
    save(fig, "fig11_b4_robustness.png")


@figure
def fig12_b4_fingerprint_fdr():
    need("fingerprint_signtest_bh.csv")
    F = csv("fingerprint_signtest_bh.csv")
    chs = [c for c in ["d1", "d2", "elst", "Pauli", "OI", "disp", "CPCM", "CDS", "ΔE‡"] if c in set(F.channel)]
    sides = [s for s in ("dipolarophile", "dipole") if s in set(F.side)]
    fig, axs = plt.subplots(1, len(sides), figsize=(4.9 * len(sides) + 0.8, 5.6), squeeze=False)
    im = None
    for a, s in zip(axs[0], sides):
        g = F[F.side == s]
        xs = [x for x in X_ORDER if x in set(g.X)] + sorted(set(g.X) - set(X_ORDER))
        Z = np.full((len(xs), len(chs)), np.nan); sig = np.zeros_like(Z, bool)
        for r in g.itertuples():
            if r.channel in chs:
                i, j = xs.index(r.X), chs.index(r.channel)
                Z[i, j] = np.sign(r.median) * min(-np.log10(max(r.q_bh, 1e-6)), 3) / 3
                sig[i, j] = str(r.sig_fdr05).lower() in ("true", "1")
        im = a.imshow(Z, aspect="auto", cmap=DIV, vmin=-1, vmax=1)
        for i in range(len(xs)):
            for j in range(len(chs)):
                if sig[i, j]:
                    a.text(j, i, "•", ha="center", va="center", fontsize=9, color="white" if abs(Z[i, j]) > 0.6 else T1)
        a.set_xticks(range(len(chs))); a.set_xticklabels(chs, fontsize=8)
        a.set_yticks(range(len(xs))); a.set_yticklabels(xs, fontsize=7)
        a.set_title(f"{s} side"); a.tick_params(length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
    if im is not None:
        q05 = -np.log10(0.05) / 3
        cb = fig.colorbar(im, ax=axs.ravel().tolist(), ticks=[-1, -q05, 0, q05, 1], fraction=0.03)
        cb.ax.set_yticklabels(["lowers, q ≤ 1e-3", "lowers, q = 0.05", "q = 1", "raises, q = 0.05", "raises, q ≤ 1e-3"], fontsize=7)
    fig.suptitle("B4 — fingerprint sign tests over cores: colour = sign of the median × evidence (−log10 q, BH), • = q < 0.05 "
                 "(colour is NOT effect size)", fontsize=9.5, fontweight="bold")
    save(fig, "fig12_b4_fingerprint_fdr.png")


# ---------------------------------------------------------------- B5
@figure
def fig13_b5_houk_domingo():
    need("q2_azide_table.csv")
    Tq = csv("q2_azide_table.csv")
    XO = ["NMe2", "NH2", "OMe", "Me", "CO2Me", "CN", "NO2"]
    keys = [k for k in [("azide", "A"), ("azide", "B")] if len(Tq[(Tq.dipole == k[0]) & (Tq.skel == k[1])])]
    keys += [k for k in sorted(set(zip(Tq.dipole, Tq.skel))) if k not in keys]
    nc = 4; nr = int(np.ceil(len(keys) / nc)) + 1
    fig = plt.figure(figsize=(12.5, 2.8 * nr))
    for p, (dp, sk) in enumerate(keys):
        a = fig.add_subplot(nr, nc, p + 1)
        g = Tq[(Tq.dipole == dp) & (Tq.skel == sk)].set_index("X")
        xs = [x for x in XO if x in g.index]
        for i, x in enumerate(xs):
            a.plot([i, i], [g.loc[x, "ddE"], g.loc[x, "dd_nosolv"]], color=T2, lw=1, zorder=2)
        a.plot(range(len(xs)), g.loc[xs, "ddE"], "o", color=S[0], label="ΔΔE‡ (water SMD)", zorder=3)
        a.plot(range(len(xs)), g.loc[xs, "dd_nosolv"], "s", color=S[1], label="ΔΔE‡ − ΔΔ(CPCM+CDS)", zorder=3)
        a.axhline(0, color=REF, lw=1)
        nd = sum(x in ("NMe2", "NH2") for x in xs); na = sum(x in ("CN", "NO2") for x in xs)
        for b_ in (nd - 0.5, len(xs) - na - 0.5):
            if 0 < b_ < len(xs) - 1:
                a.axvline(b_, color=GRID, lw=1)
        a.set_xticks(range(len(xs))); a.set_xticklabels(xs, fontsize=7, rotation=30); grid(a, "y")
        a.set_title(f"{'HN3' if dp == 'azide' else dp} + {SKN.get(sk, sk)}", fontsize=8.5)
        if p == 0:
            a.set_ylabel("ΔΔ vs H (kcal/mol)"); a.legend(frameon=False, fontsize=6.5)
    a = fig.add_subplot(nr, 1, nr)
    rows = []
    for (dp, sk), g in Tq.groupby(["dipole", "skel"]):
        don, acc = g[g.X.isin(["NMe2", "NH2"])], g[g.X.isin(["NO2", "CN"])]
        rows.append(dict(k=f"{dp} {sk}", d=don.ddE.mean(), a=acc.ddE.mean(), dn=don.dd_nosolv.mean(), an=acc.dd_nosolv.mean()))
    A = pd.DataFrame(rows); x = np.arange(len(A)); w = 0.2
    for k, (c, nm, col, hat) in enumerate((("d", "donors, water", S[0], None), ("a", "acceptors, water", S[1], None),
                                           ("dn", "donors, − solv", S[0], "//"), ("an", "acceptors, − solv", S[1], "//"))):
        a.bar(x + (k - 1.5) * w, A[c], w, color=col, alpha=1 if hat is None else 0.45, hatch=hat, label=nm, zorder=3)
    a.axhline(0, color=REF, lw=1); a.set_xticks(x); a.set_xticklabels(A.k, fontsize=7, rotation=20); grid(a, "y")
    a.set_ylabel("mean ΔΔE‡ (kcal/mol)"); a.legend(frameon=False, fontsize=7, ncol=4)
    a.set_title("Donor (NMe2, NH2) vs acceptor (NO2, CN) lowering, with and without the solvation channels")
    fig.suptitle("B5 — Houk (ambiphilic) vs Domingo (donors only): water vs solvation-stripped decomposition "
                 "(HN3 proxy · B3LYP · SMD water; q2 table: lowest ΔΔ per X)", fontsize=10, fontweight="bold")
    save(fig, "fig13_b5_houk_domingo.png")


@figure
def fig14_b5_regio_magnitude():
    need("sel_regio.csv")
    R = csv("sel_regio.csv")
    R = R[(R.series == "R2") & (R.attach == "direct")]
    XO = ["NMe2", "NH2", "OMe", "OH", "NO2", "CN", "CO2Me", "CHO", "Ac", "CF3"]
    M = R.assign(a=R.barrier_kcal.abs()).pivot_table(index="X", columns="dipole", values="a", aggfunc="median").reindex(XO)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, d in enumerate(M.columns):
        ax.plot(range(len(XO)), M[d], "o-", color=DC.get(d, S[k % 4]), lw=1.4, ms=4, label=d, zorder=3)
    ax.axvline(3.5, color=GRID, lw=1)
    ax.set_xticks(range(len(XO))); ax.set_xticklabels(XO, rotation=40, fontsize=7.5); grid(ax, "y")
    ax.set_ylabel("median |ΔE‡(a − b)| (kcal/mol)"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("B5 — regioselectivity magnitude, dipolarophile X (donors | acceptors)")
    save(fig, "fig14_b5_regio_magnitude.png")


@figure
def fig15_b5_gedt():
    need("gedt_partial.csv", "gedt_d1.csv", "pairs_H.csv")
    G = csv("gedt_partial.csv"); g1 = csv("gedt_d1.csv").set_index("job_id").gedt_mulliken
    P = csv("pairs_H.csv"); P = P[~boolcol(P.q_bad)].copy()
    P["absd"] = (P.job_id.map(g1) - P.parent_job.map(g1)).abs()
    P["strain"] = P.d1_kcal + P.d2_kcal
    sl = csv("gedt_sign_slopes.csv")
    fig = plt.figure(figsize=(13, 7.6))
    dips = [d for d in DIPS if d in set(P.dipole)]
    for j, d in enumerate(dips):
        for i, (col, xl) in enumerate((("absd", "|ΔGEDT| (e, Mulliken)"), ("strain", "ΔΔ strain (kcal/mol)"))):
            a = fig.add_subplot(3, len(dips), i * len(dips) + j + 1)
            g = P[(P.dipole == d)].dropna(subset=[col])
            a.scatter(g[col], g.barrier_kcal, s=3, alpha=0.35, color=DC[d], linewidths=0)
            rr = np.corrcoef(g[col], g.barrier_kcal)[0, 1] if len(g) > 2 else np.nan
            a.set_title(f"{d}: r = {rr:+.2f}", fontsize=8); a.axhline(0, color=REF, lw=0.6)
            a.set_xlabel(xl, fontsize=7)
            if j == 0:
                a.set_ylabel("ΔΔE‡ (kcal/mol)", fontsize=8)
    a = fig.add_subplot(3, 2, 5)
    q = G[(G.charges == "mulliken") & (G.gedt_term == "|ΔGEDT|") & (G.side == "all") & (G.dipole != "all")]
    for k, (yb, mk) in enumerate((("ΔΔE‡", "o"), ("ΔΔE‡ − ΔΔ(CPCM+CDS)", "s"))):
        qq = q[q.barrier == yb].set_index("dipole").reindex(dips)
        if not qq["pr_y_g__s"].notna().any():
            continue
        for m, (col, c) in enumerate((("pr_y_g__s", S[0]), ("pr_y_s__g", S[1]))):
            lohi = [ci(s) for s in (qq[f"{col}_ci"] if f"{col}_ci" in qq else [None] * len(qq))]
            x = np.arange(len(dips)) + (2 * k + m - 1.5) * 0.17
            ivl(a, x, [l for l, _ in lohi], [h for _, h in lohi], False, c)
            a.plot(x, qq[col], mk, color=c, mfc=c if k == 0 else "white", zorder=3,
                   label=f"{'GEDT | strain' if m == 0 else 'strain | GEDT'}, {'water' if k == 0 else '− solv'}")
    a.axhline(0, color=REF, lw=1); a.set_xticks(range(len(dips))); a.set_xticklabels(dips, fontsize=7, rotation=15)
    grid(a, "y"); a.set_ylabel("partial r with ΔΔE‡"); a.legend(frameon=False, fontsize=6.5, ncol=2)
    a.set_title("Does |ΔGEDT| explain the barrier beyond strain? (partial r, core-bootstrap CI)", fontsize=9)
    a = fig.add_subplot(3, 2, 6)
    if sl is not None and len(sl):
        s2 = sl[(sl.charges == "mulliken") & (sl.barrier == "y_all")].set_index("dipole")
        ks = [k for k in ["all"] + dips if k in s2.index]
        x = np.arange(len(ks))
        for m, (lab, c) in enumerate((("FEDF", S[0]), ("REDF", S[1]))):
            if f"slope_{lab}" not in s2:
                continue
            xb = x + (m - 0.5) * 0.38
            yv = s2.loc[ks, f"slope_{lab}"].to_numpy(float)
            a.bar(xb, yv, 0.38, color=c, alpha=0.8, zorder=3,
                  label=f"{lab} ({'dipole → dipolarophile' if lab == 'FEDF' else 'dipolarophile → dipole'})")
            if f"slope_{lab}_ci" in s2:
                lohi = [ci(v) for v in s2.loc[ks, f"slope_{lab}_ci"]]
                ivl(a, xb, [l for l, _ in lohi], [h for _, h in lohi], False, T1)
            if f"n_{lab}" in s2:
                for xx, yy, nn in zip(xb, yv, s2.loc[ks, f"n_{lab}"].to_numpy(float)):
                    if np.isfinite(yy) and np.isfinite(nn):
                        a.text(xx, 0, f"n{int(nn)}", fontsize=5.5, ha="center", va="bottom" if yy < 0 else "top", color=T2)
        a.axhline(0, color=REF, lw=1); a.set_xticks(x); a.set_xticklabels(ks, fontsize=7, rotation=15); grid(a, "y")
        a.set_ylabel("slope ΔΔE‡ ~ Δ|GEDT| (kcal/mol per e)"); a.legend(frameon=False, fontsize=6.5)
        a.set_title("Sign of electron flow: slope asymmetry (95 % CI)", fontsize=9)
    fig.suptitle("B5 — GEDT (Domingo's key variable) vs strain; DFT Mulliken charges at the real TS (Domingo uses NPA, gas phase)",
                 fontsize=10, fontweight="bold")
    save(fig, "fig15_b5_gedt.png")


@figure
def fig16_b5_sigma_asymmetry():
    need("inflection/asymmetry_slopes.csv")
    A = csv("inflection/asymmetry_slopes.csv")
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    key = ["dipole", "skeleton", "side", "pos"]
    a0 = A[A.target == "barrier_kcal"].set_index(key)
    a1 = A[A.target == "nosolv"].set_index(key)
    PM = {"direct": "o", "p": "^", "m": "v"}
    for k in a0.index:
        c = DC.get(k[0], T2); mk = PM.get(k[3], "o")
        x0, y0 = a0.loc[k, "slope_donor"], a0.loc[k, "slope_acceptor"]
        ax.plot(x0, y0, mk, ms=4, color=c, alpha=0.85, zorder=3)
        if k in a1.index:
            x1, y1 = a1.loc[k, "slope_donor"], a1.loc[k, "slope_acceptor"]
            ax.plot(x1, y1, mk, ms=4, mfc="none", color=c, alpha=0.85, zorder=3)     # keeps the heads inside the limits
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.6, alpha=0.6))
        if k[0] == "azide" and k[1] in ("A", "B") and k[2] == "dipolarophile" and k[3] == "direct":
            ax.text(x0, y0, f"  HN3 + {SKN.get(k[1], k[1])}", fontsize=7)
    ax.axhline(0, color=REF, lw=1); ax.axvline(0, color=REF, lw=1)
    ax.text(0.98, 0.02, "Houk / ambiphilic: donor > 0, acceptor < 0", transform=ax.transAxes, ha="right", fontsize=7, color=T2)
    ax.text(0.98, 0.97, "Domingo: donor > 0, acceptor ≥ 0", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=T2)
    for d in DIPS:
        ax.plot([], [], "o", color=DC[d], label=d)
    for p_, m_ in PM.items():
        ax.plot([], [], m_, color=T2, label=f"{p_} (vs {'H' if p_ == 'direct' else 'Ph'})")
    ax.plot([], [], "o", mfc="none", color=T2, label="solvation stripped (arrow head)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_xlabel("donor-side slope, σ < 0 (kcal/mol per σ; > 0 = donors lower the barrier)")
    ax.set_ylabel("acceptor-side slope, σ > 0 (kcal/mol per σ; < 0 = acceptors lower it)")
    ax.set_title("B5 — σ asymmetry: water (filled) → solvation stripped (open)")
    save(fig, "fig16_b5_sigma_asymmetry.png")


# ---------------------------------------------------------------- B6 (σ table)
@figure
def fig17_b6_inflection():
    need("inflection/inflection_summary.csv", "inflection/inflection_series.csv")
    Su = csv("inflection/inflection_summary.csv", index_col=0)
    Se = csv("inflection/inflection_series.csv")
    order = ["barrier_kcal", "d1_kcal", "d2_kcal", "strain", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft",
             "Eint", "nosolv"]
    order = [o for o in order if o in Su.index]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    x = np.arange(len(order))
    axs[0].bar(x, Su.loc[order, "frac_inflection"], color=S[0], zorder=3)
    nul = float(Su.null_rate_permutation.iloc[0])
    axs[0].axhline(nul, color=T1, lw=1, ls="--")
    axs[0].text(len(order) - 0.5, nul, f"permutation null {nul:.3f}", ha="right", va="bottom", fontsize=7)
    axs[0].set_xticks(x); axs[0].set_xticklabels([CHN.get(o, o) for o in order], rotation=45, fontsize=7); grid(axs[0], "y")
    axs[0].set_ylabel("fraction of series with an inflection"); axs[0].set_title(f"Inflection rate ({int(Su.n_series.iloc[0])} series)")
    hid = [o for o in order if o != "barrier_kcal"]
    axs[1].bar(range(len(hid)), Su.loc[hid, "hidden_frac_vs_barrier"], color=S[1], zorder=3)
    axs[1].set_xticks(range(len(hid))); axs[1].set_xticklabels([CHN.get(o, o) for o in hid], rotation=45, fontsize=7)
    grid(axs[1], "y"); axs[1].set_ylabel("channel bends, barrier linear (fraction)"); axs[1].set_title("Hidden inflections")
    prim = ["barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
    flag = Se[boolcol(Se.inflection) & Se.target.isin(prim)]
    one_sided = ((flag.dipole == "carbonyl ylide") & (flag.side == "dipolarophile")).to_numpy()
    bp = flag.bp.to_numpy(float)
    if np.isfinite(bp).any():
        e = np.histogram_bin_edges(bp[np.isfinite(bp)], 20)
        axs[2].hist([bp[~one_sided], bp[one_sided]], bins=e, stacked=True, color=[S[2], "#bdbcb6"], zorder=3,
                    label=["flagged (series × channel; ΔE‡ + 8 channels)", "carbonyl ylide R2 (one-sided, B8: not interpreted)"])
        axs[2].legend(frameon=False, fontsize=6.5)
    axs[2].axvline(0, color=T1, lw=1, ls="--"); axs[2].set_xlabel("breakpoint σ"); grid(axs[2], "y")
    axs[2].set_title("Where do channels bend? (σ ≈ 0 = flow reversal)")
    fig.suptitle("B6 — channel inflections along EDG → EWG (literature σ): observed vs permutation null",
                 fontsize=10, fontweight="bold")
    save(fig, "fig17_b6_inflection.png")


# ---------------------------------------------------------------- B7
@figure
def fig18_b7_methyl():
    need("q5_methyl.csv")
    M = csv("q5_methyl.csv")
    comps = list(dict.fromkeys(M.comp))
    dips = [d for d in DIPS if d in set(M.dipole)]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(comps) + 2.6))
    seen = set()
    for a, (col, ttl) in zip(axs, (("barrier_kcal", "ΔE‡ (s1 − s0 skeleton, same dipole and R1)"),
                                  ("strain", "strain (filled) and E_int (open)"))):
        for k, d in enumerate(dips):
            g = M[M.dipole == d].set_index("comp")
            for i, c in enumerate(comps):
                if c not in g.index:
                    continue
                r = g.loc[c]; y = i + (k - 2) * 0.13
                if not np.isfinite(r.get(col, np.nan)):
                    a.text(0, y, "n/a", fontsize=5.5, color=T2, va="center"); continue
                ivl(a, y, r[f"{col}_lo"], r[f"{col}_hi"], True, DC[d], lw=0.9, cap=1.5)
                lab = d if (col == "barrier_kcal" and d not in seen) else None
                if lab:
                    seen.add(d)
                a.plot(r[col], y, "o", ms=3.5, color=DC[d], zorder=3, label=lab)
                if col == "strain" and np.isfinite(r.get("Eint", np.nan)):
                    ivl(a, y, r.Eint_lo, r.Eint_hi, True, DC[d], lw=0.9, cap=1.5)
                    a.plot(r.Eint, y, "o", ms=3.5, mfc="white", color=DC[d], zorder=3)
        a.axvline(0, color=REF, lw=1); a.set_yticks(range(len(comps))); a.set_yticklabels(comps, fontsize=7)
        a.invert_yaxis(); grid(a); a.set_xlabel("median over common R1 (kcal/mol; bars 95 % bootstrap CI)"); a.set_title(ttl)
    axs[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("B7 — methyl patterns and cis/trans: ΔE‡, strain and E_int (median, 95 % bootstrap CI; n < 8: n/a)",
                 fontsize=10, fontweight="bold")
    save(fig, "fig18_b7_methyl.png")


# ---------------------------------------------------------------- B8
@figure
def fig19_b8_carbonyl_ylide():
    need("q4_sensitivity.csv")
    Q = csv("q4_sensitivity.csv", index_col=0)
    T = csv("q4_missing_by_dipole.csv", index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    for a, keys, ttl in ((axs[0], ["r_strain", "r_Eint", "regio_acc_strain", "regio_acc_Eint", "exo_frac"], "r and accuracies"),
                         (axs[1], ["sh_Pauli", "sh_elst", "sh_OI", "sh_strain"], "covariance shares")):
        keys = [k for k in keys if k in Q.index]; y = np.arange(len(keys))
        for i, k in enumerate(keys):
            a.plot([Q.loc[k].iloc[0], Q.loc[k].iloc[1]], [i, i], color=T2, lw=1, zorder=2)
        a.plot(Q.loc[keys].iloc[:, 0], y, "o", color=S[0], label="with carbonyl ylide", zorder=3)
        a.plot(Q.loc[keys].iloc[:, 1], y, "s", color=S[1], label="without", zorder=3)
        a.axvline(0, color=REF, lw=1); a.set_yticks(y); a.set_yticklabels(keys, fontsize=7.5); a.invert_yaxis(); grid(a)
        a.set_title(ttl); a.legend(frameon=False, fontsize=7)
    n = {k: Q.loc[k].tolist() for k in ("n_pairs", "n_regio", "n_exo") if k in Q.index}
    axs[0].text(0.0, -0.22, "  ".join(f"{k}: {int(v[0])} → {int(v[1])}" for k, v in n.items() if all(np.isfinite(v))),
                transform=axs[0].transAxes, fontsize=7, color=T2)
    if T is not None:
        x = np.arange(len(T))
        axs[2].bar(x - 0.2, T.present_frac, 0.4, color=S[2], label="present / designed", zorder=3)
        axs[2].bar(x + 0.2, T.neg_barrier_frac, 0.4, color=S[3], label="negative absolute barrier (clean)", zorder=3)
        axs[2].set_xticks(x); axs[2].set_xticklabels(T.index, rotation=25, fontsize=7); axs[2].set_ylim(0, 1); grid(axs[2], "y")
        axs[2].legend(frameon=False, fontsize=6.5); axs[2].set_title("presence and negative barriers by dipole")
    fig.suptitle("B8 — carbonyl ylide kept: headline with vs without it (missingness reported as MNAR)",
                 fontsize=10, fontweight="bold")
    save(fig, "fig19_b8_carbonyl_ylide.png")


# ---------------------------------------------------------------- run
for fn in FIGS:
    (FD / f"{fn.__name__}.png").unlink(missing_ok=True)       # no stale figure survives a skip or a failure
    try:
        fn()
    except FileNotFoundError as e:
        index["skipped"][fn.__name__] = f"input missing or empty: {e}"
        plt.close("all")
    except Exception as e:                                               # noqa: BLE001
        index["failed"][fn.__name__] = f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-600:]}"
        plt.close("all")
with open(FD / "figs_b_index.json", "w") as f:
    json.dump(index, f, indent=1, ensure_ascii=False)
print(json.dumps(index, indent=1, ensure_ascii=False))
sys.exit(1 if index["failed"] else 0)
