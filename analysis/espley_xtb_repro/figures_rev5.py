#!/usr/bin/env python3
"""figures_rev5.py — rev 5 Phase E: the paper figures fig1..fig8 of docs/specs/REV5_FEATURES_FIGURES.md (Phase E), run
by r5_figures.sh inside a SLURM allocation (refuses to run outside SLURM).

  python figures_rev5.py [--only fig1,fig7] [--force]

Per figure, in results_rev5/figures/ (each written atomically): <name>.pdf (vector, TrueType fonts), <name>.png
(300 dpi) and <name>.csv (the plotted values, same basename). results_rev5/figures/figures_rev5_manifest.json
records per figure the input files + sha256, the status, notes and an automatic text-overlap / clipping check of the
drawn figure (an aid for the visual check of the PNGs the spec asks for, not a replacement).

  figure                content                                                      inputs (results_rev5/)
  fig1_espley_mae       5 targets x 4 series: mean test MAE, SE over the 5 seeds, the     D3a_per_seed.csv,
                        seed MAEs as dots, value labels; per target Espley ÷ EXT_SEL      D3a_summary.csv,
                        (mean of the per-seed ratios) and the Nadeau–Bengio p of          D3a_tests.csv
                        EXT_SEL vs Espley SVR
  fig2_espley_lockbox   lockbox head-to-head, Espley SVR (their protocol) vs EXT_SEL KRR:  D3b_summary.csv,
                        MAE with the bootstrap 95 % CI; paired difference with its CI      D3b_paired.csv,
                                                                                           D3b_predictions.csv
  fig3_parity           2 rows (Espley / EXT_SEL) x 5 targets: DFT vs the prediction       D3a_predictions.csv
                        averaged over the seeds in which the reaction was a test row;      (+ D3a_per_seed.csv)
                        y = x, ±1 / ±2 kcal/mol bands, axes shared per target; MAE and r²
                        of the plotted points; vector PDF; interaction in Espley's sign
                        (positive = stabilising, compare_espley.SIGN), flagged in title,
                        note and CSV (sign_convention)
  fig4_error_ecdf       |error| ECDF per target, 4 series (test rows of every seed         D3a_predictions.csv
                        pooled); lines at 1 and 2 kcal/mol; fraction within 1 kcal/mol     (+ D3a_per_seed.csv)
                        labelled directly
  fig5_seed_pairs       per target the 5 seeds' paired lines Espley -> EXT_SEL; p in the   D3a_per_seed.csv,
                        title                                                              D3a_tests.csv
  fig6_learning_curves  test MAE vs training rows (mean ± SE over seeds), Espley SVR and   D4_learning_curves.csv
                        EXT_SEL KRR; the n_train at which EXT_SEL reaches Espley's         + its _summary.csv and
                        full-data MAE (= Espley SVR on all curve training rows, D-4 frac   _reach.csv
                        1.0; linear interpolation), or "not reached". The D-3(a)
                        reference row of _reach.csv is not drawn
  fig7_channel_blocks   elst / Pauli / OI dev-CV NMAE of BASE, BASE + each block, EXT_SEL,  C2_block_cv_folds.csv,
                        EXT_ALL, with the outer-fold points                                C2_block_cv.csv
  fig8_channel_lockbox  9 targets, lockbox MAE of BASE vs EXT_SEL (KRR) with bootstrap      D1_lockbox.csv,
                        95 % CIs, and the paired difference EXT_SEL − BASE with its CI      D1_lockbox_paired.csv

Every number drawn is read from those files (spec §1). Each one that a finer-grained file determines is recomputed from
it and must agree, else the figure is not drawn: seed means / SE / ratios / per-seed differences from the per-seed rows
(1e-9), per-seed MAE from the per-row predictions (1e-5: espley_rev5.py rounds those to 6 decimals), learning-curve
means / SE from the per-seed curve rows, the reach point from the learning-curve summary, D-1 / D-3(b) paired
differences from their MAEs; test rows identical across series and the same y for every series.

Series (espley_rev5.py `series` / `side` columns = the rev5_common.COLORS keys): Espley = Espley's SVR (D-3(a): stored
test predictions for interaction / ΔE‡ / ΔG‡, their-protocol retrain for the role distortions; D-3(b): their protocol on
dev ∩ Espley rows; D-4: their SVR with fixed hyperparameters on the same subsets as EXT_SEL); ESPLEY46 / ESPLEY73 /
EXT_SEL = our KRR (RBF) on that arm. BASE = ESPLEY73 (colour of ESPLEY73).

Style (spec Phase E, constants from rev5_common): colours fixed per entity (COLORS; greys for the fig7 bars that are no
entity: BASE + one block, EXT_ALL); widths FIG_W1 / FIG_W2 (every figure here has 5+ panels or groups: FIG_W2);
Arial if installed, else DejaVu Sans, FONT_PT everywhere; no twin y-axes; a legend on every figure (all have >= 2
series); text in black / grey ink only; light grid; a 2-px gap between bars (each bar narrowed about its centre by
2 px at 300 dpi once the layout is fixed); value labels on bars.
Idempotent: a figure whose three outputs exist and whose manifest signature (input sha256, sha256 of the code files
CODE_FILES = this file, rev5_common.py, compare_espley.py, aggregate_ml.py; font, matplotlib version) is unchanged is
not redrawn (--force redraws). A figure whose input is missing or inconsistent is skipped with a message and the run
exits 1 after the other figures; 0 = every requested figure drawn or up to date.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import textwrap
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev5_common as R5  # noqa: E402
import compare_espley as CE  # noqa: E402
import aggregate_ml as AM  # noqa: E402

TARGET_LABEL = AM.TARGET_LABEL
# every module whose definitions reach a figure (targets, their order, labels, constants): hashed into each figure's
# idempotency signature, so a change in any of them redraws
CODE_FILES = (Path(__file__).resolve(),) + tuple(Path(m.__file__).resolve() for m in (R5, CE, AM))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

# ---------------------------------------------------------------- inputs (file, the stage that writes it)
RES = R5.RES5
FIG_DIR = RES / "figures"
MANIFEST = FIG_DIR / "figures_rev5_manifest.json"
_D3A = "D-3(a): espley_rev5.py d3a (r5_espley.sh)"
_D3B = "D-3(b): espley_rev5.py d3b (r5_espley.sh)"
_D4 = "D-4: espley_rev5.py d4 (r5_espley.sh)"
FILES = {
    "d3a_per_seed": (RES / "D3a_per_seed.csv", _D3A),
    "d3a_summary": (RES / "D3a_summary.csv", _D3A),
    "d3a_tests": (RES / "D3a_tests.csv", _D3A),
    "d3a_preds": (RES / "D3a_predictions.csv", _D3A),
    "d3b_summary": (RES / "D3b_summary.csv", _D3B),
    "d3b_paired": (RES / "D3b_paired.csv", _D3B),
    "d3b_preds": (RES / "D3b_predictions.csv", _D3B),
    "d4": (RES / "D4_learning_curves.csv", _D4),
    "d4_summary": (RES / "D4_learning_curves_summary.csv", _D4),
    "d4_reach": (RES / "D4_learning_curves_reach.csv", _D4),
    "c2_folds": (RES / "C2_block_cv_folds.csv", "C-2: select_blocks.py select (r5_select.sh)"),
    "c2_cv": (RES / "C2_block_cv.csv", "C-2: select_blocks.py select (r5_select.sh)"),
    "d1": (RES / "D1_lockbox.csv", "D-1: final_eval.py (r5_lockbox_eval.sh)"),
    "d1_paired": (RES / "D1_lockbox_paired.csv", "D-1: final_eval.py (r5_lockbox_eval.sh)"),
}

# ---------------------------------------------------------------- design (display only; every number comes from files)
ENTS = ("Espley", "ESPLEY46", "ESPLEY73", "EXT_SEL")          # plotting order; = rev5_common.COLORS keys
assert set(ENTS) == set(R5.COLORS), R5.COLORS
ELAB = {"Espley": "Espley et al. (SVR)", "ESPLEY46": "Ours ESPLEY46 (KRR)", "ESPLEY73": "Ours ESPLEY73 (KRR)",
        "EXT_SEL": "Ours EXT_SEL (KRR)"}
RAW3 = [t for t in CE.TARGETS if not t.startswith("distortion_energy_")]   # interaction, ΔE‡, ΔG‡ (Espley's columns)
T5 = list(CE.ROLE_TARGETS) + RAW3                                          # = espley_rev5.TARGETS5
assert len(T5) == 5 and T5 == list(getattr(CE, "MAIN_TARGETS", T5)), T5
TLAB = {"dipole_distortion_role": "Dipole distortion", "dipolarophile_distortion_role": "Dipolarophile distortion",
        "interaction_energies_dft": "Interaction", "e_barrier_dft": "ΔE‡", "q_barrier_dft": "ΔG‡"}
assert set(TLAB) == set(T5), T5
TLAB2 = {t: TLAB[t].replace(" ", "\n", 1) for t in T5}                     # two-line panel / tick labels
# signed values (fig3): targets stored by Espley with the opposite sign of ours (compare_espley.SIGN); espley_rev5.py
# trains and scores every side on Espley's numbers, so y / yhat of these targets carry Espley's sign
SIGN_CONV = {"interaction_energies_dft": "Espley: positive = stabilising (= −E_int)"}
assert set(SIGN_CONV) == set(CE.SIGN) and set(SIGN_CONV) <= set(T5), CE.SIGN
SEEDS = [int(s) for s in R5.PROTO_A_SEEDS]                                 # Espley's seeds 22/23/14/1/2
HEAD_MODEL = "KRR_rbf"                                                     # D-1 headline model (final_eval.py)
ARMS7 = ["BASE"] + [f"BASE+{b}" for b in R5.BLOCK_ORDER] + ["EXT_SEL", "EXT_ALL"]   # fig7 bars (C-2 arm names)

# ---------------------------------------------------------------- style
DPI = 300                        # PNG resolution = figure dpi, so 1 display pixel = 1 PNG pixel
GAP_PX = 2.0                     # gap between adjacent bars
INK, INK2 = "#000000", "#595959" # the only text colours: black, grey
GRID = "#e6e6e6"
BAND1, BAND2 = "#d4d4d4", "#ececec"            # fig3 ±1 / ±2 kcal/mol
GREY_BLOCK, GREY_ALL = "#c8c8c8", "#8f8f8f"    # fig7: BASE + one block, EXT_ALL (not entities)
LINK = "#a6a6a6"                               # fig5 seed lines
EB = dict(fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4)

TOL = 1e-9                       # one number through a CSV round trip (pandas writes repr floats)
TOL_ROUND = 1e-5                 # MAE from per-row predictions rounded to 6 decimals (espley_rev5.py)
TOL_Y = 2e-6                     # the same y seen through two rounded rows


class FigError(Exception):
    """An input of a figure is missing or inconsistent: the figure is not drawn and the run exits 1 at the end."""

    def __init__(self, msg, missing=False):
        super().__init__(msg)
        self.missing = missing


# ---------------------------------------------------------------- small helpers
def die(msg, code=2):
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(p):
    p = Path(p)
    for root, tag in ((R5.HERE, ""), (R5.SCRATCH, "$R5_SCRATCH/")):
        try:
            return tag + str(p.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            continue
    return str(p)


def read(key, need):
    """CSV of FILES[key] with at least the columns `need`; FigError (missing=True) if the file does not exist."""
    path, who = FILES[key]
    if not path.is_file():
        raise FigError(f"{rel(path)} missing (written by {who})", missing=True)
    df = pd.read_csv(path)
    lack = sorted(set(need) - set(df.columns))
    if lack:
        raise FigError(f"{rel(path)}: columns {lack} missing (has {list(df.columns)[:40]})")
    if df.empty:
        raise FigError(f"{rel(path)}: no rows")
    return df


def one(df, mask, what):
    r = df[mask]
    if len(r) != 1:
        raise FigError(f"{what}: {len(r)} rows, expected exactly 1")
    return r.iloc[0]


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > tol:
        raise FigError(f"{what}: {a!r} != {b!r} (tolerance {tol:g})")


def as_bool(v):
    s = str(v).strip().lower()
    if s not in ("true", "false"):
        raise FigError(f"not a boolean: {v!r}")
    return s == "true"


def ints(s, what):
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any() or not np.allclose(v, np.round(v)):
        raise FigError(f"{what}: missing or non-integer values")
    return np.round(v).astype(np.int64)


def read_ids(path):
    path = Path(path)
    if not path.is_file():
        raise FigError(f"{rel(path)} missing", missing=True)
    return set(ints(pd.read_csv(path)["rxn_id"], rel(path)).tolist())


def num(x, nd=2, sign=False):
    """Number for a label, typographic minus."""
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p <= 0:
        return "p < 10$^{-300}$"
    if p >= 1e-2:
        return f"p = {p:.3f}"
    if p >= 1e-3:
        return f"p = {p:.4f}"
    m, e = f"{p:.1e}".split("e")
    return f"p = {m}×10$^{{{int(e)}}}$"


# ---------------------------------------------------------------- figure machinery
def pick_font():
    try:
        font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
        return "Arial"
    except ValueError:
        return "DejaVu Sans"


def style():
    fam = pick_font()
    pt = R5.FONT_PT
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"], "font.size": pt,
        "axes.titlesize": pt, "axes.labelsize": pt, "xtick.labelsize": pt, "ytick.labelsize": pt,
        "legend.fontsize": pt, "figure.titlesize": pt, "figure.labelsize": pt,
        "mathtext.default": "regular",               # p-value exponents ($^{-5}$) in the text font
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "legend.labelcolor": INK,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "axes.grid": False,
        "axes.titlepad": 3, "axes.labelpad": 2, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.0, "patch.linewidth": 0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": DPI, "savefig.dpi": DPI, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.unicode_minus": True,
    })
    return fam


def new_fig(width, height_mm):
    fig = plt.figure(figsize=(width, height_mm * R5.MM), dpi=DPI, layout="constrained")
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.02, hspace=0.02)
    fig._r5_bars, fig._r5_fit = [], []          # bars to narrow by GAP_PX, (axes, anchored labels) to fit
    return fig


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def bar(ax, xc, h, slot, color, **kw):
    """One bar of width `slot` centred on xc (adjacent bars touch); gap_bars() narrows it by GAP_PX after layout."""
    cont = ax.bar(xc, h, width=slot, color=color, linewidth=0, **kw)
    for patch in cont.patches:
        ax.figure._r5_bars.append((ax, patch, float(xc), float(slot)))
    return cont


def vlabel(ax, x, y, text, rot=0, pad=1.5):
    """Value label anchored at data (x, y), `pad` pt above it."""
    return ax.annotate(text, xy=(x, y), xytext=(0, pad), textcoords="offset points", ha="center", va="bottom",
                       rotation=rot, color=INK, annotation_clip=False)


def above(ax, x, y, text, lines_below=1):
    """Group annotation above `lines_below` value-label lines anchored at the same height."""
    return ax.annotate(text, xy=(x, y), xytext=(0, 3 + 1.35 * R5.FONT_PT * lines_below), textcoords="offset points",
                       ha="center", va="bottom", color=INK, annotation_clip=False)


def fit_labels(fig, passes=8, margin_px=3.0):
    """Raise the upper y-limit of each registered axes until its anchored labels end inside it. -> warnings."""
    regs = fig._r5_fit
    for _ in range(passes):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        changed = False
        for ax, arts in regs:
            box = ax.get_window_extent(r)
            y0, y1 = ax.get_ylim()
            H = box.height
            need = y1
            for a in arts:
                top = a.get_window_extent(r).y1
                if top <= box.y1 - margin_px:
                    continue
                ya = float(a.xy[1])
                off = top - ax.transData.transform((0.0, ya))[1] + margin_px
                if off >= 0.8 * H:
                    raise FigError("a label is taller than 80 % of its axes; the figure is too small")
                need = max(need, y0 + (ya - y0) * H / (H - off))
            if need > y1 + 1e-12 * max(1.0, abs(y1)):
                ax.set_ylim(y0, need)
                changed = True
        if not changed:
            return []
    return [f"value labels still extend above an axes after {passes} passes"]


def gap_bars(fig):
    """Narrow every registered bar about its centre so that adjacent bars are GAP_PX apart (at DPI)."""
    fig.canvas.draw()
    for ax, patch, xc, slot in fig._r5_bars:
        px = abs(ax.transData.transform((1.0, 0.0))[0] - ax.transData.transform((0.0, 0.0))[0])
        w = slot - GAP_PX / px
        if w <= 0:
            raise FigError("bars narrower than the 2-px gap")
        patch.set_x(xc - w / 2)
        patch.set_width(w)


def layout_check(fig):
    """Texts outside the figure and pairs of overlapping texts (bounding boxes, > 1 px each way). -> warnings."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    items, seen = [], set()

    def add(t, what):
        if t is None or id(t) in seen or not t.get_visible() or not t.get_text().strip():
            return
        seen.add(id(t))
        bb = t.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            items.append((what, t.get_text().replace("\n", " ")[:40], bb))

    for i, ax in enumerate(fig.axes):
        for t in ax.texts:
            add(t, f"ax{i} text")
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            add(t, f"ax{i} title")
        add(ax.xaxis.label, f"ax{i} xlabel")
        add(ax.yaxis.label, f"ax{i} ylabel")
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks():
                if lo - 1e-9 <= tk.get_loc() <= hi + 1e-9 and tk.label1.get_visible():
                    add(tk.label1, f"ax{i} tick")
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                add(t, f"ax{i} legend")
    for leg in fig.legends:
        for t in leg.get_texts():
            add(t, "legend")
    for t in fig.texts:
        add(t, "figure text")
    warn = [f"outside the figure: {w} '{s}'" for w, s, bb in items
            if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5]
    for a in range(len(items)):
        A = items[a][2]
        for b in range(a + 1, len(items)):
            B = items[b][2]
            if min(A.x1, B.x1) - max(A.x0, B.x0) > 1.0 and min(A.y1, B.y1) - max(A.y0, B.y0) > 1.0:
                warn.append(f"overlap: {items[a][0]} '{items[a][1]}' / {items[b][0]} '{items[b][1]}'")
    return warn


def fig_legend(fig, handles):
    """Legend above the axes in the fewest rows whose estimated width (0.65 em per character, conservative) fits the
    figure; matplotlib fills the columns as np.array_split(entries, ncol), which the estimate mirrors."""
    labels = [h.get_label() for h in handles]
    pt = R5.FONT_PT
    avail = fig.get_figwidth() * 72 * 0.97
    hand = (plt.rcParams["legend.handlelength"] + plt.rcParams["legend.handletextpad"]) * pt
    gap = plt.rcParams["legend.columnspacing"] * pt
    wid = [max(len(x) for x in s.split("\n")) * 0.65 * pt for s in labels]
    ncol = 1
    for nrow in range(1, len(labels) + 1):
        ncol = math.ceil(len(labels) / nrow)
        cols = [c for c in np.array_split(np.arange(len(labels)), ncol) if len(c)]
        if sum(max(wid[i] for i in c) + hand for c in cols) + gap * (len(cols) - 1) <= avail:
            break
    return fig.legend(handles=handles, loc="outside upper center", ncol=ncol)


def note(fig, text):
    """Grey figure note under the axes (layout-managed supxlabel), wrapped to the figure width (same conservative
    0.65 em per character as fig_legend)."""
    width = max(20, int(fig.get_figwidth() * 72 * 0.95 / (0.65 * R5.FONT_PT)))
    lines = [ln for para in text.split("\n") for ln in (textwrap.wrap(para, width) or [""])]
    fig.supxlabel("\n".join(lines), color=INK2, fontsize=R5.FONT_PT)


def save(fig, name, table):
    warn = fit_labels(fig)
    gap_bars(fig)
    warn += layout_check(fig)
    base = FIG_DIR / name
    R5.write_atomic(base.with_suffix(".csv"), lambda t: table.to_csv(t, index=False, float_format="%.6g"))
    R5.write_atomic(base.with_suffix(".pdf"), lambda t: fig.savefig(t, format="pdf", metadata={"CreationDate": None}))
    R5.write_atomic(base.with_suffix(".png"), lambda t: fig.savefig(t, format="png", dpi=DPI))
    return warn


# ---------------------------------------------------------------- inputs: D-3(a)
def build_d3a(ctx):
    """Per-seed MAE of the 4 series (D3a_per_seed.csv), checked against D3a_summary.csv, and the EXT_SEL vs Espley SVR
    tests (D3a_tests.csv), checked against the per-seed rows."""
    dcols = [f"d_seed{s}" for s in SEEDS]
    per = read("d3a_per_seed", {"series", "side", "protocol", "arm", "model", "target", "seed", "mae", "n_test",
                                "n_train"})
    summ = read("d3a_summary", {"series", "target", "mae", "se_seeds", "n_seeds"})
    tests = read("d3a_tests", {"our_arm", "comparator", "target", "esp_protocol", "esp_model", "espley_mae",
                               "ours_mae", "diff_mean", "ci95_lo", "ci95_hi", "t", "p", "J", "n_test", "n_train_ours",
                               "mae_ratio_mean", *dcols})
    per = per[per["series"].isin(ENTS)].copy()
    per["seed"] = ints(per["seed"], "D3a_per_seed.csv seed")
    M, info = {}, {}
    for e in ENTS:
        for t in T5:
            g = per[(per["series"] == e) & (per["target"] == t)]
            if sorted(g["seed"].tolist()) != sorted(SEEDS):
                raise FigError(f"D3a_per_seed.csv: series {e}, {t}: seeds {sorted(g['seed'].tolist())}, expected "
                               f"{sorted(SEEDS)} once each")
            ident = g[["side", "protocol", "arm", "model"]].astype(str).drop_duplicates()
            if len(ident) != 1:
                raise FigError(f"D3a_per_seed.csv: series {e}, {t} spans several models {ident.to_dict('records')}")
            v = g.set_index("seed")["mae"].reindex(SEEDS).to_numpy(float)
            M[(e, t)] = v
            info[(e, t)] = dict(protocol=str(g["protocol"].iloc[0]), model=str(g["model"].iloc[0]),
                                n_test=float(g["n_test"].mean()), n_train=float(g["n_train"].mean()))
            s = one(summ, (summ["series"] == e) & (summ["target"] == t), f"D3a_summary.csv {e} {t}")
            close(s["mae"], v.mean(), TOL, f"D3a_summary.csv {e} {t} mae vs mean of D3a_per_seed.csv")
            close(s["se_seeds"], v.std(ddof=1) / math.sqrt(len(v)), TOL,
                  f"D3a_summary.csv {e} {t} se_seeds vs sd / sqrt(n) of D3a_per_seed.csv")
            # Espley's own SE (std(|err|)/sqrt(n_test), seed mean; rev 4 figure): CSV only, the bars use se_seeds
            info[(e, t)]["se_espley_def"] = float(s["se_espley_def"]) if "se_espley_def" in summ else float("nan")
        if e != "Espley" and {info[(e, t)]["model"] for t in T5} != {"KRR"}:
            raise FigError(f"D3a_per_seed.csv: series {e} is not KRR: {[info[(e, t)]['model'] for t in T5]}")
    if {info[("Espley", t)]["model"] for t in T5} != {"SVR"}:
        raise FigError(f"D3a_per_seed.csv: series Espley is not SVR: {[info[('Espley', t)]['model'] for t in T5]}")
    stats = {}
    for t in T5:
        r = one(tests, (tests["our_arm"] == "EXT_SEL") & (tests["comparator"] == "Espley SVR") & (tests["target"] == t),
                f"D3a_tests.csv EXT_SEL vs Espley SVR, {t}")
        e, x = M[("Espley", t)], M[("EXT_SEL", t)]
        if str(r["esp_model"]) != "SVR" or str(r["esp_protocol"]) != info[("Espley", t)]["protocol"]:
            raise FigError(f"D3a_tests.csv {t}: comparator {r['esp_model']} / {r['esp_protocol']} is not the Espley "
                           f"series of D3a_per_seed.csv (SVR / {info[('Espley', t)]['protocol']})")
        close(r["espley_mae"], e.mean(), TOL, f"D3a_tests.csv {t} espley_mae")
        close(r["ours_mae"], x.mean(), TOL, f"D3a_tests.csv {t} ours_mae")
        for s, dv in zip(SEEDS, e - x):
            close(r[f"d_seed{s}"], dv, TOL, f"D3a_tests.csv {t} d_seed{s}")
        close(r["diff_mean"], np.mean(e - x), TOL, f"D3a_tests.csv {t} diff_mean")
        close(r["mae_ratio_mean"], np.mean(e / x), TOL, f"D3a_tests.csv {t} mae_ratio_mean")
        close(r["n_test"], info[("EXT_SEL", t)]["n_test"], TOL, f"D3a_tests.csv {t} n_test")
        if not (np.isfinite(float(r["p"])) and 0 <= float(r["p"]) <= 1):
            raise FigError(f"D3a_tests.csv {t}: p = {r['p']}")
        stats[t] = dict(ratio=float(r["mae_ratio_mean"]), p=float(r["p"]), t=float(r["t"]),
                        diff_mean=float(r["diff_mean"]), ci95_lo=float(r["ci95_lo"]), ci95_hi=float(r["ci95_hi"]),
                        J=int(r["J"]), n_test=float(r["n_test"]), n_train=float(r["n_train_ours"]))
    notes = [f"Espley series, {t}: SVR, protocol {info[('Espley', t)]['protocol']}" for t in T5]
    return dict(M=M, info=info, stats=stats, notes=notes,
                inputs=[FILES[k][0] for k in ("d3a_per_seed", "d3a_summary", "d3a_tests")])


def build_d3a_preds(ctx):
    """Per-row test predictions of the 4 series (D3a_predictions.csv): identical test rows and y across series; the
    per-seed MAE recomputed from them must be the D3a_per_seed.csv value."""
    A = ctx.get("d3a", build_d3a)
    P = read("d3a_preds", {"series", "target", "seed", "rxn_id", "y", "yhat"})
    odd = sorted(set(P["series"].astype(str)) - set(ENTS))
    if odd:
        raise FigError(f"D3a_predictions.csv: unexpected series {odd}")
    P = P[P["target"].isin(T5)].copy()
    P["seed"] = ints(P["seed"], "D3a_predictions.csv seed")
    P["rxn_id"] = ints(P["rxn_id"], "D3a_predictions.csv rxn_id")
    if P[["y", "yhat"]].isna().any().any():
        raise FigError("D3a_predictions.csv: missing y / yhat")
    if P.duplicated(["series", "target", "seed", "rxn_id"]).any():
        raise FigError("D3a_predictions.csv: duplicate (series, target, seed, rxn_id)")
    for (t, s), g in P.groupby(["target", "seed"]):
        ref = None
        for e in ENTS:
            ids = set(g.loc[g["series"] == e, "rxn_id"].tolist())
            if ref is None:
                ref = ids
            elif ids != ref:
                raise FigError(f"D3a_predictions.csv {t} seed {s}: test rows of {e} differ from those of {ENTS[0]}")
    yy = P.groupby(["target", "rxn_id"])["y"].agg(["min", "max"])
    if float((yy["max"] - yy["min"]).max()) > TOL_Y:
        raise FigError("D3a_predictions.csv: one reaction has different y across series / seeds")
    P["ae"] = (P["yhat"] - P["y"]).abs()
    got = P.groupby(["series", "target", "seed"])["ae"].agg(["mean", "size"])
    for e in ENTS:
        for t in T5:
            for s, v in zip(SEEDS, A["M"][(e, t)]):
                if (e, t, s) not in got.index:
                    raise FigError(f"D3a_predictions.csv: no rows for {e} {t} seed {s}")
                close(got.loc[(e, t, s), "mean"], v, TOL_ROUND, f"D3a_predictions.csv MAE {e} {t} seed {s} vs "
                                                                f"D3a_per_seed.csv")
    return dict(P=P, notes=list(A["notes"]), inputs=[FILES["d3a_preds"][0]] + A["inputs"][:1])


# ---------------------------------------------------------------- inputs: D-3(b), D-4, C-2, D-1
def build_d3b(ctx):
    S = read("d3b_summary", {"series", "target", "model", "protocol", "mae", "mae_ci_lo", "mae_ci_hi", "n_test",
                             "n_train"})
    Pd = read("d3b_paired", {"target", "espley_model", "primary", "espley_mae", "ours_mae", "diff_espley_minus_ours",
                             "diff_ci_lo", "diff_ci_hi", "n_test", "n_boot"})
    Q = read("d3b_preds", {"series", "target", "rxn_id", "y", "yhat"})
    lock = read_ids(R5.LOCKBOX)
    prim = Pd["primary"].map(as_bool)
    Q = Q[Q["series"].isin(("Espley", "EXT_SEL")) & Q["target"].isin(T5)].copy()
    Q["rxn_id"] = ints(Q["rxn_id"], "D3b_predictions.csv rxn_id")
    rows, diff, n = {}, {}, set()
    for t in T5:
        ids = {}
        for e in ("Espley", "EXT_SEL"):
            r = one(S, (S["series"] == e) & (S["target"] == t), f"D3b_summary.csv {e} {t}")
            want = "SVR" if e == "Espley" else "KRR"
            if str(r["model"]) != want:
                raise FigError(f"D3b_summary.csv {e} {t}: model {r['model']}, expected {want}")
            g = Q[(Q["series"] == e) & (Q["target"] == t)]
            if g["rxn_id"].duplicated().any() or len(g) != int(r["n_test"]):
                raise FigError(f"D3b_predictions.csv {e} {t}: {len(g)} rows (duplicates?) for n_test {r['n_test']}")
            if not set(g["rxn_id"]) <= lock:
                raise FigError(f"D3b_predictions.csv {e} {t}: test rows outside lockbox_ids.csv")
            close(np.mean(np.abs(g["yhat"] - g["y"])), r["mae"], TOL_ROUND, f"D3b_predictions.csv MAE {e} {t}")
            ids[e] = g.set_index("rxn_id")["y"].sort_index()
            rows[(e, t)] = dict(mae=float(r["mae"]), lo=float(r["mae_ci_lo"]), hi=float(r["mae_ci_hi"]),
                                n_test=int(r["n_test"]), n_train=int(r["n_train"]), model=str(r["model"]),
                                protocol=str(r["protocol"]))
            n.add(int(r["n_test"]))
        if not ids["Espley"].index.equals(ids["EXT_SEL"].index):
            raise FigError(f"D3b_predictions.csv {t}: Espley and EXT_SEL test rows differ")
        if float(np.max(np.abs(ids["Espley"].to_numpy() - ids["EXT_SEL"].to_numpy()))) > TOL_Y:
            raise FigError(f"D3b_predictions.csv {t}: Espley and EXT_SEL disagree on y")
        p = one(Pd, prim & (Pd["target"] == t) & (Pd["espley_model"] == "SVR"), f"D3b_paired.csv primary SVR {t}")
        close(p["espley_mae"], rows[("Espley", t)]["mae"], TOL, f"D3b_paired.csv {t} espley_mae")
        close(p["ours_mae"], rows[("EXT_SEL", t)]["mae"], TOL, f"D3b_paired.csv {t} ours_mae")
        close(p["diff_espley_minus_ours"], rows[("Espley", t)]["mae"] - rows[("EXT_SEL", t)]["mae"], TOL,
              f"D3b_paired.csv {t} diff")
        diff[t] = dict(d=float(p["diff_espley_minus_ours"]), lo=float(p["diff_ci_lo"]), hi=float(p["diff_ci_hi"]),
                       n_boot=int(p["n_boot"]))
    if len(n) != 1:
        raise FigError(f"D3b_summary.csv: n_test differs between targets / series {sorted(n)}")
    notes = [f"lockbox head-to-head on {n.pop()} reactions; Espley = SVR, protocol "
             f"{rows[('Espley', T5[0])]['protocol']}"]
    return dict(rows=rows, diff=diff, notes=notes,
                inputs=[FILES[k][0] for k in ("d3b_summary", "d3b_paired", "d3b_preds")] + [R5.LOCKBOX])


def reach(curve, level):
    """curve sorted by frac (n_train_mean, mae_mean): first point with mae <= level, linearly interpolated from the
    previous point -> (n, status in interpolated / at_first_point / not_reached)."""
    x, y = curve["n_train_mean"].to_numpy(float), curve["mae_mean"].to_numpy(float)
    below = np.flatnonzero(y <= level)
    if not below.size:
        return None, "not_reached"
    k = int(below[0])
    if k == 0:
        return float(x[0]), "at_first_point"
    w = (level - y[k - 1]) / (y[k] - y[k - 1])
    return float(x[k - 1] + w * (x[k] - x[k - 1])), "interpolated"


def build_d4(ctx):
    L = read("d4", {"target", "side", "frac", "n_train", "seed", "mae"})
    S = read("d4_summary", {"target", "side", "frac", "n_train_mean", "n_train_min", "n_train_max", "mae_mean",
                            "mae_sd", "mae_se", "n_seeds"})
    RC = read("d4_reach", {"target", "reference", "reference_mae", "reached", "n_train", "interpolated",
                           "at_first_point"})
    sides = ("Espley", "EXT_SEL")
    fracs = [round(float(f), 6) for f in R5.LC_FRACS]
    for df, name in ((L, "D4_learning_curves.csv"), (S, "D4_learning_curves_summary.csv")):
        odd = sorted(set(df["side"].astype(str)) - set(sides))
        if odd:
            raise FigError(f"{name}: unexpected side {odd}")
        df["frac"] = pd.to_numeric(df["frac"], errors="coerce").round(6)
        odd = sorted(set(df["frac"].tolist()) - set(fracs))
        if odd:
            raise FigError(f"{name}: fractions {odd} are not LC_FRACS {fracs}")
    L["seed"] = ints(L["seed"], "D4_learning_curves.csv seed")
    L["n_train"] = ints(L["n_train"], "D4_learning_curves.csv n_train")
    for t in T5:
        for f in fracs:
            nt = {}
            for e in sides:
                g = L[(L["target"] == t) & (L["side"] == e) & (L["frac"] == f)]
                if sorted(g["seed"].tolist()) != sorted(SEEDS):
                    raise FigError(f"D4_learning_curves.csv {t} {e} frac {f}: seeds {sorted(g['seed'].tolist())}")
                s = one(S, (S["target"] == t) & (S["side"] == e) & (S["frac"] == f),
                        f"D4_learning_curves_summary.csv {t} {e} frac {f}")
                v = g["mae"].to_numpy(float)
                close(s["mae_mean"], v.mean(), TOL, f"D4 summary {t} {e} {f} mae_mean")
                close(s["mae_se"], v.std(ddof=1) / math.sqrt(len(v)), TOL, f"D4 summary {t} {e} {f} mae_se")
                close(s["n_train_mean"], g["n_train"].mean(), TOL, f"D4 summary {t} {e} {f} n_train_mean")
                nt[e] = g.set_index("seed")["n_train"].reindex(SEEDS).tolist()
            if nt["Espley"] != nt["EXT_SEL"]:
                raise FigError(f"D-4 {t} frac {f}: Espley and EXT_SEL training subsets differ in size "
                               f"{nt['Espley']} vs {nt['EXT_SEL']} (spec: the same subsets)")
    ref = RC["reference"].astype(str)
    is_d4 = ref.str.contains("D-4", regex=False) & ref.str.contains("frac 1.0", regex=False)
    reach_t = {}
    for t in T5:
        level = float(one(S, (S["target"] == t) & (S["side"] == "Espley") & (S["frac"] == 1.0),
                          f"D-4 Espley frac 1.0 {t}")["mae_mean"])
        r = one(RC, is_d4 & (RC["target"] == t), f"D4_learning_curves_reach.csv 'D-4 frac 1.0' reference, {t}")
        close(r["reference_mae"], level, TOL, f"D4 reach {t} reference_mae vs Espley frac 1.0")
        cur = S[(S["target"] == t) & (S["side"] == "EXT_SEL")].sort_values("frac")
        n_r, status = reach(cur, level)
        if as_bool(r["reached"]) != (status != "not_reached"):
            raise FigError(f"D4 reach {t}: file says reached={r['reached']}, recomputed {status}")
        if status != "not_reached":
            close(r["n_train"], n_r, 1e-6 * max(1.0, n_r), f"D4 reach {t} n_train")
            if as_bool(r["at_first_point"]) != (status == "at_first_point"):
                raise FigError(f"D4 reach {t}: at_first_point {r['at_first_point']} vs recomputed {status}")
        reach_t[t] = dict(level=level, n=n_r, status=status, reference=str(r["reference"]))
    S = S[S["target"].isin(T5)].copy()
    return dict(S=S, reach=reach_t,
                notes=["reference line = Espley SVR at D-4 frac 1.0 (all curve training rows = Espley's training "
                       "split ∩ row set, the same rows as the curve); the D-3(a) reference (Espley SVR stored / their "
                       "protocol, trained on their training rows) is in D4_learning_curves_reach.csv and not drawn"],
                inputs=[FILES[k][0] for k in ("d4", "d4_summary", "d4_reach")])


def build_c2(ctx):
    F = read("c2_folds", {"arm", "alias_of", "target", "fold", "nmae", "mae", "n_features"})
    F = F[F["target"].isin(R5.SEL_TARGETS) & F["arm"].isin(ARMS7)].copy()
    F["fold"] = ints(F["fold"], "C2_block_cv_folds.csv fold")
    for a in ARMS7:
        for t in R5.SEL_TARGETS:
            got = sorted(F.loc[(F["arm"] == a) & (F["target"] == t), "fold"].tolist())
            if got != list(range(R5.N_OUTER)):
                raise FigError(f"C2_block_cv_folds.csv {a} {t}: folds {got}, expected 0..{R5.N_OUTER - 1} once each")
    alias = sorted(set(F.loc[F["arm"] == "EXT_SEL", "alias_of"].dropna().astype(str)))
    if len(alias) != 1:
        raise FigError(f"C2_block_cv_folds.csv: EXT_SEL alias_of {alias}, expected one arm name")
    alias = alias[0]
    blocks = [] if alias == "BASE" else alias.split("+")[1:]
    if alias != "BASE" and (not alias.startswith("BASE+") or any(b not in R5.BLOCK_ORDER for b in blocks)):
        raise FigError(f"C2_block_cv_folds.csv: EXT_SEL alias_of {alias!r} is not a selection arm")
    inputs, notes = [FILES["c2_folds"][0]], []
    if R5.PREREG5B_JSON.is_file():
        pre = R5.ext_sel_blocks()
        if list(pre) != blocks:
            raise FigError(f"prereg_rev5b.json ext_sel_blocks {pre} != C2 EXT_SEL = {alias}")
        inputs.append(R5.PREREG5B_JSON)
    else:
        notes.append("prereg_rev5b.json absent: EXT_SEL composition from C2_block_cv_folds.csv alias_of")
    cv_path = FILES["c2_cv"][0]
    C = None
    if cv_path.is_file():
        C = read("c2_cv", {"arm", "target", "nmae_mean", "nmae_sd", "d_nmae_vs_base", "folds_nmae_below_base"})
        for a in ARMS7:
            for t in R5.SEL_TARGETS:
                v = F.loc[(F["arm"] == a) & (F["target"] == t)].sort_values("fold")["nmae"].to_numpy(float)
                r = one(C, (C["arm"] == a) & (C["target"] == t), f"C2_block_cv.csv {a} {t}")
                close(r["nmae_mean"], v.mean(), TOL, f"C2_block_cv.csv {a} {t} nmae_mean")
                close(r["nmae_sd"], v.std(ddof=1), TOL, f"C2_block_cv.csv {a} {t} nmae_sd")
        inputs.append(cv_path)
    else:
        notes.append("C2_block_cv.csv absent: means from C2_block_cv_folds.csv only")
    return dict(F=F, C=C, alias=alias, blocks=blocks, notes=notes, inputs=inputs)


def build_d1(ctx):
    T = read("d1", {"arm", "model", "target", "label", "mae", "mae_ci_lo", "mae_ci_hi", "n_test", "n_features"})
    Q = read("d1_paired", {"model", "target", "arm", "vs", "mae_arm", "mae_base", "diff_mae", "diff_ci_lo",
                           "diff_ci_hi", "n_test", "same_columns_as_base", "n_boot"})
    rows, pair, n = {}, {}, set()
    for t in R5.TARGETS9:
        for a in ("BASE", "EXT_SEL"):
            r = one(T, (T["model"] == HEAD_MODEL) & (T["arm"] == a) & (T["target"] == t), f"D1_lockbox.csv {a} {t}")
            rows[(a, t)] = dict(mae=float(r["mae"]), lo=float(r["mae_ci_lo"]), hi=float(r["mae_ci_hi"]),
                                n_test=int(r["n_test"]), n_features=int(r["n_features"]), label=str(r["label"]))
            n.add(int(r["n_test"]))
        q = one(Q, (Q["model"] == HEAD_MODEL) & (Q["arm"] == "EXT_SEL") & (Q["vs"] == "BASE") & (Q["target"] == t),
                f"D1_lockbox_paired.csv EXT_SEL - BASE {t}")
        close(q["mae_arm"], rows[("EXT_SEL", t)]["mae"], TOL, f"D1 paired {t} mae_arm")
        close(q["mae_base"], rows[("BASE", t)]["mae"], TOL, f"D1 paired {t} mae_base")
        close(q["diff_mae"], rows[("EXT_SEL", t)]["mae"] - rows[("BASE", t)]["mae"], TOL, f"D1 paired {t} diff_mae")
        pair[t] = dict(d=float(q["diff_mae"]), lo=float(q["diff_ci_lo"]), hi=float(q["diff_ci_hi"]),
                       same=as_bool(q["same_columns_as_base"]), n_boot=int(q["n_boot"]))
    if len(n) != 1:
        raise FigError(f"D1_lockbox.csv: n_test differs between rows {sorted(n)}")
    n = n.pop()
    notes = [f"D-1 lockbox n = {n}, {HEAD_MODEL}"]
    if all(p["same"] for p in pair.values()):
        notes.append("EXT_SEL has the columns of BASE (no block selected): paired differences are 0")
    return dict(rows=rows, pair=pair, n=n, notes=notes, inputs=[FILES["d1"][0], FILES["d1_paired"][0]])


# ---------------------------------------------------------------- fig1 / fig5: D-3(a) per seed
def draw_fig1(A):
    M, st, info = A["M"], A["stats"], A["info"]
    fig = new_fig(R5.FIG_W2, 82)
    ax = fig.subplots()
    grid(ax)
    group = 0.84
    slot = group / len(ENTS)
    arts, rows, ymax = [], [], 0.0
    for j, t in enumerate(T5):
        top = 0.0
        for i, e in enumerate(ENTS):
            v = M[(e, t)]
            m, sd = float(v.mean()), float(v.std(ddof=1))
            se = sd / math.sqrt(len(v))
            xc = j - group / 2 + slot * (i + 0.5)
            bar(ax, xc, m, slot, R5.COLORS[e], zorder=2)
            ax.errorbar(xc, m, yerr=se, **EB)
            ax.scatter(xc + np.linspace(-0.25, 0.25, len(v)) * slot, v, s=2.5, color=INK, linewidths=0, zorder=5)
            yt = max(m + se, float(v.max()))
            arts.append(vlabel(ax, xc, yt, num(m)))
            top = max(top, yt)
            rows.append(dict(target=t, target_label=TLAB[t], series=e, series_label=ELAB[e],
                             protocol=info[(e, t)]["protocol"], model=info[(e, t)]["model"], n_seeds=len(v),
                             mae_mean=m, se_seeds=se, sd_seeds=sd, se_espley_def=info[(e, t)]["se_espley_def"],
                             **{f"mae_seed_{s}": float(x) for s, x in zip(SEEDS, v)},
                             n_test_mean=info[(e, t)]["n_test"], n_train_mean=info[(e, t)]["n_train"],
                             ratio_espley_over_ext_sel=st[t]["ratio"], p_nadeau_bengio=st[t]["p"],
                             nb_t=st[t]["t"], nb_diff_mean=st[t]["diff_mean"], nb_ci95_lo=st[t]["ci95_lo"],
                             nb_ci95_hi=st[t]["ci95_hi"], nb_J=st[t]["J"], nb_n_test=st[t]["n_test"],
                             nb_n_train=st[t]["n_train"]))
        arts.append(above(ax, j, top, f"÷{num(st[t]['ratio'])}\n{fmt_p(st[t]['p'])}"))
        ymax = max(ymax, top)
    ax.set_xlim(-0.5, len(T5) - 0.5)
    ax.set_xticks(range(len(T5)), [TLAB2[t] for t in T5])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ymax * 1.02)
    ax.set_ylabel("test MAE (kcal/mol)\nmean of 5 seeds ± SE")
    fig._r5_fit.append((ax, arts))
    fig_legend(fig, [Patch(facecolor=R5.COLORS[e], label=ELAB[e]) for e in ENTS]
               + [Line2D([], [], ls="none", marker="o", ms=1.8, color=INK, label="single seed")])
    note(fig, f"Espley's splits (seeds {'/'.join(map(str, SEEDS))}); every series scored on the same test rows.\n"
              "÷ = Espley MAE / EXT_SEL MAE (mean of per-seed ratios); p = Nadeau–Bengio corrected t-test, "
              "EXT_SEL vs Espley SVR (J = 5)")
    return fig, pd.DataFrame(rows)


def draw_fig5(A):
    M, st = A["M"], A["stats"]
    fig = new_fig(R5.FIG_W2, 62)
    axs = fig.subplots(1, len(T5))
    rows = []
    for j, t in enumerate(T5):
        ax = axs[j]
        grid(ax)
        a, b = M[("Espley", t)], M[("EXT_SEL", t)]
        for s, va, vb in zip(SEEDS, a, b):
            ax.plot([0, 1], [va, vb], color=LINK, lw=0.6, zorder=1)
            rows.append(dict(target=t, target_label=TLAB[t], seed=s, mae_espley=float(va), mae_ext_sel=float(vb),
                             diff_espley_minus_ext_sel=float(va - vb), p_nadeau_bengio=st[t]["p"]))
        for x, v, e in ((0, a, "Espley"), (1, b, "EXT_SEL")):
            ax.scatter(np.full(len(v), x), v, s=10, color=R5.COLORS[e], edgecolors=INK, linewidths=0.3, zorder=3)
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        pad = 0.08 * max(hi - lo, 0.05 * hi)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], ["Espley", "EXT_SEL"])
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{TLAB2[t]}\n{fmt_p(st[t]['p'])}")
    axs[0].set_ylabel("test MAE per seed\n(kcal/mol)")
    fig_legend(fig, [Line2D([], [], ls="none", marker="o", ms=3.2, mfc=R5.COLORS["Espley"], mec=INK, mew=0.3,
                            label=ELAB["Espley"]),
                     Line2D([], [], ls="none", marker="o", ms=3.2, mfc=R5.COLORS["EXT_SEL"], mec=INK, mew=0.3,
                            label=ELAB["EXT_SEL"]),
                     Line2D([], [], color=LINK, lw=0.6, label="same seed (identical test rows)")])
    note(fig, "p = Nadeau–Bengio corrected resampled t-test on the 5 paired per-seed differences")
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------- fig3 / fig4: D-3(a) per row
def draw_fig3(B):
    P = B["P"]
    ents = ("Espley", "EXT_SEL")
    fig = new_fig(R5.FIG_W2, 96)
    axs = fig.subplots(2, len(T5), sharex="col", sharey="col")
    rows = []
    for j, t in enumerate(T5):
        per = {}
        for e in ents:
            g = P[(P["series"] == e) & (P["target"] == t)]
            per[e] = (g.groupby("rxn_id", sort=True)
                      .agg(y=("y", "mean"), yhat=("yhat", "mean"), n_test_seeds=("seed", "nunique")).reset_index())
        allv = np.concatenate([np.r_[a["y"].to_numpy(float), a["yhat"].to_numpy(float)] for a in per.values()])
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        xx = np.array([lo, hi])
        for i, e in enumerate(ents):
            ax, a = axs[i, j], per[e]
            y, p = a["y"].to_numpy(float), a["yhat"].to_numpy(float)
            ax.fill_between(xx, xx - 2, xx + 2, color=BAND2, linewidth=0, zorder=0)
            ax.fill_between(xx, xx - 1, xx + 1, color=BAND1, linewidth=0, zorder=0.5)
            ax.plot(xx, xx, color=INK, lw=0.6, zorder=1)
            ax.scatter(y, p, s=1.6, color=R5.COLORS[e], alpha=0.55, linewidths=0, zorder=2)   # vector (spec)
            mae = float(np.mean(np.abs(p - y)))
            r2 = float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
            ax.text(0.04, 0.96, f"MAE {num(mae)}\nr² {num(r2)}", transform=ax.transAxes, ha="left", va="top",
                    color=INK)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            grid(ax, "both")
            if i == 0:
                ax.set_title(TLAB2[t] + ("\n(+ = stabilising)" if t in SIGN_CONV else ""))
            rows += [dict(target=t, series=e, rxn_id=int(r), y_dft=float(yv), yhat_mean=float(pv),
                          n_test_seeds=int(k), mae_points=mae, r2_points=r2, n_points=len(a),
                          sign_convention=SIGN_CONV.get(t, ""))
                     for r, yv, pv, k in zip(a["rxn_id"], y, p, a["n_test_seeds"])]
    axs[0, 0].set_ylabel("Espley SVR\npredicted (kcal/mol)")
    axs[1, 0].set_ylabel("EXT_SEL KRR\npredicted (kcal/mol)")
    fig_legend(fig, [Line2D([], [], ls="none", marker="o", ms=2.5, color=R5.COLORS["Espley"], label=ELAB["Espley"]),
                     Line2D([], [], ls="none", marker="o", ms=2.5, color=R5.COLORS["EXT_SEL"], label=ELAB["EXT_SEL"]),
                     Line2D([], [], color=INK, lw=0.6, label="y = x"),
                     Patch(facecolor=BAND1, label="±1 kcal/mol"), Patch(facecolor=BAND2, label="±2 kcal/mol")])
    note(fig, "DFT (kcal/mol). One point per reaction: prediction averaged over the seeds in which it was a test "
              "row; MAE and r² of the points shown. Interaction in Espley's sign convention (positive = stabilising, "
              "= −E_int)")
    return fig, pd.DataFrame(rows)


def draw_fig4(B):
    P = B["P"]
    fig = new_fig(R5.FIG_W2, 64)
    axs = fig.subplots(1, len(T5))
    rows = []
    for j, t in enumerate(T5):
        ax = axs[j]
        grid(ax, "both")
        err = {e: np.sort(P.loc[(P["series"] == e) & (P["target"] == t), "ae"].to_numpy(float)) for e in ENTS}
        xmax = max(2.5, math.ceil(2 * max(float(np.quantile(v, 0.99)) for v in err.values())) / 2)
        xs = np.linspace(0.0, xmax, 201)
        for e in ENTS:
            v = err[e]
            n = len(v)
            ax.step(np.r_[0.0, v], np.r_[0.0, np.arange(1, n + 1) / n], where="post", color=R5.COLORS[e], lw=0.9,
                    zorder=3)
            cum = np.searchsorted(v, xs, side="right") / n
            rows += [dict(row_type="ecdf", target=t, series=e, abs_err=float(x), cum_frac=float(c), n_errors=n)
                     for x, c in zip(xs, cum)]
            for thr in (1.0, 2.0):
                rows.append(dict(row_type=f"within_{thr:g}", target=t, series=e, abs_err=thr,
                                 cum_frac=float(np.mean(v <= thr)), n_errors=n))
        for thr in (1.0, 2.0):
            ax.axvline(thr, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.text(0.97, 0.52, "≤ 1 kcal/mol", transform=ax.transAxes, ha="right", va="center", color=INK2)
        for i, e in enumerate(ENTS):
            f1 = float(np.mean(err[e] <= 1.0))
            ax.plot([1.0], [f1], ls="none", marker="o", ms=2.6, mfc=R5.COLORS[e], mec=INK, mew=0.3, zorder=5)
            yy = 0.52 - 0.105 * (i + 1)
            ax.plot([0.60], [yy], transform=ax.transAxes, ls="none", marker="s", ms=3.2, color=R5.COLORS[e],
                    clip_on=False)
            ax.text(0.97, yy, f"{100 * f1:.0f}%", transform=ax.transAxes, ha="right", va="center", color=INK)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(TLAB2[t])
    axs[0].set_ylabel("cumulative fraction\nof test predictions")
    fig_legend(fig, [Line2D([], [], color=R5.COLORS[e], lw=0.9, label=ELAB[e]) for e in ENTS]
               + [Line2D([], [], color=INK2, lw=0.5, ls=(0, (2, 2)), label="1 and 2 kcal/mol")])
    note(fig, "|prediction − DFT| (kcal/mol); test rows of the 5 seeds pooled; percentages = fraction within "
              "1 kcal/mol")
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------- fig2: D-3(b)
def draw_fig2(D):
    rows, diff = D["rows"], D["diff"]
    ents = ("Espley", "EXT_SEL")
    lab = {"Espley": "Espley et al. (SVR, their protocol)", "EXT_SEL": "Ours EXT_SEL (KRR, nested)"}
    fig = new_fig(R5.FIG_W2, 72)
    ax = fig.subplots()
    grid(ax)
    group = 0.6
    slot = group / len(ents)
    arts, out, ymax, clipped = [], [], 0.0, []
    for j, t in enumerate(T5):
        top = 0.0
        for i, e in enumerate(ents):
            r = rows[(e, t)]
            xc = j - group / 2 + slot * (i + 0.5)
            bar(ax, xc, r["mae"], slot, R5.COLORS[e], zorder=2)
            lo_e, hi_e = r["mae"] - r["lo"], r["hi"] - r["mae"]
            if lo_e < 0 or hi_e < 0:
                clipped.append(f"{e} {t}")
            ax.errorbar(xc, r["mae"], yerr=[[max(lo_e, 0.0)], [max(hi_e, 0.0)]], **EB)
            yt = max(r["hi"], r["mae"])
            arts.append(vlabel(ax, xc, yt, num(r["mae"])))
            top = max(top, yt)
            out.append(dict(target=t, target_label=TLAB[t], series=e, series_label=lab[e], model=r["model"],
                            protocol=r["protocol"], n_train=r["n_train"], n_test=r["n_test"], mae=r["mae"],
                            mae_ci_lo=r["lo"], mae_ci_hi=r["hi"], diff_espley_minus_ext_sel=diff[t]["d"],
                            diff_ci_lo=diff[t]["lo"], diff_ci_hi=diff[t]["hi"], n_boot=diff[t]["n_boot"]))
        d = diff[t]
        arts.append(above(ax, j, top, f"Δ {num(d['d'], sign=True)}\n[{num(d['lo'], sign=True)}, "
                                      f"{num(d['hi'], sign=True)}]"))
        ymax = max(ymax, top)
    ax.set_xlim(-0.5, len(T5) - 0.5)
    ax.set_xticks(range(len(T5)), [TLAB2[t] for t in T5])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ymax * 1.02)
    ax.set_ylabel("lockbox test MAE (kcal/mol)")
    fig._r5_fit.append((ax, arts))
    n_test = rows[("EXT_SEL", T5[0])]["n_test"]
    fig_legend(fig, [Patch(facecolor=R5.COLORS[e], label=lab[e]) for e in ents]
               + [Line2D([], [], color=INK, lw=0.6, marker="_", ms=4, label="95 % bootstrap CI")])
    note(fig, f"Train: dev ∩ Espley rows; test: lockbox ∩ Espley rows (n = {n_test:,}). Δ = Espley − EXT_SEL MAE "
              f"with its paired reaction-level bootstrap 95 % CI ({diff[T5[0]]['n_boot']:,} resamples)")
    if clipped:
        D.setdefault("notes", []).append(f"point estimate outside its percentile CI (error bar clipped at 0): "
                                         f"{clipped}")
    return fig, pd.DataFrame(out)


# ---------------------------------------------------------------- fig6: D-4
def draw_fig6(D):
    S, R = D["S"], D["reach"]
    sides = ("Espley", "EXT_SEL")
    fig = new_fig(R5.FIG_W2, 64)
    axs = fig.subplots(1, len(T5))
    rows = []
    for j, t in enumerate(T5):
        ax = axs[j]
        grid(ax, "both")
        lo, hi = np.inf, -np.inf
        for e in sides:
            g = S[(S["target"] == t) & (S["side"] == e)].sort_values("frac")
            x, y, se = (g["n_train_mean"].to_numpy(float), g["mae_mean"].to_numpy(float),
                        g["mae_se"].to_numpy(float))
            ax.errorbar(x, y, yerr=se, color=R5.COLORS[e], marker="o", ms=2.2, lw=0.9, elinewidth=0.6, capsize=1.2,
                        capthick=0.6, zorder=3)
            lo, hi = min(lo, float((y - se).min())), max(hi, float((y + se).max()))
            rows += [dict(target=t, target_label=TLAB[t], series=e, frac=float(r.frac),
                          n_train_mean=float(r.n_train_mean), n_train_min=float(r.n_train_min),
                          n_train_max=float(r.n_train_max), mae_mean=float(r.mae_mean),
                          mae_se=float(r.mae_se), mae_sd=float(r.mae_sd), n_seeds=int(r.n_seeds),
                          ref_mae_espley_svr_frac1=R[t]["level"], reference=R[t]["reference"],
                          reach_n_train=R[t]["n"], reach_status=R[t]["status"])
                     for r in g.itertuples(index=False)]
        rr = R[t]
        ax.axhline(rr["level"], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        if rr["status"] == "not_reached":
            txt = "not reached"
        else:
            ax.axvline(rr["n"], color=INK2, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
            ax.plot([rr["n"]], [rr["level"]], ls="none", marker="o", ms=3.2, mfc="white", mec=INK, mew=0.6, zorder=4)
            txt = f"n {'≈' if rr['status'] == 'interpolated' else '≤'} {rr['n']:,.0f}"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", color=INK)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 3 * pad)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(TLAB2[t])
    axs[0].set_ylabel("test MAE (kcal/mol)\nmean ± SE of 5 seeds")
    handles = [Line2D([], [], color=R5.COLORS[e], marker="o", ms=2.2, lw=0.9, label=lab)
               for e, lab in (("Espley", "Espley et al. (SVR)"), ("EXT_SEL", "Ours EXT_SEL (KRR)"))]
    handles.append(Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 2)),
                          label="Espley SVR, all curve training rows (frac 1.0)"))
    how = {"interpolated": "n ≈: linear interpolation", "at_first_point": "n ≤: at the smallest subset"}
    got = [how[s] for s in how if any(R[t]["status"] == s for t in T5)]
    if got:                                                      # the marker is drawn only in panels that reach it
        handles.append(Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=INK, mew=0.6,
                              label=f"EXT_SEL reaches it ({'; '.join(got)})"))
    fig_legend(fig, handles)
    note(fig, "training reactions (n; the same random subsets for both sides, fixed test rows, Espley's splits). "
              "Dashed line: Espley SVR trained on all curve training rows (Espley's training split ∩ our row set), "
              "not the D-3(a) Espley SVR trained on all their training rows")
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------- fig7: C-2
def draw_fig7(C):
    F, blocks = C["F"], C["blocks"]
    color = {"BASE": R5.COLORS["ESPLEY73"], "EXT_SEL": R5.COLORS["EXT_SEL"], "EXT_ALL": GREY_ALL}
    ticks = ["BASE"] + [f"+{b}" for b in R5.BLOCK_ORDER] + ["EXT_SEL", "EXT_ALL"]
    sel_txt = "EXT_SEL = BASE + " + " + ".join(blocks) if blocks else "EXT_SEL = BASE (no block selected)"
    fig = new_fig(R5.FIG_W2, 86)
    axs = fig.subplots(1, len(R5.SEL_TARGETS))
    rows = []
    for j, t in enumerate(R5.SEL_TARGETS):
        ax = axs[j]
        grid(ax)
        arts, ymax = [], 0.0
        base = F[(F["arm"] == "BASE") & (F["target"] == t)]["nmae"].mean()
        for k, a in enumerate(ARMS7):
            g = F[(F["arm"] == a) & (F["target"] == t)].sort_values("fold")
            v = g["nmae"].to_numpy(float)
            m = float(v.mean())
            bar(ax, k, m, 0.8, color.get(a, GREY_BLOCK), zorder=2)
            ax.scatter(k + np.linspace(-0.22, 0.22, len(v)) * 0.8, v, s=2.5, color=INK, linewidths=0, zorder=4)
            yt = max(m, float(v.max()))
            arts.append(vlabel(ax, k, yt, num(m, 3), rot=90, pad=4))     # clear of the fold dots
            ymax = max(ymax, yt)
            row = dict(target=t, target_label=TARGET_LABEL[t], arm=a, arm_label=ticks[k],
                       ext_sel_is=C["alias"] if a == "EXT_SEL" else "", n_features=int(g["n_features"].iloc[0]),
                       nmae_mean=m, nmae_sd=float(v.std(ddof=1)), d_nmae_vs_base=m - float(base),
                       mae_mean=float(g["mae"].mean()), **{f"nmae_fold{int(f)}": float(x) for f, x in
                                                         zip(g["fold"], v)})
            if C["C"] is not None:
                r = one(C["C"], (C["C"]["arm"] == a) & (C["C"]["target"] == t), f"C2_block_cv.csv {a} {t}")
                row["folds_nmae_below_base"] = int(r["folds_nmae_below_base"])
            rows.append(row)
        ax.axhline(float(base), color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.6, len(ARMS7) - 0.4)
        ax.set_xticks(range(len(ARMS7)), ticks, rotation=90)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, ymax * 1.02)
        ax.set_title(TARGET_LABEL[t])
        fig._r5_fit.append((ax, arts))
    axs[0].set_ylabel(f"dev-CV NMAE\n(mean of {R5.N_OUTER} outer folds)")
    fig_legend(fig, [Patch(facecolor=R5.COLORS["ESPLEY73"], label="BASE = ESPLEY73"),
                     Patch(facecolor=GREY_BLOCK, label="BASE + one block"),
                     Patch(facecolor=R5.COLORS["EXT_SEL"], label=sel_txt),
                     Patch(facecolor=GREY_ALL, label=f"EXT_ALL = BASE + {R5.BLOCK_ORDER[0]}–{R5.BLOCK_ORDER[-1]}"),
                     Line2D([], [], ls="none", marker="o", ms=1.8, color=INK, label="outer fold"),
                     Line2D([], [], color=INK2, lw=0.5, ls=(0, (3, 2)), label="BASE mean")])
    note(fig, "KRR (RBF), dev rows only (lockbox never used); NMAE = MAE / mean absolute deviation of the fold's "
              "test targets")
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------- fig8: D-1
def draw_fig8(D):
    rows, pair = D["rows"], D["pair"]
    arms = ("BASE", "EXT_SEL")
    col = {"BASE": R5.COLORS["ESPLEY73"], "EXT_SEL": R5.COLORS["EXT_SEL"]}
    same = all(p["same"] for p in pair.values())
    lab = {"BASE": "BASE = ESPLEY73 (KRR)",
           "EXT_SEL": "EXT_SEL (KRR)" + (" = BASE: no block selected" if same else "")}
    fig = new_fig(R5.FIG_W2, 132)
    axs = fig.subplots(3, 3)
    slot = 0.36
    out, clipped = [], []
    for k, t in enumerate(R5.TARGETS9):
        ax = axs.flat[k]
        grid(ax)
        nd = 2 if max(rows[(a, t)]["mae"] for a in arms) >= 1 else 3
        arts, ymax = [], 0.0
        for i, a in enumerate(arms):
            r = rows[(a, t)]
            xc = (i - 0.5) * slot
            bar(ax, xc, r["mae"], slot, col[a], zorder=2)
            lo_e, hi_e = r["mae"] - r["lo"], r["hi"] - r["mae"]
            if lo_e < 0 or hi_e < 0:
                clipped.append(f"{a} {t}")
            ax.errorbar(xc, r["mae"], yerr=[[max(lo_e, 0.0)], [max(hi_e, 0.0)]], **EB)
            yt = max(r["hi"], r["mae"])
            arts.append(vlabel(ax, xc, yt, num(r["mae"], nd)))
            ymax = max(ymax, yt)
            p = pair[t]
            out.append(dict(target=t, target_label=r["label"], arm=a, n_features=r["n_features"], n_test=r["n_test"],
                            mae=r["mae"], mae_ci_lo=r["lo"], mae_ci_hi=r["hi"], diff_ext_sel_minus_base=p["d"],
                            diff_ci_lo=p["lo"], diff_ci_hi=p["hi"], same_columns_as_base=p["same"],
                            n_boot=p["n_boot"]))
        p = pair[t]
        ax.set_title(f"{rows[('BASE', t)]['label']}\nΔ {num(p['d'], nd, True)} [{num(p['lo'], nd, True)}, "
                     f"{num(p['hi'], nd, True)}]")
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.set_ylim(0, ymax * 1.02)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if k % 3 == 0:
            ax.set_ylabel("lockbox MAE\n(kcal/mol)")
        fig._r5_fit.append((ax, arts))
    fig_legend(fig, [Patch(facecolor=col[a], label=lab[a]) for a in arms]
               + [Line2D([], [], color=INK, lw=0.6, marker="_", ms=4, label="95 % bootstrap CI")])
    note(fig, f"Trained on the dev rows (nested tuning), scored once on the lockbox (n = {D['n']:,}). Δ = EXT_SEL − "
              f"BASE MAE with its paired reaction-level bootstrap 95 % CI ({pair[R5.TARGETS9[0]]['n_boot']:,} "
              f"resamples)")
    if clipped:
        D.setdefault("notes", []).append(f"point estimate outside its percentile CI (error bar clipped at 0): "
                                         f"{clipped}")
    return fig, pd.DataFrame(out)


# ---------------------------------------------------------------- driver
class Ctx:
    """Input builders run once per run; a FigError is kept and re-raised for every figure that needs the input."""

    def __init__(self):
        self.cache = {}

    def get(self, key, fn):
        if key not in self.cache:
            try:
                self.cache[key] = (True, fn(self))
            except FigError as e:
                self.cache[key] = (False, e)
        ok, v = self.cache[key]
        if not ok:
            raise FigError(str(v), missing=v.missing)
        return v


FIGS = (
    ("fig1_espley_mae", "D-3(a)", lambda c: c.get("d3a", build_d3a), draw_fig1),
    ("fig2_espley_lockbox", "D-3(b)", lambda c: c.get("d3b", build_d3b), draw_fig2),
    ("fig3_parity", "D-3(a)", lambda c: c.get("d3a_preds", build_d3a_preds), draw_fig3),
    ("fig4_error_ecdf", "D-3(a)", lambda c: c.get("d3a_preds", build_d3a_preds), draw_fig4),
    ("fig5_seed_pairs", "D-3(a)", lambda c: c.get("d3a", build_d3a), draw_fig5),
    ("fig6_learning_curves", "D-4", lambda c: c.get("d4", build_d4), draw_fig6),
    ("fig7_channel_blocks", "C-2", lambda c: c.get("c2", build_c2), draw_fig7),
    ("fig8_channel_lockbox", "D-1", lambda c: c.get("d1", build_d1), draw_fig8),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma list of figures, full names or prefixes (fig1,fig7)")
    ap.add_argument("--force", action="store_true", help="redraw even if the outputs are up to date")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        die("figures_rev5.py runs inside a SLURM job only (sbatch r5_figures.sh), never on the login node")
    names = [f[0] for f in FIGS]
    only = [o.strip() for o in args.only.split(",") if o.strip()]
    unknown = [o for o in only if not any(n == o or n.startswith(o + "_") for n in names)]
    if unknown:
        die(f"--only: unknown figure(s) {unknown}; known {names}")
    todo = [f for f in FIGS if not only or any(f[0] == o or f[0].startswith(o + "_") for o in only)]

    family = style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    code = {rel(p): R5.sha256(p) for p in CODE_FILES}
    prev = {}
    if MANIFEST.is_file():
        try:
            prev = json.loads(MANIFEST.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"NOTE: {rel(MANIFEST)} unreadable ({type(e).__name__}); every requested figure is redrawn")
    man = dict(spec="docs/specs/REV5_FEATURES_FIGURES.md Phase E", updated=now(), job=os.environ.get("SLURM_JOB_ID"),
               host=socket.gethostname(), code=code, font=family, matplotlib=matplotlib.__version__,
               style=dict(colors=R5.COLORS, width_mm=dict(single=89, double=183), font_pt=R5.FONT_PT, png_dpi=DPI,
                          bar_gap_px=GAP_PX, text_colors=[INK, INK2], grid=GRID,
                          fig7_greys=dict(block=GREY_BLOCK, ext_all=GREY_ALL)),
               figures=dict(prev.get("figures") or {}))
    print(f"[figures_rev5] {now()} job {os.environ.get('SLURM_JOB_ID')} host {socket.gethostname()} font {family} "
          f"matplotlib {matplotlib.__version__}\n  results {rel(RES)} -> {rel(FIG_DIR)}", flush=True)

    ctx, bad = Ctx(), []
    for name, phase, data_fn, draw_fn in todo:
        t0 = time.time()
        rec = dict(phase=phase, checked=now())
        try:
            data = data_fn(ctx)
            sig = json.loads(json.dumps(dict(code=code, font=family, matplotlib=matplotlib.__version__,
                                             inputs={rel(p): R5.sha256(p) for p in data["inputs"]})))
            outs = [FIG_DIR / f"{name}.{x}" for x in ("pdf", "png", "csv")]
            old = man["figures"].get(name) or {}
            if not args.force and all(o.is_file() for o in outs) and old.get("status") == "ok" \
                    and old.get("signature") == sig:
                rec = dict(old, checked=now(), this_run="up to date (not redrawn)")
                print(f"{name}: up to date", flush=True)
            else:
                fig, table = draw_fn(data)
                warn = save(fig, name, table)
                rec.update(status="ok", this_run="drawn", signature=sig, outputs=[rel(o) for o in outs],
                           csv_rows=int(len(table)), notes=list(data.get("notes") or []), layout_warnings=warn,
                           seconds=round(time.time() - t0, 1))
                print(f"{name}: drawn ({len(table)} CSV rows, {time.time() - t0:.1f} s)", flush=True)
                for w in warn:
                    print(f"  LAYOUT WARNING {name}: {w}", flush=True)
        except FigError as e:
            rec.update(status="skipped", reason=str(e), input_missing=e.missing)
            bad.append(name)
            print(f"{name}: SKIPPED — {e}", flush=True)
        except Exception as e:                                                    # noqa: BLE001
            rec.update(status="failed", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            bad.append(name)
            print(f"{name}: FAILED — {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        finally:
            plt.close("all")
        man["figures"][name] = rec
        R5.write_json(MANIFEST, man)                         # after every figure: a clipped run keeps its record

    print(f"\n{'figure':24s} status")
    for name, *_ in todo:
        r = man["figures"][name]
        extra = r.get("reason") or r.get("this_run", "")
        nw = len(r.get("layout_warnings") or [])
        print(f"{name:24s} {r.get('status', '?'):8s} {extra}{f'  ({nw} layout warnings)' if nw else ''}")
    print(f"manifest {rel(MANIFEST)}")
    if bad:
        print(f"\n{len(bad)} of {len(todo)} figures not drawn: {bad}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
