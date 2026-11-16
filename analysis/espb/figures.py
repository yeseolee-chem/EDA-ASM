#!/usr/bin/env python3
"""ESPB Phase F (docs/specs/ESPB.md 9): four figures -> results/figures/ (PDF + PNG 300 dpi + the plotted values as CSV).

    python figures.py           PYENV=reactot (jobs/py.slurm, submitted from analysis/espb), after compare.py

fig1_primary      MAE of GNN3 and ESPB per target with the core-bootstrap 95 % interval; panels P_main all / P_main para;
                  value labels, d and Holm p per target, the chosen candidate under each ESPB bar
fig2_parity       P_main barrier ΔΔ, DFT vs predicted; panels GNN3 / ESPB on shared axes, y = x, ±1 kcal/mol band
fig3_candidates   per target: the 10 candidates' P_main MAE (E-A circle, E-B square), GNN3 line, MAE(0) dashed line,
                  inner-score rank next to each point
fig4_positions    P_main MAE per position (p, m, o, direct), 4 targets x (GNN3, ESPB)
Style: Arial (else DejaVu Sans) 7 pt, 89 / 183 mm widths, GNN3 #eb6834, Espley side #2a78d6, text black / grey, one y
axis, legends for >= 2 series, 2 px gaps between bars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eb_common as C  # noqa: E402

C.path_setup()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402

from gnn3 import metrics as M  # noqa: E402

OUT = C.RES / "figures"
GNN3_COL, ESP_COL, GREY, INK = "#eb6834", "#2a78d6", "#6b6b6b", "#1a1a1a"
MM = 1 / 25.4
LABEL = {"strain": "strain", "inter": "inter", "solv": "solv", "barrier": "barrier"}


def style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in names else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": GREY,
                         "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.0,
                         "axes.linewidth": 0.6})
    return family


def save(fig, name, table):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        tmp = OUT / f".{name}.{ext}.tmp"
        fig.savefig(tmp, format=ext, bbox_inches="tight", **kw)
        tmp.replace(OUT / f"{name}.{ext}")
    C.write_atomic(OUT / f"{name}.csv", table.to_csv(index=False))
    plt.close(fig)


def long_frame():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gnn3_evaluate", C.GNN3_PACKAGE / "scripts" / "evaluate.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    truth, attrs = M.channel_truth(C.PAIRS)
    sets = M.pair_sets(C.S_FOLDS)
    cols = ["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]
    espb = pd.read_parquet(C.RES / "C_espb.parquet")
    cp = pd.read_parquet(C.RES / "C_predictions.parquet")
    models = {"GNN3": E.read_predictions(C.GNN3_CV, False), "ESPB": espb[cols].assign(pair_id=espb.pair_id.astype(str))}
    for arm in C.ARMS:
        for m in C.MODELS:
            d = cp[(cp.arm == arm) & (cp.model == m)][cols]
            models[C.model_name(arm, m)] = d.assign(pair_id=d.pair_id.astype(str))
    return M.assemble(models, truth, attrs, sets)


def subset(long, model, target, pos=None):
    f = long[(long.model == model) & (long.target == target)]
    f = f[M.set_mask(f, "P_main")]
    if pos is not None:
        f = f[f.pos == pos]
    return f.sort_values("pair_id", kind="mergesort")


def mae_ci(f):
    e = (f.yhat - f.y).to_numpy(float)
    mae, lo, hi, _ = M.bootstrap_mae_difference(e, np.zeros(len(e)), f.core.to_numpy(), C.BOOT_N, C.BOOT_SEED)
    return mae, lo, hi


def bars(ax, x, values, err, color, width, offset):
    """bars with a 2 px surface gap (white edge) and the error bar of the CI"""
    b = ax.bar(x + offset, values, width=width, color=color, edgecolor="white", linewidth=2 * 72 / 300 * 1.0)
    ax.errorbar(x + offset, values, yerr=err, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    return b


def fig1(long, prim, sec, chosen):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), sharey=False)
    for ax, (panel, pos) in zip(axes, (("P_main all", None), ("P_main para", "p"))):
        x = np.arange(len(C.CHANNELS))
        for name, col, off in (("GNN3", GNN3_COL, -0.19), ("ESPB", ESP_COL, 0.19)):
            vals, los, his = [], [], []
            for t in C.CHANNELS:
                mae, lo, hi = mae_ci(subset(long, name, t, pos))
                vals.append(mae); los.append(lo); his.append(hi)
                rows.append(dict(panel=panel, model=name, target=t, mae=mae, ci_lo=lo, ci_hi=hi,
                                 candidate=chosen[t] if name == "ESPB" else ""))
            err = np.vstack([np.array(vals) - np.array(los), np.array(his) - np.array(vals)])
            bars(ax, x, vals, err, col, 0.36, off)
            for xi, v, h in zip(x + off, vals, his):
                ax.text(xi, h + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
        top = max(r["ci_hi"] for r in rows if r["panel"] == panel)
        for i, t in enumerate(C.CHANNELS):
            if pos is None:
                r = prim.set_index("target").loc[t]
            else:
                r = sec[(sec.comparison == "ESPB vs GNN3") & (sec.eval_set == "P_main") & (sec.subset == "para")
                        & (sec.target == t)].iloc[0]
            ax.text(i, top * 1.18, f"d {r['d']:+.2f}\nHolm p {r['p_holm']:.2g}", ha="center", va="bottom", fontsize=6,
                    color=GREY)
            ax.text(i + 0.19, -0.06 * top, chosen[t], ha="center", va="top", fontsize=5.5, color=ESP_COL, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[t] for t in C.CHANNELS])
        ax.tick_params(axis="x", pad=9)
        ax.set_ylim(0, top * 1.45)
        ax.set_ylabel("MAE of ΔΔ (kcal/mol)")
        ax.set_title(panel, loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GNN3_COL), plt.Rectangle((0, 0), 1, 1, color=ESP_COL)]
    # figure-level legend below the panels: inside an axis it covers the d / Holm p notes
    fig.legend(handles, ["GNN3", "ESPB (Espley-type)"], frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig1_primary", pd.DataFrame(rows))


def fig2(long):
    rows, frames = [], {}
    for name in ("GNN3", "ESPB"):
        f = subset(long, name, "barrier")
        frames[name] = f
        rows += [dict(model=name, pair_id=p, y=a, yhat=b) for p, a, b in zip(f.pair_id, f.y, f.yhat)]
    lo = min(min(f.y.min(), f.yhat.min()) for f in frames.values())
    hi = max(max(f.y.max(), f.yhat.max()) for f in frames.values())
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 88 * MM), sharex=True, sharey=True)
    for ax, (name, col) in zip(axes, (("GNN3", GNN3_COL), ("ESPB", ESP_COL))):
        f = frames[name]
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="#d9d9d9", linewidth=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color=GREY, linewidth=0.8, zorder=1)
        ax.scatter(f.y, f.yhat, s=4, color=col, alpha=0.5, linewidths=0, zorder=2)
        mae, r = float(np.abs(f.yhat - f.y).mean()), M.pearson(f.y, f.yhat)
        ax.text(0.04, 0.96, f"{name}\nMAE {mae:.2f} kcal/mol\nr {r:.3f}\nn {len(f):,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("DFT ΔΔ barrier (kcal/mol)")
    axes[0].set_ylabel("predicted ΔΔ barrier (kcal/mol)")
    fig.tight_layout()
    save(fig, "fig2_parity", pd.DataFrame(rows))


def fig3(long, sel):
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 60 * MM))
    for ax, t in zip(axes, C.CHANNELS):
        g = float(np.abs(subset(long, "GNN3", t).yhat - subset(long, "GNN3", t).y).mean())
        zero = float(np.abs(subset(long, "GNN3", t).y).mean())
        s = sel[sel.target == t].set_index("candidate")
        for i, (arm, mk) in enumerate((("EA", "o"), ("EB", "s"))):
            for j, m in enumerate(C.MODELS):
                nm = C.model_name(arm, m)
                f = subset(long, nm, t)
                mae = float(np.abs(f.yhat - f.y).mean())
                xi = j + (i - 0.5) * 0.3
                ax.scatter([xi], [mae], marker=mk, s=16, color=ESP_COL, edgecolors="white", linewidths=0.5, zorder=3)
                ax.text(xi, mae, f" {int(s.loc[nm, 'rank'])}", fontsize=5.5, color=GREY, va="center", ha="left",
                        zorder=4, path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
                rows.append(dict(target=t, candidate=nm, arm=arm, model=m, mae_P_main=mae, inner_score=float(s.loc[nm, "score"]),
                                 inner_rank=int(s.loc[nm, "rank"]), mae_gnn3=g, mae_zero=zero))
        ax.axhline(g, color=GNN3_COL, linewidth=1.0, label="GNN3")
        ax.axhline(zero, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), label="predict 0")
        ax.set_xticks(range(len(C.MODELS)))
        ax.set_xticklabels(C.MODELS, rotation=45, ha="right")
        ax.set_title(LABEL[t], loc="left")
        ax.set_ylim(0, max(zero, max(r["mae_P_main"] for r in rows if r["target"] == t)) * 1.15)
    axes[0].set_ylabel("P_main MAE of ΔΔ (kcal/mol)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=ESP_COL, linestyle="", label="E-A (predict a, b, subtract)"),
               Line2D([], [], marker="s", color=ESP_COL, linestyle="", label="E-B (pair model)"),
               Line2D([], [], color=GNN3_COL, label="GNN3"),
               Line2D([], [], color=GREY, linestyle=(0, (3, 2)), label="predict 0")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_candidates", pd.DataFrame(rows))


def fig4(long):
    rows = []
    pos = ["p", "m", "o", "direct"]
    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 56 * MM))
    for ax, t in zip(axes, C.CHANNELS):
        x = np.arange(len(pos))
        for name, col, off in (("GNN3", GNN3_COL, -0.19), ("ESPB", ESP_COL, 0.19)):
            vals = []
            for p in pos:
                f = subset(long, name, t, p)
                v = float(np.abs(f.yhat - f.y).mean())
                vals.append(v)
                rows.append(dict(target=t, pos=p, model=name, n=int(len(f)), mae=v))
            ax.bar(x + off, vals, width=0.36, color=col, edgecolor="white", linewidth=2 * 72 / 300, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(pos)
        ax.set_title(LABEL[t], loc="left")
    axes[0].set_ylabel("P_main MAE of ΔΔ (kcal/mol)")
    # figure-level legend below the panels: inside the last axis it covers the "direct" bars
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig4_positions", pd.DataFrame(rows))


def main() -> int:
    family = style()
    long = long_frame()
    prim = pd.read_csv(C.RES / "E_primary.csv")
    sec = pd.read_csv(C.RES / "E_secondary.csv")
    sel = pd.read_csv(C.RES / "C_selection.csv")
    chosen = sel[sel.chosen].set_index("target").candidate.to_dict()
    fig1(long, prim, sec, chosen)
    fig2(long)
    fig3(long, sel)
    fig4(long)
    rep = dict(font=family, figures=sorted(p.name for p in OUT.glob("fig*")),
               sha256={p.name: C.sha256(p) for p in sorted(OUT.glob("fig*"))})
    C.write_json(OUT / "F_report.json", rep)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
