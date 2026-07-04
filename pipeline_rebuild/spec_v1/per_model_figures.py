"""Per-model NMAE / RMSE / MAE bars + 6-panel parity grid, with OOD removed.

OOD detection: pooled per-channel residuals across all 25 cells (member 0
for parity; all 5 members for bars). Modified Z-score (Iglewicz–Hoaglin):
  mod_z = 0.6745 · (r - median(r)) / MAD(r)
Reaction rid excluded if |mod_z| > Z_THRESH on ANY of the 5 channels,
applied uniformly across m1 / m2 / m3 so the kept set is identical.

Outputs into m{1,2,3}/figures/:
  nmae_bar.png, rmse_bar.png, mae_bar.png, parity_grid.png
Also writes m{1,2,3}/figures/excluded_rids.json for provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
CHANNELS = ["strain", "Pauli", "V_elst", "oi", "disp"]
CHANNELS_BAR = CHANNELS + ["barrier"]
COLORS = {"m1": "#1E2761", "m2": "#1C7293", "m3": "#C45A4D"}

Z_THRESH = 5.0


def load_cells(model_root: Path) -> list[dict]:
    cells = []
    for f in sorted(model_root.glob("fold*/member*.json")):
        cells.append(json.load(open(f)))
    return cells


def channel_metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ybar = float(yt.mean())
    denom = float(np.mean(np.abs(yt - ybar)))
    nmae = mae / denom if denom > 0 else float("nan")
    ss_tot = float(np.sum((yt - ybar) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    x = yt - ybar; yc = yp - float(yp.mean())
    slope = float(np.sum(x * yc) / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "NMAE": nmae, "R2": r2, "slope": slope}


def detect_ood(cells_by_model: dict, z_thresh: float = Z_THRESH) -> set[str]:
    """Return set of rids to exclude (union across models, any channel)."""
    excluded = set()
    for name, cells in cells_by_model.items():
        # Pool per-channel residuals across all cells for this model
        residuals_by_ch = [[] for _ in CHANNELS]
        rids_flat = []
        for c in cells:
            yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
            for r, y_true_row, y_pred_row in zip(c["reaction_ids"], yt, yp):
                rids_flat.append(r)
                for i in range(5):
                    residuals_by_ch[i].append(y_pred_row[i] - y_true_row[i])
        rids_arr = np.array(rids_flat)
        for i in range(5):
            r = np.array(residuals_by_ch[i])
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            if mad == 0:
                continue
            mod_z = 0.6745 * (r - med) / mad
            bad = rids_arr[np.abs(mod_z) > z_thresh]
            excluded.update(bad.tolist())
    return excluded


def per_cell_metrics(cells: list[dict], excluded: set[str]) -> dict:
    """Return {channel: {metric: [per-cell values]}}, with excluded rids
    dropped from each cell before computing metrics."""
    out = {c: {"MAE": [], "RMSE": [], "NMAE": [], "R2": []} for c in CHANNELS_BAR}
    for c in cells:
        yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
        rids = np.array(c["reaction_ids"])
        keep = ~np.isin(rids, list(excluded))
        yt = yt[keep]; yp = yp[keep]
        if len(yt) < 2:
            continue
        for i, ch in enumerate(CHANNELS):
            m = channel_metrics(yt[:, i], yp[:, i])
            for k in ("MAE", "RMSE", "NMAE", "R2"):
                out[ch][k].append(m[k])
        m = channel_metrics(yt.sum(axis=1), yp.sum(axis=1))
        for k in ("MAE", "RMSE", "NMAE", "R2"):
            out["barrier"][k].append(m[k])
    return out


def bar_fig(metrics: dict, model_name: str, ykey: str, ylabel: str, color: str,
            out_path: Path, ref_line: float | None = None,
            note: str = "") -> None:
    means = np.array([np.mean(metrics[c][ykey]) for c in CHANNELS_BAR])
    stds = np.array([np.std(metrics[c][ykey]) for c in CHANNELS_BAR])
    x = np.arange(len(CHANNELS_BAR))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x, means, 0.6, yerr=stds, color=color, capsize=3,
           edgecolor="white", linewidth=0.5)
    if ref_line is not None:
        ax.axhline(ref_line, color="gray", ls="--", lw=0.8,
                   label=f"mean-predictor ({ref_line:.0f})")
        ax.legend(fontsize=9, framealpha=0.9)
    ax.axvline(len(CHANNELS) - 0.5, color="gray", ls=":", lw=0.6, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS_BAR, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    title = f"{model_name} — {ylabel} (mean ± std across cells)"
    if note: title += f"\n{note}"
    ax.set_title(title, fontsize=10)
    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.2f}" if m >= 1 else f"{m:.3f}"
        ax.text(i, m + s + (means.max() * 0.02), label,
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parity_grid(cells: list[dict], excluded: set[str], model_name: str,
                color: str, out_path: Path) -> None:
    m0 = [c for c in cells if c.get("member", 0) == 0]
    yt_all = []
    yp_all = []
    for c in m0:
        yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
        rids = np.array(c["reaction_ids"])
        keep = ~np.isin(rids, list(excluded))
        yt_all.append(yt[keep]); yp_all.append(yp[keep])
    yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    for k, ch in enumerate(CHANNELS_BAR):
        ax = axes[k // 3, k % 3]
        if ch == "barrier":
            yt = yt_all.sum(axis=1); yp = yp_all.sum(axis=1)
        else:
            yt = yt_all[:, CHANNELS.index(ch)]
            yp = yp_all[:, CHANNELS.index(ch)]
        m = channel_metrics(yt, yp)
        ax.scatter(yt, yp, s=8, c=color, alpha=0.5, edgecolor="none")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = (hi - lo) * 0.05
        span = np.array([lo - pad, hi + pad])
        ax.plot(span, span, "--", color="#888", lw=0.7, label="y = x")
        x_c = yt - yt.mean(); y_c = yp - yp.mean()
        b = float(np.sum(x_c * y_c) / np.sum(x_c ** 2)) if np.sum(x_c ** 2) > 0 else 1.0
        a = float(yp.mean() - b * yt.mean())
        ax.plot(span, b * span + a, "-", color="orange", lw=1.2,
                label=f"OLS: slope={b:.2f}")
        ax.set_title(f"{ch}", fontsize=11)
        ax.text(0.03, 0.97,
                f"MAE={m['MAE']:.2f}\nNMAE={m['NMAE']:.3f}\nR²={m['R2']:.3f}\nslope={m['slope']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"))
        ax.set_xlabel("y_true (kcal/mol)", fontsize=9)
        ax.set_ylabel("y_pred (kcal/mol)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{model_name} — parity plots (member-0, {len(yt_all)} pts, "
                 f"OOD removed at mod-Z > {Z_THRESH})",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cells_by_model = {name: load_cells(REPO / name / "results")
                      for name in ("m1", "m2", "m3")}
    excluded = detect_ood(cells_by_model)
    print(f"OOD excluded across all models: {len(excluded)} unique rids")

    for name in ("m1", "m2", "m3"):
        model_root = REPO / name / "results"
        fig_root = REPO / name / "figures"
        fig_root.mkdir(exist_ok=True)
        cells = cells_by_model[name]
        met = per_cell_metrics(cells, excluded)
        color = COLORS[name]
        note = f"OOD removed (mod-Z > {Z_THRESH}), {len(excluded)} rids excluded"
        bar_fig(met, name, "NMAE", "NMAE = MAE / MAD(y_true)", color,
                fig_root / "nmae_bar.png", ref_line=1.0, note=note)
        bar_fig(met, name, "RMSE", "RMSE (kcal/mol)", color,
                fig_root / "rmse_bar.png", note=note)
        bar_fig(met, name, "MAE", "MAE (kcal/mol)", color,
                fig_root / "mae_bar.png", note=note)
        parity_grid(cells, excluded, name, color, fig_root / "parity_grid.png")
        (fig_root / "excluded_rids.json").write_text(json.dumps(sorted(excluded), indent=1))
        print(f"{name}: wrote 4 figures + excluded_rids.json  "
              f"(kept {len(cells[0]['reaction_ids']) - len([r for r in cells[0]['reaction_ids'] if r in excluded])} rxns/cell)")


if __name__ == "__main__":
    main()
