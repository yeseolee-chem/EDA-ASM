"""Render m3 spec2 ablation figures. Login-node OK (matplotlib only)."""
from __future__ import annotations
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent               # models/m3/code
M3_DIR = HERE.parent                                 # models/m3
FIG_DIR = M3_DIR / "figures" / "ablation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMPS = ["strain", "Pauli", "Velst", "oi", "disp"]


def load_cells(pattern: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(pattern))]


def channel_stats(cells: list[dict]) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    pc = np.array([c["test_mae_per_channel"] for c in cells])        # (m, 5)
    bm = np.array([c["test_barrier_mae"] for c in cells])
    br = np.array([c["test_barrier_rmse"] for c in cells])
    return pc.mean(0), pc.std(0), bm.mean(), bm.std(), br.mean(), br.std()


def label_mad_from_full(cells: list[dict]) -> np.ndarray:
    """Compute MAD(y_true) across pooled test predictions of these cells,
    so NMAE = MAE / MAD is comparable across variants."""
    yt = np.concatenate([np.array(c["y_true"]) for c in cells], axis=0)
    bt = np.concatenate([np.array(c["barrier_true"]) for c in cells], axis=0)
    mad_c = np.median(np.abs(yt - np.median(yt, axis=0)), axis=0)
    mad_bar = float(np.median(np.abs(bt - np.median(bt))))
    return np.concatenate([mad_c, [mad_bar]])


# ---------------------------------------------------------------------------
# Load the three variants
full_cells = load_cells(
    str(M3_DIR / "code/trackB_lowlr_v9_xtb_geom6_plus_v2/m3_delta/fold0/member*.json"))
delta_cells = load_cells(
    str(M3_DIR / "code/trackB_ablation/delta_only/fold0/member*.json"))

# Grid: pick best alpha by val_mae_mean
grid_dir = M3_DIR / "code/trackB_ablation/baseline_only"
best_tag, best_val = None, float("inf")
grid_summary = []
for alpha_dir in sorted(grid_dir.iterdir()):
    cells = load_cells(str(alpha_dir / "fold0/member*.json"))
    if not cells:
        continue
    val_mean = float(np.mean([np.mean(c["val_mae_per_channel"]) for c in cells]))
    val_std = float(np.std([np.mean(c["val_mae_per_channel"]) for c in cells]))
    tm_mean = float(np.mean([c["test_mae_mean_kcal"] for c in cells]))
    tm_std = float(np.std([c["test_mae_mean_kcal"] for c in cells]))
    bm_mean = float(np.mean([c["test_barrier_mae"] for c in cells]))
    br_mean = float(np.mean([c["test_barrier_rmse"] for c in cells]))
    # decode alpha value from tag ("a0p001" -> 0.001, "a0" -> 0)
    tag = alpha_dir.name
    aval = float(tag[1:].replace("p", "."))
    grid_summary.append(dict(
        alpha=aval, tag=tag, val=val_mean, val_std=val_std,
        test=tm_mean, test_std=tm_std, bmae=bm_mean, brmse=br_mean,
    ))
    if val_mean < best_val:
        best_val, best_tag = val_mean, tag

baseline_cells = load_cells(str(grid_dir / best_tag / "fold0/member*.json"))
best_alpha = float(best_tag[1:].replace("p", "."))

# ---------------------------------------------------------------------------
# Figure 1 — per-channel MAE bar (3 modes)
variants = [
    ("baseline_only (α*)", baseline_cells, "#1f77b4"),
    ("delta_only",         delta_cells,     "#d62728"),
    ("full (b + δ)",       full_cells,      "#2ca02c"),
]
labels = COMPS + ["barrier"]
x = np.arange(len(labels))
w = 0.26

fig, ax = plt.subplots(figsize=(10, 5.2))
for i, (name, cells, color) in enumerate(variants):
    mu_c, sd_c, mu_b, sd_b, _, _ = channel_stats(cells)
    heights = np.concatenate([mu_c, [mu_b]])
    errs = np.concatenate([sd_c, [sd_b]])
    ax.bar(x + (i - 1) * w, heights, w, yerr=errs, capsize=3,
           label=name, color=color, edgecolor="black", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("test MAE (kcal/mol)")
ax.set_title(f"m3 fold0 ablation — best α = {best_alpha:g} for baseline_only")
ax.axvline(len(COMPS) - 0.5, color="grey", ls="--", lw=0.8, alpha=0.6)
ax.legend()
ax.grid(axis="y", alpha=0.3, ls=":")
fig.tight_layout()
p1 = FIG_DIR / "mae_bar_ablation.png"
fig.savefig(p1, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 — per-channel NMAE (MAE / MAD(y_true)) bar
mad_ref = label_mad_from_full(full_cells)   # per-channel + barrier
mad_ref = np.where(mad_ref < 1e-9, 1.0, mad_ref)

fig, ax = plt.subplots(figsize=(10, 5.2))
for i, (name, cells, color) in enumerate(variants):
    mu_c, sd_c, mu_b, sd_b, _, _ = channel_stats(cells)
    heights = np.concatenate([mu_c, [mu_b]]) / mad_ref
    errs = np.concatenate([sd_c, [sd_b]]) / mad_ref
    ax.bar(x + (i - 1) * w, heights, w, yerr=errs, capsize=3,
           label=name, color=color, edgecolor="black", linewidth=0.4)
ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6, label="mean-predictor")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("NMAE = MAE / MAD(y_true)")
ax.set_title(f"m3 fold0 ablation NMAE — best α = {best_alpha:g} for baseline_only")
ax.axvline(len(COMPS) - 0.5, color="grey", ls="--", lw=0.8, alpha=0.6)
ax.legend()
ax.grid(axis="y", alpha=0.3, ls=":")
fig.tight_layout()
p2 = FIG_DIR / "nmae_bar_ablation.png"
fig.savefig(p2, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 — Ridge α grid sweep
grid_summary.sort(key=lambda r: r["alpha"])
alphas = np.array([r["alpha"] for r in grid_summary])
val = np.array([r["val"] for r in grid_summary])
val_sd = np.array([r["val_std"] for r in grid_summary])
tst = np.array([r["test"] for r in grid_summary])
tst_sd = np.array([r["test_std"] for r in grid_summary])
bmae = np.array([r["bmae"] for r in grid_summary])
brmse = np.array([r["brmse"] for r in grid_summary])

# Categorical x-axis (all α equally spaced) — clean spacing, no symlog padding.
xpos = np.arange(len(alphas))
xlabels = [("0 (OLS)" if a == 0 else f"{a:g}") for a in alphas]
best_xpos = int(np.where(alphas == best_alpha)[0][0])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.errorbar(xpos, val, yerr=val_sd, marker="o", label="val MAE (mean)",
            color="#1f77b4", capsize=3)
ax.errorbar(xpos, tst, yerr=tst_sd, marker="s", label="test MAE (mean)",
            color="#ff7f0e", capsize=3)
ax.axvline(best_xpos, color="green", ls="--", lw=1.0,
           label=f"best α = {best_alpha:g}")
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels, rotation=0)
ax.set_xlim(-0.4, len(alphas) - 0.6)
ax.set_xlabel("ridge α")
ax.set_ylabel("MAE (kcal/mol, mean over 5 members)")
ax.set_title("baseline_only α sweep — channel-mean MAE")
ax.grid(alpha=0.3, ls=":")
ax.legend()

ax = axes[1]
ax.plot(xpos, bmae, marker="o", label="barrier MAE",   color="#2ca02c")
ax.plot(xpos, brmse, marker="s", label="barrier RMSE", color="#9467bd")
ax.axvline(best_xpos, color="green", ls="--", lw=1.0)
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels, rotation=0)
ax.set_xlim(-0.4, len(alphas) - 0.6)
ax.set_xlabel("ridge α")
ax.set_ylabel("kcal/mol")
ax.set_title("baseline_only α sweep — barrier")
ax.grid(alpha=0.3, ls=":")
ax.legend()

fig.suptitle("m3 spec2 ablation — ridge α grid search (baseline_only, fold 0)")
fig.tight_layout()
p3 = FIG_DIR / "alpha_sweep.png"
fig.savefig(p3, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 — parity grid (3 rows: baseline_only, delta_only, full)
rows = [
    ("baseline_only (α*)", baseline_cells, "#1f77b4"),
    ("delta_only",         delta_cells,    "#d62728"),
    ("full (b + δ)",       full_cells,     "#2ca02c"),
]
n_col = len(COMPS) + 1  # 5 channels + barrier
fig, axes = plt.subplots(len(rows), n_col, figsize=(3.2 * n_col, 3.0 * len(rows)))

for ri, (name, cells, color) in enumerate(rows):
    yt = np.concatenate([np.array(c["y_true"]) for c in cells], axis=0)   # (N,5)
    yp = np.concatenate([np.array(c["y_pred"]) for c in cells], axis=0)   # (N,5)
    bt = np.concatenate([np.array(c["barrier_true"]) for c in cells])
    bp = np.concatenate([np.array(c["barrier_pred"]) for c in cells])
    stacks_true = list(yt.T) + [bt]
    stacks_pred = list(yp.T) + [bp]
    for ci, (yt_c, yp_c) in enumerate(zip(stacks_true, stacks_pred)):
        ax = axes[ri, ci]
        ax.scatter(yt_c, yp_c, s=6, alpha=0.35, color=color, edgecolors="none")
        lo, hi = float(min(yt_c.min(), yp_c.min())), float(max(yt_c.max(), yp_c.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        mae = float(np.mean(np.abs(yp_c - yt_c)))
        rmse = float(np.sqrt(np.mean((yp_c - yt_c) ** 2)))
        ax.set_title(f"{labels[ci]}\nMAE={mae:.2f}  RMSE={rmse:.2f}",
                     fontsize=9)
        if ri == len(rows) - 1:
            ax.set_xlabel("y_true (kcal/mol)")
        if ci == 0:
            ax.set_ylabel(f"{name}\ny_pred")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, ls=":")
fig.suptitle("m3 fold0 ablation — parity (rows: variants, columns: channels + barrier)",
             y=1.005)
fig.tight_layout()
p4 = FIG_DIR / "parity_grid_ablation.png"
fig.savefig(p4, dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Also write a tiny JSON summary
summary = {
    "fold": 0,
    "best_alpha_by_val_mae_mean": best_alpha,
    "variants": {
        "baseline_only": dict(
            alpha=best_alpha, n_members=len(baseline_cells),
            test_mae_mean=float(np.mean([c["test_mae_mean_kcal"] for c in baseline_cells])),
            barrier_mae=float(np.mean([c["test_barrier_mae"] for c in baseline_cells])),
            barrier_rmse=float(np.mean([c["test_barrier_rmse"] for c in baseline_cells])),
        ),
        "delta_only": dict(
            n_members=len(delta_cells),
            test_mae_mean=float(np.mean([c["test_mae_mean_kcal"] for c in delta_cells])),
            barrier_mae=float(np.mean([c["test_barrier_mae"] for c in delta_cells])),
            barrier_rmse=float(np.mean([c["test_barrier_rmse"] for c in delta_cells])),
        ),
        "full": dict(
            n_members=len(full_cells),
            test_mae_mean=float(np.mean([c["test_mae_mean_kcal"] for c in full_cells])),
            barrier_mae=float(np.mean([c["test_barrier_mae"] for c in full_cells])),
            barrier_rmse=float(np.mean([c["test_barrier_rmse"] for c in full_cells])),
        ),
    },
    "alpha_grid": grid_summary,
}
(FIG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

print("saved:")
for p in (p1, p2, p3, p4, FIG_DIR / "summary.json"):
    print(f"  {p}")
