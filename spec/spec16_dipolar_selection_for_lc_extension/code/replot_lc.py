"""Re-plot the SPEC_16 400 LC with per-panel y-scaling (barrier separated)."""
from __future__ import annotations
import os, tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec16_dipolar_selection_for_lc_extension"
SUMMARY = SPEC / "results/lc_summary.csv"
FIG_DIR = SPEC / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_SHORT = ["strain", "Pauli", "elst", "oi", "disp"]
palette = {"strain": "#4c72b0", "Pauli": "#dd8452", "elst": "#55a868",
           "oi": "#c44e52", "disp": "#8172b3", "barrier": "black"}


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".{}_".format(path.name), dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh: fh.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp): os.unlink(tmp)
        raise


def save_fig(fig, path: Path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight"); plt.close(fig)
    with open(tmp.name, "rb") as fh: atomic_write_bytes(path, fh.read())
    os.unlink(tmp.name)


summary = pd.read_csv(SUMMARY).sort_values(["channel", "size"])
sizes = sorted(summary["size"].unique().tolist())
print(f"loaded {len(summary)} summary rows, sizes = {sizes}")


# --------- Figure 1: 6 panels (5 channels + barrier) each with own y-scale ---------
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

for ax, ch in zip(axes[:5], CHANNEL_SHORT):
    sub = summary[summary.channel == ch]
    y = sub["NMAE_mean"].to_numpy()
    e = sub["NMAE_std"].to_numpy()
    ax.errorbar(sub["size"], y, yerr=e,
                marker="o", capsize=3, color=palette[ch], lw=1.5)
    ax.set_xlabel("train size"); ax.set_ylabel("NMAE")
    ax.grid(alpha=0.3); ax.set_xticks(sizes)
    # tight y-range with 10% padding, but keep a bit of headroom
    lo = max(0, float((y - e).min()) * 0.95)
    hi = float((y + e).max()) * 1.05
    ax.set_ylim(lo, hi)
    # annotate first + last points
    ax.annotate(f"{y[0]:.3f}", (sub["size"].iloc[0], y[0]),
                textcoords="offset points", xytext=(6, 6), fontsize=8, color=palette[ch])
    ax.annotate(f"{y[-1]:.3f}", (sub["size"].iloc[-1], y[-1]),
                textcoords="offset points", xytext=(-32, -14), fontsize=8, color=palette[ch])

# barrier panel with its own tight scale
ax = axes[5]
sub = summary[summary.channel == "barrier"]
y = sub["NMAE_mean"].to_numpy(); e = sub["NMAE_std"].to_numpy()
ax.errorbar(sub["size"], y, yerr=e, marker="s", capsize=3, color="black", lw=1.5)
ax.axhline(1.0, color="gray", ls="--", lw=0.7, label="NMAE = 1 (mean-pred)")
ax.set_xlabel("train size"); ax.set_ylabel("NMAE")
ax.grid(alpha=0.3); ax.set_xticks(sizes)
lo = max(0, float((y - e).min()) * 0.95); hi = float((y + e).max()) * 1.05
ax.set_ylim(lo, hi)
ax.legend(fontsize=8, loc="upper right")
ax.annotate(f"{y[0]:.3f}", (sub["size"].iloc[0], y[0]),
            textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.annotate(f"{y[-1]:.3f}", (sub["size"].iloc[-1], y[-1]),
            textcoords="offset points", xytext=(-32, -14), fontsize=8)

fig.tight_layout()
save_fig(fig, FIG_DIR / "lc_channels_and_barrier.png")
print(f"[wrote] {FIG_DIR / 'lc_channels_and_barrier.png'}")


# --------- Figure 2: two overlays (channels only + barrier only) side-by-side ---------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# left: 5 channels overlay (excluding barrier, so scale is comparable)
ax = axes[0]
for ch in CHANNEL_SHORT:
    sub = summary[summary.channel == ch]
    y = sub["NMAE_mean"].to_numpy(); e = sub["NMAE_std"].to_numpy()
    ax.errorbar(sub["size"], y, yerr=e, marker="o", capsize=2,
                label=ch, color=palette[ch], lw=1.5)
ax.set_xlabel("train size"); ax.set_ylabel("NMAE")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3); ax.set_xticks(sizes)
# tight scale for channels only (max ~0.6)
ch_data = summary[summary.channel.isin(CHANNEL_SHORT)]
ax.set_ylim(0, float((ch_data.NMAE_mean + ch_data.NMAE_std).max()) * 1.05)

# right: barrier only
ax = axes[1]
sub = summary[summary.channel == "barrier"]
y = sub["NMAE_mean"].to_numpy(); e = sub["NMAE_std"].to_numpy()
ax.errorbar(sub["size"], y, yerr=e, marker="s", capsize=3, color="black", lw=1.5,
            label="barrier NMAE")
ax.axhline(1.0, color="gray", ls="--", lw=0.7, label="NMAE = 1 (mean-pred)")
ax.set_xlabel("train size"); ax.set_ylabel("barrier NMAE")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3); ax.set_xticks(sizes)
lo = max(0.8, float((y - e).min()) * 0.98); hi = float((y + e).max()) * 1.02
ax.set_ylim(lo, hi)

fig.tight_layout()
save_fig(fig, FIG_DIR / "lc_overlay_split.png")
print(f"[wrote] {FIG_DIR / 'lc_overlay_split.png'}")

print("\n[replot complete]")
