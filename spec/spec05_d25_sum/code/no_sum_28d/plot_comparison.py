"""Aggregate the 4 per-variant JSONs and draw RMSE + NMAE bar comparisons.

Bars compare {24d, 28d} for each of {XGB, Ridge}, per channel + barrier.
Output figures + CSV under spec/spec05_d25_sum/{figures,results}/no_sum_28d/.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RES = REPO / "spec/spec05_d25_sum/results/no_sum_28d"
FIG = REPO / "spec/spec05_d25_sum/figures/no_sum_28d"
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]
CHANNELS_PLOT = CHANNELS + ["barrier"]
VARIANTS = ["xgb_24d", "xgb_28d", "ridge_24d", "ridge_28d"]
COLORS = {
    "xgb_24d":   "#7a7a7a",  # gray  (XGB ref)
    "xgb_28d":   "#8b3a62",  # magenta (XGB +4 proxies)
    "ridge_24d": "#4b779a",  # blue  (Ridge ref)
    "ridge_28d": "#4ba36c",  # green (Ridge +4 proxies)
}
LABELS = {
    "xgb_24d":   "XGB 24-d (base)",
    "xgb_28d":   "XGB 28-d (+d25,d26,d27,d28)",
    "ridge_24d": "Ridge 24-d (m3 base)",
    "ridge_28d": "Ridge 28-d (m3 + d25,d26,d27,d28)",
}


def load_all():
    rows = []
    for v in VARIANTS:
        p = RES / f"{v}_fold0.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; did all 4 jobs finish?")
        d = json.loads(p.read_text())
        for ch in CHANNELS:
            rows.append({"variant": v, "channel": ch,
                         "NMAE": d["channels"][ch]["NMAE"],
                         "RMSE": d["channels"][ch]["RMSE"],
                         "R2":   d["channels"][ch]["R2"],
                         "n_train": d["n_train"], "n_test": d["n_test"],
                         "D": d["D"]})
        rows.append({"variant": v, "channel": "barrier",
                     "NMAE": d["barrier"]["NMAE"],
                     "RMSE": d["barrier"]["RMSE"],
                     "R2":   d["barrier"]["R2"],
                     "n_train": d["n_train"], "n_test": d["n_test"],
                     "D": d["D"]})
    return pd.DataFrame(rows)


def bar_figure(df, metric, out_path):
    """Grouped bar: x = channels, groups = 4 variants."""
    x = np.arange(len(CHANNELS_PLOT)); w = 0.20
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(VARIANTS):
        vals = [float(df[(df.variant == v) & (df.channel == ch)][metric].iloc[0])
                for ch in CHANNELS_PLOT]
        ax.bar(x + (i - 1.5) * w, vals, w, label=LABELS[v],
               color=COLORS[v], edgecolor="white", lw=0.4)
        for xi, val in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333")
    if metric == "NMAE":
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="mean-predictor")
        ax.set_ylabel("NMAE (fold-0 test)")
    else:
        ax.set_ylabel(f"{metric} (kcal/mol, fold-0 test)")
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS_PLOT)
    ax.set_title(f"{metric}: 24-d vs 28-d for XGB & Ridge (fold-0, no sum-consistency)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    df = load_all()
    df.to_csv(RES / "metrics.csv", index=False)
    print(df.to_string(index=False))
    bar_figure(df, "NMAE", FIG / "compare_NMAE_24d_vs_28d.png")
    bar_figure(df, "RMSE", FIG / "compare_RMSE_24d_vs_28d.png")

    # Wide table for quick eyeballing
    wide = df.pivot_table(index="channel", columns="variant",
                          values=["NMAE", "RMSE"])
    wide = wide.reindex(CHANNELS_PLOT)
    wide.to_csv(RES / "metrics_wide.csv")
    print("\nWide table:\n", wide)


if __name__ == "__main__":
    main()
