"""spec21 D1 — position our 400 within Stuyver's 5269 by G_act, G_r.

Reports summary stats for full_5269, ours_400, locked_192, spec16_208;
Kolmogorov–Smirnov tests (ours vs full; locked vs spec16); overlaid
density plot.

Gibbs is used ONLY as a coordinate to locate our reactions — no
comparison of magnitudes to our own electronic-energy labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec21_cohort_bias_diagnosis"
IN_JOINED = STAGE / "results/cohort_joined.parquet"
IN_STUYVER = STAGE / "results/stuyver_full.parquet"
OUT_STATS = STAGE / "results/D1_reactivity_stats.csv"
OUT_KS = STAGE / "results/D1_ks_tests.csv"
OUT_FIG = STAGE / "figures/D1_reactivity_position.png"


def stats(x: np.ndarray) -> dict:
    return {
        "n":      int(x.size),
        "mean":   float(np.mean(x)),
        "sd":     float(np.std(x, ddof=1)),
        "min":    float(np.min(x)),
        "q05":    float(np.quantile(x, 0.05)),
        "q25":    float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75":    float(np.quantile(x, 0.75)),
        "q95":    float(np.quantile(x, 0.95)),
        "max":    float(np.max(x)),
    }


def main() -> int:
    STAGE.joinpath("figures").mkdir(exist_ok=True)
    joined = pd.read_parquet(IN_JOINED)
    stuyver = pd.read_parquet(IN_STUYVER)

    groups = {
        "full_5269":  stuyver,
        "ours_400":   joined,
        "locked_192": joined[joined["sub_source"] == "locked_778"],
        "spec16_208": joined[joined["sub_source"] == "spec16"],
    }

    rows = []
    for tgt in ("G_act", "G_r"):
        for name, df in groups.items():
            s = stats(df[tgt].dropna().values)
            rows.append({"target": tgt, "group": name, **s})
    pd.DataFrame(rows).to_csv(OUT_STATS, index=False)
    print(f"[write] {OUT_STATS}")

    # KS tests
    ks_rows = []
    for tgt in ("G_act", "G_r"):
        for a, b in [("ours_400", "full_5269"), ("locked_192", "spec16_208")]:
            xa = groups[a][tgt].dropna().values
            xb = groups[b][tgt].dropna().values
            stat, p = ks_2samp(xa, xb)
            ks_rows.append({"target": tgt, "a": a, "b": b, "n_a": len(xa), "n_b": len(xb),
                             "ks_stat": float(stat), "p_value": float(p)})
    pd.DataFrame(ks_rows).to_csv(OUT_KS, index=False)
    print(f"[write] {OUT_KS}")
    print(pd.DataFrame(ks_rows).to_string(index=False))

    # Overlaid density (histogram)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, tgt, xlabel in zip(axes, ("G_act", "G_r"),
                                ("Stuyver ΔG‡ [kcal/mol]", "Stuyver ΔG_r [kcal/mol]")):
        bins = np.linspace(*np.quantile(stuyver[tgt].values, [0.005, 0.995]), 50)
        ax.hist(stuyver[tgt].values, bins=bins, density=True, alpha=0.35,
                color="#888", edgecolor="none", label="full_5269")
        for name, df in groups.items():
            if name == "full_5269":
                continue
            ax.hist(df[tgt].values, bins=bins, density=True, alpha=0.55,
                    histtype="step", linewidth=1.6,
                    label=f"{name} (n={len(df)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    print(f"[write] {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
