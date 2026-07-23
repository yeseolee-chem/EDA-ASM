"""spec18 Stage 1 — Section 6.3 distribution sanity vs. Espley et al. ds3.

Their ds3 pickle (`tt_solvent_features.pkl`, n=3510) is not present on
this HPC. The comparison anchor values used here are the literal
statistics recorded in the spec (§6 item 3), taken from the ds3
artifact by the spec author.

For the DIPOLAR-400 cohort this comparison is *meaningfully*
comparable — both sides are [3+2] dipolar cycloadditions. The only
remaining offset is the reference DFT level (ours: ORCA ωB97X-3c
EDA-NOCV; theirs: B3LYP-D3(BJ)/def2-TZVP + SMD). See Deviation #4.

Emits:
  results/ds3_distribution_comparison.csv
  figures/target_hist_3panel.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec18_espley_s1_labels"
IN_PARQUET = STAGE / "results/labels_2ch_400dipolar.parquet"
OUT_CSV = STAGE / "results/ds3_distribution_comparison.csv"
OUT_FIG = STAGE / "figures/target_hist_3panel.png"

# ds3 literals from the spec §6 item 3
DS3 = {
    "e_barrier_dft":               {"n": 3510, "mean": 5.92,  "std": 8.49,  "min": -14.81, "max": 44.65},
    "sum_distortion_energies_dft": {"n": 3510, "mean": 27.13, "std": None,  "min":   2.47, "max": 79.15},
    "interaction_energies_dft":    {"n": 3510, "mean": 21.21, "std": None,  "min":   5.81, "max": 49.36},
}


def _stats(x: np.ndarray) -> dict:
    return {
        "n":    int(x.size),
        "mean": float(np.mean(x)),
        "std":  float(np.std(x, ddof=1)),
        "min":  float(np.min(x)),
        "max":  float(np.max(x)),
    }


def main() -> None:
    df = pd.read_parquet(IN_PARQUET)
    STAGE.joinpath("figures").mkdir(exist_ok=True)

    targets = ["e_barrier_dft", "sum_distortion_energies_dft", "interaction_energies_dft"]

    rows = []
    for t in targets:
        ours = _stats(df[t].values)
        theirs = DS3[t]
        for stat in ["n", "mean", "std", "min", "max"]:
            rows.append({
                "target":  t,
                "stat":    stat,
                "ours":    ours[stat],
                "ds3_ref": theirs[stat],
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV}")
    print(tbl.pivot(index="target", columns="stat", values=["ours", "ds3_ref"]).to_string())

    # sanity escalations
    escalations = []
    for t in targets:
        ours_mean = float(np.mean(df[t].values))
        theirs_mean = DS3[t]["mean"]
        if t == "interaction_energies_dft" and ours_mean < 0:
            escalations.append(f"[SIGN ERROR] {t} mean is negative ({ours_mean:.4f}) — build did not flip sign")
        if theirs_mean != 0:
            r = ours_mean / theirs_mean
            for factor, name in [(627.5, "hartree->kcal"), (4.184, "kcal->kJ")]:
                for probe in [factor, 1.0 / factor]:
                    if 0.9 * probe <= abs(r) <= 1.1 * probe:
                        escalations.append(f"[UNIT SUSPECT] {t} ratio ours/ds3 ~= {r:.3f} ~ {probe:.3f} ({name})")
    if escalations:
        print("\n=== ESCALATIONS ===")
        for e in escalations:
            print(e)
    else:
        print("\n[ok] no unit / sign escalations")

    # 3-panel histogram of ours (ds3 raw not available)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, t in zip(axes, targets):
        vals = df[t].values
        ax.hist(vals, bins=40, color="#3b7dbf", edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.set_xlabel(f"{t}  [kcal/mol]")
        ax.set_ylabel("count")
        ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.5)
        our_mean = float(np.mean(vals))
        ds3_mean = DS3[t]["mean"]
        ax.axvline(our_mean, color="#e07b00", linewidth=1.4,
                   label=f"ours mean = {our_mean:.2f}")
        ax.axvline(ds3_mean, color="#666", linewidth=1.0, linestyle="--",
                   label=f"ds3 mean = {ds3_mean:.2f}")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    print(f"[write] {OUT_FIG}")


if __name__ == "__main__":
    main()
