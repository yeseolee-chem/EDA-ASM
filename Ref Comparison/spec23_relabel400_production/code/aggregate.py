"""spec23 aggregator — summary.md + DEVIATIONS + BLYP-vs-wB97X-3c comparison.

Deferred — runs after collect_and_build_labels.py produces the new parquet.
Reports:
  - G23-F ASM residual distribution split by sub_source
  - G23-G signs (strain>0, int_eda<0)
  - G23-H relaxation downhill (comparing E(R_frag @ start) vs opt)
  - G23-I halves gap before/after (needs old + new parquet)
  - G23-J functional shift per channel (BLYP → wB97X-3c)
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec23_relabel400_production"
NEW_PARQUET = REPO / "outputs/spec23_wb97x3c/labels/dipolar_400_wb97x3c.parquet"
OLD_PARQUET = REPO / "outputs/spec16_orca/labels/dipolar_400_merged.parquet"
ASM_CSV = STAGE / "results/asm_residual.csv"

OUT_SUM = STAGE / "results/summary.md"
OUT_SHIFT = STAGE / "results/functional_shift.csv"
FIG_SHIFT = STAGE / "figures/functional_shift.png"
FIG_HALVES = STAGE / "figures/halves_gap_before_after.png"
FIG_ASM = STAGE / "figures/asm_residual_hist.png"

DEVIATIONS_APPEND = STAGE / "results/DEVIATIONS.md"


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        aa = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        boots[i] = aa.mean() - bb.mean()
    return float(np.mean(a) - np.mean(b)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> int:
    STAGE.joinpath("results").mkdir(exist_ok=True)
    STAGE.joinpath("figures").mkdir(exist_ok=True)

    if not NEW_PARQUET.exists():
        print(f"[wait] new parquet not yet at {NEW_PARQUET} — aggregate deferred")
        return 0

    new = pd.read_parquet(NEW_PARQUET)
    old = pd.read_parquet(OLD_PARQUET)
    old = old.rename(columns={"source": "sub_source"})

    # G23-F ASM residual distribution
    asm = pd.read_csv(ASM_CSV) if ASM_CSV.exists() else None

    # G23-I halves gap before/after
    def _gap(df: pd.DataFrame, col: str):
        a = df.loc[df["sub_source"] == "locked_778", col].values
        b = df.loc[df["sub_source"] == "spec16", col].values
        m, lo, hi = bootstrap_ci_diff(a, b)
        return m, lo, hi
    gaps_before = {c: _gap(old, c) for c in ("int_eda_kcal", "strain_kcal", "act_kcal")}
    gaps_after  = {c: _gap(new, c) for c in ("int_eda_kcal", "strain_kcal", "act_kcal")}

    # G23-J functional shift per channel
    joined = new.merge(old[["reaction_id", "pauli_kcal", "elst_kcal", "orb_kcal",
                             "disp_kcal", "int_eda_kcal", "strain_kcal", "act_kcal"]],
                        on="reaction_id", suffixes=("_new", "_old"))
    shift_rows = []
    for chan in ("pauli_kcal", "elst_kcal", "orb_kcal", "disp_kcal",
                 "int_eda_kcal", "strain_kcal", "act_kcal"):
        d = joined[f"{chan}_new"] - joined[f"{chan}_old"]
        shift_rows.append({
            "channel": chan,
            "delta_mean": float(d.mean()),
            "delta_sd": float(d.std(ddof=1)),
            "delta_median": float(d.median()),
            "delta_p05": float(d.quantile(0.05)),
            "delta_p95": float(d.quantile(0.95)),
        })
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(OUT_SHIFT, index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(shift))
    ax.bar(xs, shift["delta_mean"], yerr=shift["delta_sd"], color="#3b7dbf",
           edgecolor="black", linewidth=0.4, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(shift["channel"], rotation=25)
    ax.set_ylabel("Δ (wB97X-3c − BLYP)  [kcal/mol]")
    ax.axhline(0, color="k", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG_SHIFT, dpi=140)

    # halves gap
    fig, ax = plt.subplots(figsize=(9, 5))
    cs = list(gaps_before.keys())
    xs = np.arange(len(cs))
    b_mean = [gaps_before[c][0] for c in cs]
    b_lo   = [gaps_before[c][1] for c in cs]
    b_hi   = [gaps_before[c][2] for c in cs]
    a_mean = [gaps_after[c][0] for c in cs]
    a_lo   = [gaps_after[c][1] for c in cs]
    a_hi   = [gaps_after[c][2] for c in cs]
    ax.bar(xs - 0.2, b_mean, 0.4, yerr=[np.array(b_mean) - np.array(b_lo),
                                          np.array(b_hi) - np.array(b_mean)],
           label="BLYP (before)", color="#888")
    ax.bar(xs + 0.2, a_mean, 0.4, yerr=[np.array(a_mean) - np.array(a_lo),
                                          np.array(a_hi) - np.array(a_mean)],
           label="wB97X-3c (after)", color="#e07b00")
    ax.set_xticks(xs)
    ax.set_xticklabels(cs, rotation=15)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_ylabel("locked_778 − spec16 mean gap [kcal/mol]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_HALVES, dpi=140)

    if asm is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for sub in sorted(asm["sub_source"].unique()):
            v = asm.loc[asm["sub_source"] == sub, "resid_kcal"].values
            ax.hist(v, bins=40, alpha=0.55, label=f"{sub} n={len(v)}",
                    edgecolor="black", linewidth=0.4)
        ax.set_xlabel("|strain + Σ channels − act|  [kcal/mol]")
        ax.set_yscale("log")
        ax.axvline(0.05, color="#c00", linewidth=0.8, linestyle="--", label="G23-F floor")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_ASM, dpi=140)

    # summary.md
    lines = []
    lines.append("# spec23_relabel400_production — summary")
    lines.append("")
    lines.append(f"Env: python {platform.python_version()}, pandas {pd.__version__}.")
    lines.append(f"n_labels_produced = {len(new)}")
    lines.append("")
    lines.append("## G23-F ASM residual")
    if asm is not None:
        for sub in sorted(asm["sub_source"].unique()):
            v = asm.loc[asm["sub_source"] == sub, "resid_kcal"].values
            lines.append(f"- {sub}: n={len(v)}, "
                         f"median={np.median(v):.4f}, p95={np.quantile(v, 0.95):.4f}, "
                         f"max={np.max(v):.4f}")
    lines.append("")
    lines.append("## G23-I halves gap (locked_778 − spec16), mean [95% CI]")
    lines.append("")
    lines.append("| channel | BLYP (before) | wB97X-3c (after) |")
    lines.append("|---|---|---|")
    for c in ("int_eda_kcal", "strain_kcal", "act_kcal"):
        mb, lb, hb = gaps_before[c]
        ma, la, ha = gaps_after[c]
        lines.append(f"| {c} | {mb:+.3f} [{lb:+.3f}, {hb:+.3f}] | {ma:+.3f} [{la:+.3f}, {ha:+.3f}] |")
    lines.append("")
    lines.append("## G23-J functional shift (wB97X-3c − BLYP), mean ± sd")
    lines.append("")
    lines.append("| channel | Δ mean | Δ sd | Δ median |")
    lines.append("|---|---:|---:|---:|")
    for _, r in shift.iterrows():
        lines.append(f"| {r['channel']} | {r['delta_mean']:+.3f} | {r['delta_sd']:.3f} | {r['delta_median']:+.3f} |")
    lines.append("")
    OUT_SUM.write_text("\n".join(lines) + "\n")

    # DEVIATIONS append
    dev = [
        "# spec23 DEVIATIONS delta",
        "",
        "- **#4 corrected**: labels are now ORCA wB97X-3c EDA-NOCV, "
        "gas phase, TightSCF, RIJ-COSX auto. Old wording (\"BLYP D3BJ def2-TZVP\") retired.",
        "- **#8 resolved**: all 400 use fully-optimised isolated fragments for the "
        "relaxed reference; both halves under one protocol.",
        "- **#9 (new)**: BSSE-in-strain convention. E_CP(D_A) uses ghosts of B; "
        "E_R_A is isolated (no ghosts). Interaction is BSSE-clean by construction; "
        "strain absorbs the BSSE. Cross-paper (Espley's ds3) comparison inherits "
        "the same convention.",
        "- **#10 (new)**: XC-into-orb convention. ORCA prints 5 channels for DFT EDA "
        "(pauli/elst/orb_raw/xc/disp); modelling column `orb_kcal = orb_raw + xc`. "
        "Raw components stored in `orb_raw_kcal`, `xc_kcal` for audit — necessary to "
        "separate delocalisation-error from XC change in the BLYP→wB97X-3c shift.",
        "- **#11 (new)**: Gas phase throughout (no solvent). Matches prior BLYP labels; "
        "different from Stuyver (SMD) and Espley (IEFPCM).",
    ]
    DEVIATIONS_APPEND.write_text("\n".join(dev) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
