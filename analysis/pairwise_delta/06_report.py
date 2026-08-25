#!/usr/bin/env python3
"""Step 6: REPORT.md for pairwise δ direct learning (SPEC12)."""
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta")
OUT = BASE / "artifacts"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LEARNED = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
           "oi_dft", "disp_dft", "cpcm_dft"]


def gate_line(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "CONDITIONAL": "⚠️", "WARN": "⚠️"}.get(tag, "ℹ️")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# Pairwise δ direct learning — REPORT (spec12)", ""]
    ts = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                                          text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    lines.append(f"_Generated: {ts}, commit `{commit}`_")
    lines.append("")

    lines.append("## Gates")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append(gate_line(OUT / "GATE0_STATUS.txt", "GATE-0 (data prep)"))
    lines.append(gate_line(OUT / "GATE1_STATUS.txt", "GATE-1 (split integrity)"))
    lines.append(gate_line(OUT / "GATE2_STATUS.txt", "GATE-2 (Arm-A beats Arm-0 on ≥5/7)"))
    lines.append(gate_line(OUT / "GATE4_STATUS.txt", "GATE-4 (Arm-C verdict per channel)"))
    lines.append(gate_line(OUT / "GATE5_STATUS.txt", "GATE-5 (swaptype generalization)"))
    lines.append("")

    if (OUT / "delta_mae_by_arm.csv").exists():
        agg = pd.read_csv(OUT / "delta_mae_by_arm.csv")
        lines.append("## Main — MAE_δ per (arm, split, channel)")
        lines.append("")
        for split in ["component", "swaptype_holdout"]:
            lines.append(f"### split = **{split}**")
            sub = agg[agg["split"] == split].copy()
            show = sub[["arm", "channel", "mae_delta", "arm0_mae_delta",
                        "improve_vs_arm0", "sign_acc", "spearman", "slope", "verdict"]]
            lines.append("```")
            lines.append(show.to_string(index=False))
            lines.append("```")
            lines.append("")

    if (OUT / "swaptype_generalization.csv").exists():
        gen = pd.read_csv(OUT / "swaptype_generalization.csv")
        lines.append("## H5 — swap-type generalization (swap_holdout / component)")
        lines.append("")
        lines.append("Ratio > 2.0 means the model 'memorized swap types'; unseen swaps double the error.")
        lines.append("```")
        lines.append(gen.to_string(index=False))
        lines.append("```")
        lines.append("")

    if (OUT / "resolution_table.csv").exists():
        res = pd.read_csv(OUT / "resolution_table.csv")
        lines.append("## Resolution — sign accuracy binned by |δ_true|")
        lines.append("")
        for arm in ["A_diff", "B_concat", "C_symmetric"]:
            lines.append(f"### arm = **{arm}** (split=component)")
            piv = res[(res["arm"] == arm) & (res["split"] == "component")].pivot_table(
                index="channel", columns="bin", values="sign_acc", observed=False
            )
            lines.append("```")
            lines.append(piv.round(3).to_string())
            lines.append("```")
            lines.append("")

    if (OUT / "invariant_check.csv").exists():
        inv = pd.read_csv(OUT / "invariant_check.csv")
        lines.append("## Invariant checks (seed=42, fold=0)")
        lines.append("- self_check: predicted δ(A,A) should be ~0")
        lines.append("- antisym: |δ(A,B) + δ(B,A)| should be ~0")
        lines.append("```")
        lines.append(inv.to_string(index=False))
        lines.append("```")
        lines.append("")

    if (OUT / "rho_equiv.csv").exists():
        rho = pd.read_csv(OUT / "rho_equiv.csv")
        lines.append("## ρ_equiv — 'ρ boost' that subtract-baseline would need to match this arm")
        lines.append("```")
        lines.append(rho.to_string(index=False))
        lines.append("```")
        lines.append("")

    lines.append("## Files")
    lines.append("```")
    for p in sorted(OUT.iterdir()):
        lines.append(f"artifacts/{p.name}")
    for p in sorted(FIG.iterdir()):
        lines.append(f"figures/{p.name}")
    lines.append("```")

    # Figure: MAE_δ arm comparison
    if (OUT / "delta_mae_by_arm.csv").exists():
        agg = pd.read_csv(OUT / "delta_mae_by_arm.csv")
        for split in ["component", "swaptype_holdout"]:
            sub = agg[agg["split"] == split]
            if len(sub) == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            arms = ["A_diff", "B_concat", "C_symmetric"]
            x = np.arange(len(LEARNED))
            width = 0.22
            for i, arm in enumerate(arms):
                s = sub[sub["arm"] == arm].set_index("channel").reindex(LEARNED)
                ax.bar(x + (i - 1) * width, s["mae_delta"], width, label=arm)
            arm0 = sub.drop_duplicates("channel").set_index("channel").reindex(LEARNED)
            ax.plot(x, arm0["arm0_mae_delta"], "k--", marker="D",
                    markersize=6, label="Arm-0 (subtract)", linewidth=1.5)
            ax.axhline(1.0, color="r", linestyle=":", label="target 1 kcal/mol")
            ax.set_xticks(x); ax.set_xticklabels(LEARNED, rotation=45, ha="right")
            ax.set_ylabel("MAE_δ (kcal/mol)")
            ax.set_title(f"MAE_δ per channel per arm — split={split}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIG / f"mae_delta_arm_comparison_{split}.png", dpi=120)
            plt.close(fig)

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
