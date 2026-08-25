#!/usr/bin/env python3
"""Step 6: REPORT.md — δ-MAE baseline main deliverable."""
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline")
OUT = BASE / "artifacts"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def gate_line(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "CONDITIONAL": "⚠️", "WARN": "⚠️",
            "REVIEW": "⚠️", "REPORT": "ℹ️"}.get(tag, "ℹ️")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# δ-MAE Baseline — REPORT (spec11)", ""]
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
    lines.append(gate_line(OUT / "GATE0_STATUS.txt", "GATE-0 (data integrity)"))
    lines.append(gate_line(OUT / "GATE1_STATUS.txt", "GATE-1 (absolute MAE within ±10%)"))
    lines.append(gate_line(OUT / "GATE2_STATUS.txt", "GATE-2 (mae_δ < √2·abs_mae)"))
    lines.append(gate_line(OUT / "GATE3_STATUS.txt", "GATE-3 (ρ_pair vs required — informational)"))
    lines.append("")

    dm_path = OUT / "delta_mae_table.csv"
    if dm_path.exists():
        dm = pd.read_csv(dm_path)
        lines.append("## Q — δ-MAE per channel (main deliverable)")
        lines.append("")
        for scheme in dm["scheme"].unique():
            lines.append(f"### scheme = **{scheme}**")
            sub = dm[dm["scheme"] == scheme].copy()
            sub = sub.sort_values("mae_delta")
            show = sub[["channel", "abs_mae", "baseline_mae_delta", "mae_delta",
                        "improvement_over_zero", "sign_accuracy", "spearman_delta",
                        "verdict", "excluded_from_learning"]]
            lines.append("```")
            lines.append(show.to_string(index=False))
            lines.append("```")
            lines.append("")

    rp_path = OUT / "rho_pair.csv"
    if rp_path.exists():
        rp = pd.read_csv(rp_path)
        lines.append("## ρ_pair — pairwise error correlation (strategy decision)")
        lines.append("")
        for scheme in rp["scheme"].unique():
            lines.append(f"### scheme = **{scheme}**")
            sub = rp[rp["scheme"] == scheme].copy()
            show = sub[["channel", "abs_mae_pair", "rho_pair_measured",
                        "predicted_mae_delta", "required_rho_for_1kcal",
                        "gap_actual_minus_required", "strategy_tier"]]
            lines.append("```")
            lines.append(show.to_string(index=False))
            lines.append("```")
            lines.append("")

    lines.append("## Split-scheme comparison (leakage size)")
    if (OUT / "delta_mae_table.csv").exists():
        dm = pd.read_csv(OUT / "delta_mae_table.csv")
        piv = dm.pivot_table(index="channel", columns="scheme", values="mae_delta")
        piv["leakage_ratio"] = piv["kfold_random"] / piv["groupkfold_component"]
        piv["abs_ratio_ref"] = 1.0
        lines.append("```")
        lines.append(piv.to_string())
        lines.append("```")
        lines.append("")
        lines.append("A ratio < 1.0 in `leakage_ratio` means random KFold is optimistic ")
        lines.append("(same connected component leaked between train/test).")
        lines.append("")

    lines.append("## Cross-channel δ-ERROR correlation")
    lines.append("")
    lines.append("> ⚠️ This is **model prediction error** correlation — different from ")
    lines.append("> `mmp_gate_a` §5.4 which was correlation of **actual δ values** (physics).")
    lines.append("> - Actual δ correlation → physics; channels co-move")
    lines.append("> - **Prediction-error correlation → model limitation**; what one channel")
    lines.append(">   gets wrong, another gets wrong the same way")
    lines.append("> Do not conflate.")
    lines.append("")
    for scheme in ["kfold_random", "groupkfold_component"]:
        p = OUT / f"error_corr_pearson_{scheme}.csv"
        if p.exists():
            lines.append(f"### scheme = **{scheme}**")
            cor = pd.read_csv(p, index_col=0)
            lines.append("```")
            lines.append(cor.round(3).to_string())
            lines.append("```")
            cx = OUT / f"cancellation_{scheme}.txt"
            if cx.exists():
                lines.append(cx.read_text().strip())
            lines.append("")

    bd = OUT / "barrier_delta.csv"
    if bd.exists():
        lines.append("## Barrier δ — direct vs channel-sum")
        lines.append("```")
        lines.append(pd.read_csv(bd).to_string(index=False))
        lines.append("```")
        lines.append("")

    lines.append("## Files")
    lines.append("```")
    for p in sorted(OUT.iterdir()):
        lines.append(f"artifacts/{p.name}")
    for p in sorted(FIG.iterdir()):
        lines.append(f"figures/{p.name}")
    lines.append("```")

    # Figures: mae_delta vs target + rho_pair required vs actual
    dm_path = OUT / "delta_mae_table.csv"
    rp_path = OUT / "rho_pair.csv"
    if dm_path.exists():
        dm = pd.read_csv(dm_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        schemes = dm["scheme"].unique()
        chans = dm["channel"].unique()
        width = 0.4
        x = np.arange(len(chans))
        for i, sc in enumerate(schemes):
            sub = dm[dm["scheme"] == sc].set_index("channel").loc[chans]
            ax.bar(x + (i - 0.5) * width, sub["mae_delta"], width, label=sc)
        ax.axhline(1.0, color="r", linestyle="--", label="target 1 kcal/mol")
        ax.set_xticks(x); ax.set_xticklabels(chans, rotation=45, ha="right")
        ax.set_ylabel("MAE_δ (kcal/mol)")
        ax.set_title("δ-MAE per channel vs target")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "mae_delta_vs_target.png", dpi=120)
        plt.close(fig)

    if rp_path.exists():
        rp = pd.read_csv(rp_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        chans = rp["channel"].unique()
        x = np.arange(len(chans))
        for i, sc in enumerate(rp["scheme"].unique()):
            sub = rp[rp["scheme"] == sc].set_index("channel").loc[chans]
            ax.plot(x, sub["rho_pair_measured"], "o-", label=f"actual ({sc})")
            ax.plot(x, sub["required_rho_for_1kcal"], "s--", label=f"required ({sc})", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(chans, rotation=45, ha="right")
        ax.set_ylabel("ρ_pair")
        ax.set_title("ρ_pair: actual vs required for MAE_δ < 1 kcal/mol")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "rho_pair_required_vs_actual.png", dpi=120)
        plt.close(fig)

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
