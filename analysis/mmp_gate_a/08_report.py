#!/usr/bin/env python3
"""Step 8: assemble REPORT.md."""
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"


def gate_line(path: Path, label: str) -> str:
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "CONDITIONAL": "⚠️", "WARN": "⚠️", "REVIEW": "⚠️"}.get(tag, "?")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# MMP Gate A — Report", ""]
    ts = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                                          text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    lines.append(f"_Generated: {ts}, commit `{commit}`_")
    lines.append("")

    lines.append("## Gate summary")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append("| GATE-0 (cohort join, composition) | ✅ PASS (asserted in 00_load_join.py) |")
    lines.append(gate_line(OUT / "GATE1_STATUS.txt", "GATE-1 (known-library hit rate)"))
    lines.append(gate_line(OUT / "GATE2_STATUS.txt", "GATE-2 (eda.inp fragment cross-check)"))
    lines.append(gate_line(OUT / "GATE4_STATUS.txt", "GATE-4 (8-channel MMP pair count)"))
    lines.append(gate_line(OUT / "GATE5_STATUS.txt", "GATE-5 (channel δ distribution)"))
    lines.append(gate_line(OUT / "GATE6_STATUS.txt", "GATE-6 (outlier fraction)"))
    lines.append("")

    q1 = (OUT / "Q1_ANSWER.txt").read_text() if (OUT / "Q1_ANSWER.txt").exists() else "(missing)"
    lines.append("## Q1 — 8-channel-complete MMP pair count")
    lines.append("```")
    lines.append(q1.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Q2 — per-channel δ distribution")
    stats_path = OUT / "delta_channel_stats.csv"
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        show_cols = ["channel", "n", "mean_abs", "median_abs", "median_abs_ci_lo", "median_abs_ci_hi",
                     "std_signed", "frac_lt_1.0", "baseline_MAE"]
        show_cols = [c for c in show_cols if c in df.columns]
        lines.append("```")
        lines.append(df[show_cols].to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Cross-channel δ Pearson correlation (all-8 subset)")
    p_path = OUT / "delta_corr_pearson.csv"
    if p_path.exists():
        cor = pd.read_csv(p_path, index_col=0)
        lines.append("```")
        lines.append(cor.round(3).to_string())
        lines.append("```")
    lines.append("")

    lines.append("## Channel coverage (labels_v1 vs 3504 cohort)")
    cov_path = OUT / "channel_coverage.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        lines.append("```")
        lines.append(cov.to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Substituent inventory (top 20)")
    inv_path = OUT / "substituent_inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path).head(20)
        lines.append("```")
        lines.append(inv.to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Outlier summary")
    out_path = OUT / "outliers.csv"
    if out_path.exists():
        out = pd.read_csv(out_path)
        lines.append(f"- Outliers (|Δ G_act| > 15 kcal/mol): **{len(out)}**")
        if len(out):
            top = out.head(10)
            lines.append("```")
            keep = [c for c in ["rxn_id_A", "rxn_id_B", "sub_A", "sub_B", "delta_G_act"] if c in top.columns]
            lines.append(top[keep].to_string(index=False))
            lines.append("```")
    lines.append("")

    lines.append("## Files")
    lines.append("```")
    for p in sorted(OUT.iterdir()):
        lines.append(f"artifacts/{p.name}")
    for p in sorted((BASE / "figures").iterdir()):
        lines.append(f"figures/{p.name}")
    lines.append("```")

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
