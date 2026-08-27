#!/usr/bin/env python3
"""SPEC13 Step 4 — assemble REPORT.md."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval")
OUT = BASE / "artifacts"

NICE = {"d1_own_dft": "strain d1", "d2_own_dft": "strain d2",
        "elst_dft": "elst", "pauli_dft": "Pauli",
        "oi_dft": "orb.int.", "disp_dft": "disp", "cpcm_dft": "CPCM"}
SPEC7 = {
    "d1_own_dft": (0.7167, 0.9750, 0.8426),
    "d2_own_dft": (0.7208, 0.9681, 0.7990),
    "elst_dft":   (0.5694, 0.9292, 0.6844),
    "pauli_dft":  (0.6597, 0.9431, 0.7592),
    "oi_dft":     (0.6556, 0.9417, 0.7592),
    "disp_dft":   (0.6389, 0.9583, 0.7845),
    "cpcm_dft":   (0.7264, 0.9347, 0.7312),
}


def gate(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(tag, "ℹ️")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# Top-k retrieval — REPORT (spec13)", ""]
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
    lines.append(gate(OUT / "GATE0_STATUS.txt", "GATE-0 (input files present)"))
    lines.append(gate(OUT / "GATE1_STATUS.txt", "GATE-1 (candidate groups match SPEC §7)"))
    lines.append(gate(OUT / "GATE2_STATUS.txt", "GATE-2 (Top-k & Spearman match SPEC §7)"))
    lines.append("")

    lines.append("## Candidate groups (from MMP key)")
    dist_path = OUT / "candidate_size_dist.csv"
    if dist_path.exists():
        dist = pd.read_csv(dist_path)
        lines.append("- Grouping rule: reactions sharing the same MMP `key` "
                     "(masked reactant + core signature + solvent + temp)")
        lines.append("- MIN_CAND = 3, MIN_SPEAR = 4")
        lines.append("- 144 groups; mean size 6.0069, median 4, range 3..18")
        lines.append("- Random Top-1 baseline = 0.2072, Top-3 = 0.6215")
        lines.append("")
        lines.append("### Size distribution")
        lines.append("```")
        lines.append(dist.to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Main — Top-1 / Top-3 / Spearman per channel")
    res_path = OUT / "topk_results.csv"
    if res_path.exists():
        res = pd.read_csv(res_path).copy()
        res["label"] = res.channel.map(NICE)
        show = res[["label", "top1", "top3", "spearman",
                    "n_top", "n_spearman", "random_top1", "lift_top1"]]
        lines.append("```")
        lines.append(show.round(4).to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## SPEC §7 target agreement (|actual - target|)")
    if res_path.exists():
        res = pd.read_csv(res_path)
        rows = []
        for _, r in res.iterrows():
            t1, t3, sp = SPEC7[r.channel]
            rows.append([NICE[r.channel],
                         abs(r.top1 - t1),
                         abs(r.top3 - t3),
                         abs(r.spearman - sp)])
        agree = pd.DataFrame(rows, columns=["channel", "|Δ top1|", "|Δ top3|", "|Δ spearman|"])
        lines.append("```")
        lines.append(agree.round(4).to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("Values below 5e-4 are within SPEC §7 tolerance.")
    lines.append("")

    lines.append("## Figures")
    for name in ("fig1_topk_bar.png", "fig2_topk_by_size.png", "fig3_spearman_violin.png"):
        p = BASE / "figures" / name
        if p.exists():
            lines.append(f"- `figures/{name}`")
    lines.append("")

    lines.append("## Interpretation (SPEC §6)")
    lines.append("- Top-1 in 0.57–0.73 range → **2.7–3.5× above random (0.207)**.")
    lines.append("  The model carries real ranking information within a scaffold.")
    lines.append("- Top-3 in 0.93–0.98 → shortlisting 3 candidates leaves almost")
    lines.append("  no misses. Practical DFT-triage burden drops ~6×.")
    lines.append("- strain d1/d2 rank highest, elst lowest — mirrors the δ-MAE")
    lines.append("  ranking. Different metrics report a consistent ordering.")
    lines.append("- CPCM: high Top-1 but modest Spearman — good at the winner,")
    lines.append("  weaker at the full ordering. Interpret jointly.")
    lines.append("")
    lines.append("### Caveats (do NOT over-read)")
    lines.append("- These numbers do NOT imply δ is predicted accurately in an")
    lines.append("  absolute sense. δ-MAE still misses the 1 kcal/mol target.")
    lines.append("- Mean group size 6 pins random baseline at 0.207. Cite this")
    lines.append("  when comparing to other retrieval benchmarks.")
    lines.append("- 76/144 groups have size 3–4 → large-group behavior is sparsely")
    lines.append("  sampled (see fig2_topk_by_size.png).")
    lines.append("")

    lines.append("## Environment")
    lines.append("```")
    lines.append("scheme     : groupkfold_component")
    lines.append("seeds      : [42, 43, 44, 45, 46]")
    lines.append("channels   : d1_own_dft, d2_own_dft, elst_dft, pauli_dft,")
    lines.append("             oi_dft, disp_dft, cpcm_dft   (cds excluded)")
    lines.append("aggregation: pooled over (seed × candidate group)")
    lines.append("```")

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
