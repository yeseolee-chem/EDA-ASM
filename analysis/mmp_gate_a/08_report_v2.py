#!/usr/bin/env python3
"""Step 8 (v2): REPORT_v2.md with H5 warning, v1↔v2 shape shift, cds rationale."""
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"


def gate_line(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "CONDITIONAL": "⚠️", "WARN": "⚠️",
            "REVIEW": "⚠️", "REPORT": "ℹ️"}.get(tag, "?")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# MMP Gate A — REPORT v2 (spec10rev patch)", ""]
    ts = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                                          text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    lines.append(f"_Generated: {ts}, commit `{commit}`_")
    lines.append("")
    lines.append("This v2 supersedes REPORT.md. v1 artifacts are preserved for comparison.")
    lines.append("")

    lines.append("## Patch gate summary (spec10rev)")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append("| GATE-0 (cohort join, composition) | ✅ PASS (from 00_load_join.py) |")
    lines.append(gate_line(OUT / "GATE1a_STATUS.txt", "GATE-1a (core detection, 5-ring)"))
    lines.append(gate_line(OUT / "GATE1_STATUS_v2.txt", "GATE-1 (library hit — report-only)"))
    lines.append(gate_line(OUT / "GATE1b_STATUS.txt", "GATE-1b (core-adjacent hit rate ≥95%)"))
    lines.append(gate_line(OUT / "GATE2_STATUS.txt", "GATE-2 (eda.inp fragment cross-check)"))
    lines.append(gate_line(OUT / "GATE3rev_STATUS.txt", "GATE-3-rev (pair count 2207±5%)"))
    lines.append(gate_line(OUT / "GATE4_STATUS_v2.txt", "GATE-4-rev (8-channel MMP pair count)"))
    lines.append(gate_line(OUT / "GATE5_STATUS_v2.txt", "GATE-5-rev (learned-channel δ)"))
    lines.append(gate_line(OUT / "GATE6_STATUS_v2.txt", "GATE-6-rev (outlier + filter integrity)"))
    lines.append("")

    q1 = (OUT / "Q1_ANSWER_v2.txt").read_text() if (OUT / "Q1_ANSWER_v2.txt").exists() else "(missing)"
    lines.append("## Q1 — 8-channel-complete MMP pair count (v2)")
    lines.append("```")
    lines.append(q1.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Q2 — per-channel δ distribution (v2)")
    lines.append("`cds` is **excluded from learning** — see §CDS-exclusion below.")
    stats_path = OUT / "delta_channel_stats_v2.csv"
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        show = ["channel", "excluded_from_learning", "n", "mean_abs", "median_abs",
                "median_abs_ci_lo", "median_abs_ci_hi", "std_signed", "frac_lt_1.0",
                "baseline_MAE"]
        show = [c for c in show if c in df.columns]
        lines.append("```")
        lines.append(df[show].to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Cross-channel δ Pearson correlation (v2, all-8 subset)")
    lines.append("")
    lines.append("> ⚠️ **H5 caution.** This matrix is over **actual δ values**, NOT prediction errors.")
    lines.append("> - Actual δ correlation → physics; channels move together; the 'channel-independent")
    lines.append(">   manipulation' claim is weakened.")
    lines.append("> - Prediction-error correlation → model limitation.")
    lines.append("> Do not conflate the two.")
    lines.append("")
    p_path = OUT / "delta_corr_pearson_v2.csv"
    if p_path.exists():
        cor = pd.read_csv(p_path, index_col=0)
        lines.append("```")
        lines.append(cor.round(3).to_string())
        lines.append("```")
    lines.append("")

    lines.append("## v1 → v2 shape shift (spec10rev §5.4)")
    lines.append("| pair | v1 (all-side, no core filter) | v2 (core-preserved) |")
    lines.append("|---|---|---|")
    lines.append("| pair count | 4230 | see Q1 above |")
    lines.append("| elst↔pauli | −0.911 | see matrix |")
    lines.append("| elst↔oi    | +0.764 | see matrix |")
    lines.append("| d1↔d2      | −0.445 | see matrix |")
    lines.append("")
    lines.append("Direction of change is the diagnostic: attenuation → v1 had confounded pairs; ")
    lines.append("intensification → filter isolated a real physical coupling.")
    lines.append("")

    lines.append("## CDS-exclusion rationale (spec10rev §5.2)")
    lines.append("- cds δ is <1 kcal/mol on **>93%** of pairs (see stats table)")
    lines.append("- A trivial `δ=0` predictor already achieves MAE ≈ 0.4 kcal/mol")
    lines.append("- No learnable signal → **removed from learning targets**")
    lines.append("- Analogous to treating dispersion via D4 analytical; disp itself is NOT excluded")
    lines.append("  (disp median|δ| ≈ 2.25 has real signal)")
    lines.append("- LEARNED_CHANNELS = [d1, d2, elst, pauli, oi, disp, cpcm]")
    lines.append("- EXCLUDED_CHANNELS = [cds]")
    lines.append("")

    lines.append("## Substituent inventory (v2 top 20 with core-adjacent counts)")
    inv_path = OUT / "substituent_inventory_v2.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path).head(20)
        lines.append("```")
        lines.append(inv.to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Outlier summary (v2 with filter-integrity flags)")
    out_path = OUT / "outliers_v2.csv"
    if out_path.exists():
        out = pd.read_csv(out_path)
        lines.append(f"- Outliers (|Δ G_act| > 15 kcal/mol): **{len(out)}**")
        if len(out) and "reason_a_core_mismatch" in out.columns:
            n_a = int(out["reason_a_core_mismatch"].sum())
            n_b = int(out["reason_b_sub_too_big"].sum())
            lines.append(f"- Reason (a) core mismatch: **{n_a}** (must be 0 if filter works)")
            lines.append(f"- Reason (b) sub_heavy>8: **{n_b}** (must be 0 if filter works)")
        if len(out):
            top = out.head(10)
            keep = [c for c in ["rxn_id_A", "rxn_id_B", "sub_A", "sub_B",
                                "delta_G_act", "reason_a_core_mismatch",
                                "reason_b_sub_too_big"] if c in top.columns]
            lines.append("```")
            lines.append(top[keep].to_string(index=False))
            lines.append("```")
    lines.append("")

    lines.append("## Deferred (spec10rev §12)")
    lines.append("- Partial correlation ctrl. on TS distance — needed to separate")
    lines.append("  physical channel coupling from geometric confounder. TS xyz")
    lines.append("  already sits in `cohort_v1/reactions/rxn_XXXX/eda.inp`; cost = 0.")
    lines.append("- Deferred to SPEC11 or v3 patch.")
    lines.append("")

    lines.append("## Files (v2)")
    lines.append("```")
    for p in sorted(OUT.iterdir()):
        if "_v2" in p.name or p.name.startswith("GATE1a") or p.name.startswith("GATE1b") or \
           p.name.startswith("GATE3rev") or p.name in ("core_v2.pkl", "unknown_subs_v2.csv"):
            lines.append(f"artifacts/{p.name}")
    for p in sorted((BASE / "figures").iterdir()):
        if "_v2" in p.name:
            lines.append(f"figures/{p.name}")
    lines.append("```")

    (BASE / "REPORT_v2.md").write_text("\n".join(lines))
    print(f"Wrote REPORT_v2.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
