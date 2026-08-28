#!/usr/bin/env python3
"""SPEC14 Step 7 — assemble REPORT.md."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
OUT = BASE / "artifacts"


def gate(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "CONDITIONAL": "⚠️", "WARN": "⚠️"}.get(tag, "ℹ️")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# Geometry-level validation — REPORT (spec14)", ""]
    ts = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                                          text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    lines.append(f"_Generated: {ts}, commit `{commit}`_")
    lines.append("")

    lines.append("## Setup")
    lines.append("- Question: how much does the choice of TS geometry level (low-level")
    lines.append("  semi-empirical vs B3LYP-D3BJ/def2-TZVP full DFT) shift the EDA channels?")
    lines.append("- Cheap experiment: 218 EDA single-points at B3LYP-D3BJ/def2-TZVP CPCM(SMD,water)")
    lines.append("  on ds3 (Stuyver/Coley 2023) B3LYP-optimised TS geometries, then compare")
    lines.append("  channel-by-channel against current phase5 labels.")
    lines.append("- Same B3LYP EDA header — differ only in xyz coordinates.")
    lines.append("- strain (d1, d2) excluded: requires fragment re-optimization.")
    lines.append("")

    lines.append("## Gates")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append(gate(OUT / "GATE0_STATUS.txt", "GATE-0 (inputs present)"))
    lines.append(gate(OUT / "GATE1_STATUS.txt", "GATE-1 (218 stratified sample)"))
    lines.append(gate(OUT / "GATE2_STATUS.txt", "GATE-2 (eda.inp built 218/218)"))
    lines.append(gate(OUT / "GATE3_STATUS.txt", "GATE-3 (ORCA normal termination ≥95%)"))
    lines.append(gate(OUT / "GATE4_STATUS.txt", "GATE-4 (channel δ-propagation verdict)"))
    lines.append("")

    lines.append("## Main — channel shift table (B3LYP TS − low-level TS)")
    cpath = OUT / "channel_comparison.csv"
    if cpath.exists():
        df = pd.read_csv(cpath)
        show = df[["channel", "n", "mean_signed", "mae", "median_abs",
                   "p95_abs", "max_abs", "rel_to_scale", "delta_propagated", "verdict"]]
        lines.append("```")
        lines.append(show.round(3).to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("- `mae` = single-reaction absolute channel shift.")
        lines.append("- `delta_propagated` = mae × √2 (independent-noise assumption for δ = B − A).")
        lines.append("- Verdict scale: PASS if δ<1, CONDITIONAL if 1≤δ<3, FAIL if δ≥3.")
    lines.append("")

    lines.append("## Formed-bond length shifts")
    bpath = OUT / "bond_length_shifts.csv"
    if bpath.exists():
        bl = pd.read_csv(bpath)
        lines.append(f"- Sample: n = {len(bl)}")
        for c in ["d1_diff", "d2_diff"]:
            v = bl[c].astype(float)
            lines.append(f"  - {c.replace('_diff','')}: mean = {v.mean():+.3f} Å,  "
                         f"|diff| median = {v.abs().median():.3f},  p95 = {v.abs().quantile(0.95):.3f}")
        lines.append("- If |Δ formed-bond| correlates with |Δ channel|, the channel shift")
        lines.append("  is attributable to geometry (not to solvent / SCF noise).")
    lines.append("")

    lines.append("## Cohort-bias caveat")
    lines.append("- The stratified sample includes only 9 CCNNN and 11 CCNNO reactions")
    lines.append("  (multi-N dipoles are rare in the 3504 cohort). Any claim specific")
    lines.append("  to these cores has low statistical weight; note in the paper.")
    lines.append("")

    lines.append("## What this SPEC decides")
    lines.append("- All PASS   → keep current labels; the 1 kcal/mol δ-MAE target is meaningful.")
    lines.append("  Also expand training set with 960 fresh reactions (cost-justified).")
    lines.append("- Mixed     → per-channel target revision; document geometry limit in paper.")
    lines.append("- Any FAIL  → recompute 3504 labels at B3LYP-D3BJ TS geometry (EDA SPE")
    lines.append("  only; no TS re-optimization needed since ds3 provides B3LYP TS).")
    lines.append("- Either way → paper Methods must state the low-level TS + B3LYP SPE two-step")
    lines.append("  protocol explicitly and cite this validation.")
    lines.append("")

    lines.append("## Figures")
    for name in ("fig1_geometry_error.png", "fig2_parity_by_channel.png",
                 "fig3_bondlen_vs_channel.png"):
        p = BASE / "figures" / name
        if p.exists():
            lines.append(f"- `figures/{name}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("- Header of every built eda.inp is byte-identical to the existing")
    lines.append("  reactions/rxn_XXXX/eda.inp header. Only xyz coordinates differ.")
    lines.append("- Channel parser reuses regexes from `scripts/parse_orca_5channel.py`")
    lines.append("  (Pauli combined with Delta E^0(XC) per project convention).")
    lines.append("- CPCM / CDS regex patterns verified on real output on execution day;")
    lines.append("  see `artifacts/parser_notes.md` if reconciliation was needed.")

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
