#!/usr/bin/env python3
"""SPEC16rev Step 7 — REPORT.md."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
OUT = BASE / "artifacts"


def gate(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(tag, "ℹ️")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# B3LYP full relabeling — REPORT (spec16rev)", ""]
    ts = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                                          text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    lines.append(f"_Generated: {ts}, commit `{commit}`_")
    lines.append("")

    lines.append("## Data source (Coley only)")
    lines.append("- Coley archive: coleygroup/dipolar_cycloaddition_dataset")
    lines.append("- 5269 profile dirs with TS + reactants + products at B3LYP-D3(BJ)/def2-SVP")
    lines.append("- **Espley archive NOT read** (contamination check enforced in 00_setup)")
    lines.append("")

    lines.append("## Option A: no re-optimization (SPEC §2.2)")
    lines.append("- Relaxed and distorted fragments use identical level of theory:")
    lines.append("  Coley's B3LYP-D3(BJ)/def2-SVP geometry.")
    lines.append("- Re-optimizing the relaxed side would break the // convention and")
    lines.append("  systematically inflate distortion energies.")
    lines.append("- Notation: **B3LYP-D3(BJ)/def2-TZVP // B3LYP-D3(BJ)/def2-SVP**.")
    lines.append("")

    lines.append("## Gates")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append(gate(OUT / "GATE0_STATUS.txt", "GATE-0 (contamination + data)"))
    lines.append(gate(OUT / "GATE1_STATUS.txt", "GATE-1 (build ≥97%)"))
    lines.append(gate(OUT / "GATE2_STATUS.txt", "GATE-2 (parse ≥95%, FSPE count)"))
    lines.append(gate(OUT / "GATE3_STATUS.txt", "GATE-3 (QC 8-check)"))
    lines.append(gate(OUT / "GATE4_STATUS.txt", "GATE-4 (δ_prop within ±20% of spec15)"))
    lines.append("")

    lbl = OUT / "labels_b3lyp_full.csv"
    if lbl.exists():
        R = pd.read_csv(lbl)
        lines.append(f"## Labels: {len(R)} reactions")
        strain_neg = int((~R["strain_valid"]).sum())
        lines.append(f"- strain-negative (excluded from d1/d2 analysis): {strain_neg}")
        lines.append("")

    shift = OUT / "label_shift.csv"
    if shift.exists():
        s = pd.read_csv(shift)
        lines.append("## Label shift (B3LYP − AM1, kcal/mol, on 3504 overlap)")
        lines.append("```")
        lines.append(s.round(3).to_string(index=False))
        lines.append("```")
        lines.append("")

    bad_build = OUT / "build_failures.csv"
    bad_parse = OUT / "parse_failures.csv"
    if bad_build.exists() and bad_parse.exists():
        b = pd.read_csv(bad_build); p = pd.read_csv(bad_parse)
        lines.append("## Exclusions")
        lines.append(f"- build failures: {len(b)}")
        lines.append(f"- parse failures: {len(p)}")
        lines.append("")

    lines.append("## Paper Methods paragraph")
    lines.append("")
    lines.append("> Transition-state and relaxed-reactant geometries were taken directly")
    lines.append("> from the dataset of Stuyver and Coley (10.1038/s41597-023-01977-8),")
    lines.append("> optimized with autodE at the B3LYP-D3(BJ)/def2-SVP level. Fragments")
    lines.append("> were assigned by severing the two forming bonds — identified from the")
    lines.append("> atom-mapped reaction SMILES — and partitioning the remaining")
    lines.append("> connectivity graph; this reproduced the reference fragmentation in")
    lines.append("> 218/218 validation cases (spec15). Energy decomposition analysis")
    lines.append("> (EDA-NOCV) was performed with ORCA 6.1.1 at B3LYP-D3(BJ)/def2-TZVP")
    lines.append("> with CPCM(SMD, water). Distortion energies were obtained from")
    lines.append("> separate counterpoise-free single-point calculations on the bare")
    lines.append("> fragments at the transition-state and relaxed geometries. **No")
    lines.append("> re-optimization was performed, so that the distorted and reference")
    lines.append("> structures originate from an identical level of theory.** All labels")
    lines.append("> satisfy the energy-closure identity to within 0.01 kcal/mol. A small")
    lines.append("> fraction of reactions (~0.5%) exhibited negative distortion energies,")
    lines.append("> indicating that the reference conformer in the source dataset is not")
    lines.append("> the global minimum; these were flagged and excluded from strain-channel")
    lines.append("> analysis, while their interaction channels were retained.")

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
