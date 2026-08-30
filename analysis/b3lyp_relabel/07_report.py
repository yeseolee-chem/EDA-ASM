#!/usr/bin/env python3
"""SPEC15 Step 7 — REPORT.md."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
OUT = BASE / "artifacts"


def gate(path, label):
    if not path.exists():
        return f"| {label} | ⊘ not run |"
    txt = path.read_text().strip().splitlines()[0]
    tag = txt.split()[0]
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(tag, "ℹ️")
    return f"| {label} | {icon} {txt} |"


def main():
    lines = ["# B3LYP relabel validation — REPORT (spec15)", ""]
    ts = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                                          text=True).strip()[:12]
    except Exception:
        commit = "unknown"
    lines.append(f"_Generated: {ts}, commit `{commit}`_")
    lines.append("")

    lines.append("## Setup")
    lines.append("- Goal: validate B3LYP-TS relabeling pipeline on 200-sample batch")
    lines.append("  before scaling to the full 5269 ds3 reactions.")
    lines.append("- 5 SPEs per rxn: complex EDA, frag1/2 distorted (BARE, no ghost),")
    lines.append("  frag1/2 relaxed (at ds3 r*.xyz).")
    lines.append("- All 218 spec14 rxns reused (EDA cached, only fragments new).")
    lines.append("- strain = (dist − rel) × 627.5094740631 (matches existing convention).")
    lines.append("")

    lines.append("## Gates")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append(gate(OUT / "GATE0_STATUS.txt", "GATE-0 (inputs present)"))
    lines.append(gate(OUT / "GATE1_STATUS.txt", "GATE-1 (sample includes all 218)"))
    lines.append(gate(OUT / "GATE2_STATUS.txt", "GATE-2 (inputs built ≥98%)"))
    lines.append(gate(OUT / "GATE3_STATUS.txt", "GATE-3 (parse ≥95%, no sign viol.)"))
    lines.append(gate(OUT / "GATE4_STATUS.txt", "GATE-4 (δ_prop within ±20% of spec14)"))
    lines.append(gate(OUT / "GATE5_STATUS.txt", "GATE-5 (strain validation 4/4)"))
    lines.append("")

    lines.append("## Label shift (AM1 TS → B3LYP TS)")
    p = OUT / "label_shift.csv"
    if p.exists():
        df = pd.read_csv(p)
        lines.append("```")
        lines.append(df.round(3).to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("- `mean_signed` = systematic bias (B3LYP − AM1)")
        lines.append("- `delta_prop` = std × √2 = pair-difference noise floor")
        lines.append("- `pearson_r` = agreement between old and new labels")
    lines.append("")

    lines.append("## Full 5269 run authorization")
    all_gates = ["GATE0_STATUS.txt", "GATE1_STATUS.txt", "GATE2_STATUS.txt",
                 "GATE3_STATUS.txt", "GATE4_STATUS.txt", "GATE5_STATUS.txt"]
    ok = True
    for g in all_gates:
        p = OUT / g
        if not p.exists() or not p.read_text().startswith("PASS"):
            ok = False
    lines.append(f"- All gates PASS: **{'YES — proceed to 5269 run' if ok else 'NO — resolve failing gates first'}**")
    lines.append("")

    lines.append("## Preservation of old (AM1-geom) labels")
    lines.append("- DO NOT delete phase5_dataset_v2.pkl or reactions/rxn_XXXX/ tree.")
    lines.append("- Rename in v2 rollout: `labels_am1geom_v1/` (existing),")
    lines.append("  `labels_b3lypgeom_v2/` (new).")
    lines.append("- README should explicitly note that v1 is preserved for reproducibility")
    lines.append("  and comparison; only v2 is used by training scripts going forward.")
    lines.append("")

    lines.append("## Files")
    for f in sorted(OUT.glob("*")):
        lines.append(f"- `artifacts/{f.name}`")

    (BASE / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote REPORT.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
