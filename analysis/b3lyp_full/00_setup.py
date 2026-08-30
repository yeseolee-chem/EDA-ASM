#!/usr/bin/env python3
"""SPEC16rev Step 0 — contamination check + Coley data verification.

GATE-0: no Espley/AM1 paths referenced in scripts (except 05_compare.py),
Coley archive has 5269 profile dirs matching the CSV.
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
# Reuse existing gpfs extraction (verified in spec14/15)
COLEY_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition")
PROF_ROOT = COLEY_ROOT / "extracted/full_dataset_profiles"
CSV_PATH = COLEY_ROOT / "full_dataset.csv"

FORBIDDEN = [
    "bath1480_probe/experiments/cohort_v1/reactions",
    "manifest.parquet",
    "am1_ts",
    "tt_eda_kisti_results",
]
ALLOWED_IN = {"05_compare.py"}


def main():
    for d in ["artifacts", "inputs", "logs", "figures"]:
        (BASE / d).mkdir(parents=True, exist_ok=True)

    # ---- Contamination check ----
    print("=== contamination check ===")
    violations = []
    for py in sorted(BASE.glob("*.py")):
        if py.name in ALLOWED_IN:
            continue
        txt = py.read_text()
        for f in FORBIDDEN:
            if f in txt:
                violations.append(f"  {py.name}: contains '{f}'")
    if violations:
        print("Forbidden path references found:")
        for v in violations:
            print(v)
        with open(BASE / "artifacts" / "GATE0_STATUS.txt", "w") as fh:
            fh.write(f"FAIL contamination_violations={len(violations)}\n")
            for v in violations:
                fh.write(v + "\n")
        raise SystemExit(1)
    print("  no forbidden references")

    # ---- Coley archive ----
    assert PROF_ROOT.exists(), f"MISSING Coley profiles at {PROF_ROOT}"
    dirs = sorted(d for d in PROF_ROOT.iterdir() if d.is_dir() and d.name.isdigit())
    print(f"Coley profile dirs: {len(dirs)}")
    assert len(dirs) == 5269, f"expected 5269, got {len(dirs)}"

    csv = pd.read_csv(CSV_PATH)
    assert len(csv) == 5269, f"CSV expected 5269, got {len(csv)}"
    csv["rxn_id"] = csv["rxn_id"].astype(int)
    missing = set(csv["rxn_id"]) - {int(d.name) for d in dirs}
    print(f"CSV rxn_ids missing from profiles: {len(missing)}")

    with open(BASE / "artifacts" / "GATE0_STATUS.txt", "w") as fh:
        fh.write(f"PASS profiles={len(dirs)} csv={len(csv)} missing={len(missing)}\n")

    print(f"python {sys.version.split()[0]}  pandas {pd.__version__}")
    print("=== GATE-0 PASS ===")


if __name__ == "__main__":
    main()
