#!/usr/bin/env python3
"""SPEC15 Step 0 — input existence + output dirs.

GATE-0: ds3 profiles present (5269), spec14's results_218 present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
GEOM = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
# ds3 profiles live on gpfs scratch (spec14 didn't extract into its dir)
PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")

for d in ["artifacts", "inputs", "outputs", "figures", "logs"]:
    (BASE / d).mkdir(parents=True, exist_ok=True)

assert PROF.exists(), f"MISSING ds3 profiles at {PROF} — run spec14 setup first"
dirs = [d for d in PROF.iterdir() if d.is_dir() and d.name.isdigit()]
print(f"ds3 profiles: {len(dirs)}")
assert len(dirs) == 5269, f"expected 5269, got {len(dirs)}"

r218 = GEOM / "results_218"
done218 = sorted(d.name for d in r218.iterdir() if d.is_dir()) if r218.exists() else []
print(f"spec14 results_218 rxn dirs: {len(done218)}")

labels = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
print(f"phase5 cohort: {labels.shape}")

with open(BASE / "artifacts" / "GATE0_STATUS.txt", "w") as f:
    f.write(f"PASS profiles={len(dirs)} results_218={len(done218)} cohort={len(labels)}\n")
print(f"python {sys.version.split()[0]}  pandas {pd.__version__}  numpy {np.__version__}")
print("=== GATE-0 PASS ===")
