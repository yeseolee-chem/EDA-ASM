#!/usr/bin/env python3
"""SPEC14 Step 0 — environment + input existence check.

GATE-0: ds3 profiles present (5269 dirs), phase5 cohort present (3504),
missing set = 0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
DS3 = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")

for d in ["artifacts", "inputs", "outputs", "figures", "logs"]:
    (BASE / d).mkdir(parents=True, exist_ok=True)

prof_root = DS3 / "extracted/full_dataset_profiles"
assert prof_root.exists(), f"MISSING ds3 profiles root: {prof_root}"
dirs = [d for d in prof_root.iterdir() if d.is_dir() and d.name.isdigit()]
print(f"ds3 profile dirs: {len(dirs)}")
assert len(dirs) == 5269, f"expected 5269 profile dirs, got {len(dirs)}"

labels = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
labels["reaction_number"] = labels["reaction_number"].astype(int)
print(f"cohort labels    : {labels.shape}")
assert len(labels) == 3504, f"expected 3504 cohort rows, got {len(labels)}"

have = {int(d.name) for d in dirs}
missing = set(labels["reaction_number"]) - have
print(f"cohort rxns absent from ds3 profiles: {len(missing)}")

status = "PASS" if len(missing) == 0 else "FAIL"
with open(BASE / "artifacts" / "GATE0_STATUS.txt", "w") as f:
    f.write(f"{status} n_profiles={len(dirs)} n_cohort={len(labels)} missing={len(missing)}\n")
    if missing:
        f.write(f"first missing ids: {sorted(missing)[:20]}\n")

print(f"python  : {sys.version.split()[0]}")
print(f"pandas  : {pd.__version__}")
print(f"numpy   : {np.__version__}")
print(f"=== GATE-0 {status} ===")
if missing:
    raise SystemExit(1)
