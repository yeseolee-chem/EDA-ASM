#!/usr/bin/env python3
"""SPEC13 Step 0 — environment check + input existence + output dirs.

GATE-0: both input files from delta_mae_baseline must exist.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval")
SRC = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts")

(BASE / "artifacts").mkdir(parents=True, exist_ok=True)
(BASE / "figures").mkdir(parents=True, exist_ok=True)

for f in ["oof_predictions.pkl", "pairs_dedup.pkl"]:
    p = SRC / f
    assert p.exists(), f"MISSING input: {p}. Run delta_mae_baseline first."

print("python  :", sys.version.split()[0])
print("pandas  :", pd.__version__)
print("numpy   :", np.__version__)
print("scipy   :", scipy.__version__)
print("mpl     :", matplotlib.__version__)
print("=== GATE-0 PASS ===")

with open(BASE / "artifacts" / "GATE0_STATUS.txt", "w") as f:
    f.write("PASS inputs=[oof_predictions.pkl, pairs_dedup.pkl] both present\n")
