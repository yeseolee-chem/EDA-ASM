#!/usr/bin/env amspython
"""Run run_asr_spec.run_one against the Validate/refrag/ alt stage5a tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
for p in (ROOT, ROOT / "ADF_500" / "scripts", ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import run_asr_spec  # type: ignore

run_asr_spec.STAGE5A_DIR = ROOT / "Validate" / "refrag" / "stage5a"
run_asr_spec.OUT_DIR = ROOT / "Validate" / "refrag" / "results"


if __name__ == "__main__":
    run_asr_spec.main()
