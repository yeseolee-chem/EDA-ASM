"""Driver for Stages 3.9 & 3.10 — final integration + HDF5 export."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.stage_3_9_final_fragments import run as run_final  # noqa: E402
from eda_asm.phase1.stage_3_10_h5 import run as run_h5  # noqa: E402


if __name__ == "__main__":
    res = run_final()
    print(f"Stage 3.9 done: final entries={len(res['final'])}, rejected={len(res['rejected'])}")
    out = run_h5()
    print(f"Stage 3.10 done: HDF5 written to {out}")
