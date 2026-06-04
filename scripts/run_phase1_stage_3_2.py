"""Driver for Stage 3.2 — bond-change pre-computation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.stage_3_2_bond_changes import run  # noqa: E402


if __name__ == "__main__":
    out = run()
    print(f"\nBond-change parquet written to: {out}")
