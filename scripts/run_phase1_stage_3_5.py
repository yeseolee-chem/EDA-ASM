"""Driver for Stage 3.5 — Case A/B/C classification."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.stage_3_5_classify import run  # noqa: E402


if __name__ == "__main__":
    out = run()
    print("\nCase counts:", out["counts"])
