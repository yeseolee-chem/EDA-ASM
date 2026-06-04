"""Driver for Stage 3.4 — 5-point extraction."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.stage_3_4_five_points import run  # noqa: E402


if __name__ == "__main__":
    tmp_dir, written = run()
    print(f"\nWrote {len(written)} npz bundles to: {tmp_dir}")
