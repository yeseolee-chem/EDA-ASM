"""Driver for Stages 3.6 & 3.7 — auto fragment definition for Cases A and B."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.stage_3_6_frag_caseA import run as run_a  # noqa: E402
from eda_asm.phase1.stage_3_7_frag_caseB import run as run_b  # noqa: E402


if __name__ == "__main__":
    a = run_a()
    print(f"Stage 3.6 done: processed={a['processed']}, skipped={len(a['skipped'])}")
    b = run_b()
    print(f"Stage 3.7 done: processed={b['processed']}, reclassified to C={len(b['reclassified'])}")
