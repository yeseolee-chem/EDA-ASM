"""Driver for Stage 3.8 — manual review queue."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.stage_3_8_review_queue import run  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case-c",
        choices=["manual", "exclude"],
        default="manual",
        help="manual: build pages for every Case C; exclude: drop Case C from queue.",
    )
    args = ap.parse_args()
    out = run(case_c_strategy=args.case_c)
    print(f"\nQueued {len(out['queued'])} reactions, wrote {len(out['written'])} review pages")
