"""Copy review_log_complete.json into Phase 1's manual_review_log.json and re-run the
final integration stages. Use this once review is complete.

    python tools/phase1_5_review/finalize_phase1.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPLETE = ROOT / "outputs" / "phase1.5" / "review_log_complete.json"
TARGET = ROOT / "outputs" / "phase1" / "manual_review_log.json"


def main() -> int:
    if not COMPLETE.exists():
        print(f"missing {COMPLETE}; complete the review first.", file=sys.stderr)
        return 1
    shutil.copyfile(COMPLETE, TARGET)
    print(f"copied {COMPLETE} -> {TARGET}")

    cmd = [sys.executable, str(ROOT / "scripts" / "run_phase1.py"), "--finalize"]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    return rc


if __name__ == "__main__":
    sys.exit(main())
