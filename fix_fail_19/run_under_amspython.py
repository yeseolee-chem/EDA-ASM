#!/usr/bin/env amspython
"""Single-reaction worker for fix_fail_19 group A or B.

Spawn one of these per reaction (xargs -P 2 → low gate1 load). Each call
processes exactly one reaction_id from the appropriate queue file.

Usage:
  amspython -m fix_fail_19.run_under_amspython --group A --rxn_id <RID> \
            --halo8-dir <halo8_dir> --out-dir <out_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fix_fail_19.config import Config


def _find_entry(queue_path: Path, rid: str) -> dict | None:
    """Return the queue entry whose reaction_id matches rid."""
    q = json.loads(queue_path.read_text())
    for e in q:
        if e["reaction_id"] == rid:
            return e
    return None


def main() -> int:
    """Process a single reaction under amspython."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["A", "B"], required=True)
    ap.add_argument("--rxn_id", required=True)
    ap.add_argument("--queue", type=Path, default=None,
                     help="Override queue path; default work_fix_fail_19/queue_<G>.json")
    ap.add_argument("--halo8-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    queue_path = args.queue or (args.out_dir / f"queue_{args.group}.json")
    entry = _find_entry(queue_path, args.rxn_id)
    if entry is None:
        sys.stderr.write(f"rxn_id {args.rxn_id} not in {queue_path}\n")
        return 1

    cfg = Config()
    if args.group == "A":
        from fix_fail_19.group_a_reendpoint import process_one
        r = process_one(entry, args.halo8_dir, args.out_dir, cfg)
    else:
        from fix_fail_19.group_b_spinsweep import process_one
        r = process_one(entry, args.halo8_dir, args.out_dir, cfg)
    print(f"[{args.group}] {args.rxn_id}: {r.get('new_verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
