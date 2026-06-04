#!/usr/bin/env python
"""Print a roll-up of ADF run statuses per ASR_ADF_Computation_Spec_v1.0 §10.3.

Counts status.json files under adf_outputs/batch_*/{rid}/ and reports
N converged / N failed / N pending, both overall and per batch.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adf-root", type=Path, default=REPO / "adf_outputs")
    args = p.parse_args()

    overall = Counter()
    per_batch: dict[str, Counter] = {}
    for batch_dir in sorted(args.adf_root.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        b = batch_dir.name
        per_batch[b] = Counter()
        rxn_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
        for rxn_dir in rxn_dirs:
            status = rxn_dir / "status.json"
            if not status.is_file():
                # check if it's running (results dirs present but no status yet)
                if any(rxn_dir.glob("*.results")):
                    per_batch[b]["running_or_failed_no_status"] += 1
                else:
                    per_batch[b]["pending"] += 1
                continue
            try:
                s = json.loads(status.read_text())
            except Exception:
                per_batch[b]["unparseable_status"] += 1
                continue
            if s.get("exit_code", 1) == 0:
                per_batch[b]["converged"] += 1
            else:
                per_batch[b]["failed"] += 1
        overall.update(per_batch[b])

    print(f"=== overall (across {len(per_batch)} batches) ===")
    total = sum(overall.values())
    for k, v in overall.most_common():
        print(f"  {k:30s} {v:>4}/{total}")
    print()
    print("=== per batch ===")
    for b in sorted(per_batch):
        n = sum(per_batch[b].values())
        line = "  ".join(f"{k}={v}" for k, v in per_batch[b].most_common())
        print(f"  {b}: total={n}  {line}")


if __name__ == "__main__":
    main()
