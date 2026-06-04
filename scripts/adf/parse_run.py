#!/usr/bin/env python
"""Parse a single ADF EDA run directory and write asr_label.json.

Usage:
    python scripts/adf/parse_run.py <run_dir>

Reads {run_dir}/{fragA_at_TS,fragB_at_TS,eda_TS,fragA_relaxed,fragB_relaxed}.out
and writes {run_dir}/asr_label.json with the 5-channel ASR vector + reconstructed Ea.
Exits 0 on success, 1 on parsing failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from eda_asm.adf import parse_eda_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()

    if not args.run_dir.is_dir():
        sys.stderr.write(f"ERROR: run_dir does not exist: {args.run_dir}\n")
        return 1
    try:
        asr = parse_eda_run(args.run_dir)
    except Exception as e:
        sys.stderr.write(f"ERROR parsing {args.run_dir}: {e}\n")
        traceback.print_exc()
        return 1

    out_path = args.run_dir / "asr_label.json"
    payload = asr.to_dict()
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"ASR vector  : {json.dumps(payload)}")
    print(f"reconstructed Ea  = {asr.Ea_reconstructed_kcal:.3f} kcal/mol")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
