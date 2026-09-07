#!/usr/bin/env python3
"""Validate the 5-layer build reproduces the reference recovery-method assignment.

Reads build_strategy/artifacts/input_meta.csv (produced by build_inputs.py)
and compares against expected_recovery.json.

Exits 0 on match, 1 on divergence.
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    meta_path = BASE / "artifacts" / "input_meta.csv"
    exp_path = BASE / "expected_recovery.json"

    if not meta_path.is_file():
        print(f"FAIL: {meta_path} missing — run build_inputs.py first")
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    expected = json.loads(exp_path.read_text())

    if len(meta) != expected["total_expected"]:
        print(f"FAIL: built {len(meta)}, expected {expected['total_expected']}")
        sys.exit(1)

    got_totals = meta["recovery_method"].value_counts().to_dict()
    for method, exp_n in expected["method_totals"].items():
        got_n = got_totals.get(method, 0)
        if got_n != exp_n:
            print(f"FAIL: method '{method}' — got {got_n}, expected {exp_n}")
            print(f"  full got: {got_totals}")
            sys.exit(1)

    method_by_rxn = dict(zip(meta["rxn_id"].astype(int), meta["recovery_method"]))
    for rid_s, exp_method in expected["non_primary_rxns"].items():
        rid = int(rid_s)
        got = method_by_rxn.get(rid)
        if got != exp_method:
            print(f"FAIL: rxn_{rid:04d} — got '{got}', expected '{exp_method}'")
            sys.exit(1)

    non_primary = {int(k) for k in expected["non_primary_rxns"]}
    mismatched = [(r, m) for r, m in method_by_rxn.items()
                  if r not in non_primary and m != "primary"]
    if mismatched:
        print(f"FAIL: {len(mismatched)} rxns expected 'primary' got other:")
        for r, m in mismatched[:10]:
            print(f"  rxn_{r:04d}: {m}")
        sys.exit(1)

    print(f"PASS: 5269/5269 built, recovery-method assignment matches reference.")
    for k in ("primary", "self_cyclo", "factor_1_20", "smiles_map", "heavy_prefix"):
        print(f"  {k}: {got_totals.get(k, 0)}")


if __name__ == "__main__":
    main()
