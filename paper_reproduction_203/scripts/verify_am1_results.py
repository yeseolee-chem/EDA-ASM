#!/usr/bin/env python3
"""
After running Gaussian AM1 on the bundle, verify completion + basic sanity.

Usage:
    python scripts/verify_am1_results.py

Checks:
  - all 1015 .log files exist
  - all end with "Normal termination"
  - cclib can parse Mulliken and APT charges from each
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent.parent)
    args = ap.parse_args()

    n_ok = 0
    n_missing = []
    n_incomplete = []
    n_no_apt = []

    try:
        import cclib
    except ImportError:
        cclib = None
        print("WARN: cclib not installed — skipping charge check")

    for type_dir in ["ts", "gs", "dist_gs"]:
        inp_dir = args.root / "gaussian_inputs" / type_dir
        log_dir = args.root / "gaussian_inputs" / "am1_logs" / type_dir
        # fall back to same-dir logs if am1_logs/ layout not used
        if not log_dir.exists():
            log_dir = inp_dir
        for gjf in sorted(inp_dir.glob("*.gjf")):
            log = log_dir / (gjf.stem + ".log")
            if not log.exists():
                out = log_dir / (gjf.stem + ".out")
                if out.exists():
                    log = out
                else:
                    n_missing.append(str(log.relative_to(args.root)))
                    continue
            txt = log.read_text(errors="ignore")
            if "Normal termination" not in txt:
                n_incomplete.append(str(log.relative_to(args.root)))
                continue
            n_ok += 1
            if cclib is not None:
                try:
                    d_ = cclib.io.ccread(str(log))
                    if not (hasattr(d_, "atomcharges") and "mulliken" in d_.atomcharges and "apt" in d_.atomcharges):
                        n_no_apt.append(str(log.relative_to(args.root)))
                except Exception:
                    n_no_apt.append(str(log.relative_to(args.root)))

    total = 1015
    print(f"Complete (Normal termination): {n_ok}/{total}")
    if n_missing:
        print(f"MISSING: {len(n_missing)} — first 5:")
        for m in n_missing[:5]: print(f"  {m}")
    if n_incomplete:
        print(f"INCOMPLETE (no Normal termination): {len(n_incomplete)} — first 5:")
        for m in n_incomplete[:5]: print(f"  {m}")
    if n_no_apt:
        print(f"MISSING APT/mulliken from cclib: {len(n_no_apt)} — first 5:")
        for m in n_no_apt[:5]: print(f"  {m}")

    if n_ok == total and not n_no_apt:
        print("\nAll AM1 results verified. Ready for Grayson pipeline.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
