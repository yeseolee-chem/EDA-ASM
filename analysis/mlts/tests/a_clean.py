#!/usr/bin/env python3
"""tests/a_clean.py — remove A's own test outputs (sandbox only; the login node stays read-only):
  python tests/a_clean.py <sub> [<sub> ...]   -> rm -r $MLT/tmp/atest/<sub> (refuses anything outside $MLT/tmp/atest)
  python tests/a_clean.py --local             -> rm -r the node-local A test work folders of this node
                                                 (${TMPDIR:-/tmp}/{mlts,mlts_app}_atest*, mlts_unit_atest, mlts_bin_*)"""
import glob
import os
import shutil
import sys
from pathlib import Path

ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/atest").resolve()
for a in sys.argv[1:]:
    if a == "--local":
        base = {os.environ.get("TMPDIR") or "/tmp", "/tmp"}
        for b in base:
            for pat in ("mlts_atest*", "mlts_app_atest*", "mlts_unit_atest"):
                for p in glob.glob(os.path.join(b, pat)):
                    shutil.rmtree(p, ignore_errors=True)
                    print("removed", p)
        continue
    p = (ROOT / a).resolve()
    if ROOT not in p.parents:
        raise SystemExit(f"refusing {p}: not below {ROOT}")
    if p.exists():
        shutil.rmtree(p)
        print("removed", p)
