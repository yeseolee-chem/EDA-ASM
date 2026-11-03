#!/usr/bin/env python3
"""prereg_w_dev.py — prereg_w.py start / write under the recorded DEPLOY G-C deviation.

DEPLOY_MODEL_W start condition 0-1 asks for 'E_report.json G-C passed'. The lf G-C failed, and the pre-registered
null check (results/NULL_gc.json) allows proceeding with the failure recorded as a deviation (results/DECISIONS.md).
This wrapper replaces, in this process only, w_common.check_start by the original check with 0-1 read as
'C_tests all_pass and PREREG_DEPLOY committed and (G-C passed or gc_deviation.allowed())'; everything else unchanged.
  python prereg_w_dev.py start|write     (env as prereg_w.py; jobs/cpu_py.slurm DEPENV=d1fu2, DEPW_FREE_GB=<GB>)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import gc_deviation as GD  # noqa: E402
import w_common as W  # noqa: E402

_orig = W.check_start


def check_start_dev(free_gb=None):
    rec = _orig(free_gb)
    d = (rec.get("detail") or {}).get("0-1") or {}
    if not rec["checks"].get("0-1") and d.get("c_tests_all_pass") and d.get("prereg_committed") and d.get("G_C") is not True:
        ev = GD.allowed()
        if ev.get("deviation"):
            rec["checks"]["0-1"] = True
            rec["detail"]["0-1"] = dict(d, G_C_deviation=ev)
            rec["all_pass"] = all(rec["checks"].values())
    return rec


if __name__ == "__main__":
    W.check_start = check_start_dev
    import prereg_w as PW  # noqa: E402
    PW.W.check_start = check_start_dev
    a = sys.argv[1:]
    if a == ["start"]:
        sys.exit(PW.cmd_start())
    if a == ["write"]:
        sys.exit(PW.cmd_write())
    raise SystemExit(__doc__)
