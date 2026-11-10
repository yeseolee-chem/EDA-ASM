#!/usr/bin/env python3
"""run_dev.py — train_ch3.py run / final under the recorded C-3′ deviation (results/C_deviation.json; user 11-10
"추천 방식대로 진행"). Same precedent as analysis/deploy/final_dev.py: train_ch3.py is not edited (its sha256 stays the one
C_tests.json recorded); only train_ch3.gate_c_tests is replaced in this process:
  C_tests.json all_pass  -> the original gate;
  otherwise              -> c_deviation.check() (proceed true, fixed files unchanged) AND the C_tests.json
                            trainer_code_sha256 equal to the trainer files now (the C tests ran on this code).

  python run_dev.py run <i|@I> <n>      = train_ch3.py run under the deviation
  python run_dev.py final               = train_ch3.py final under the deviation
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402
import train_ch3 as TC  # noqa: E402
import c_deviation as CD  # noqa: E402

_orig_gate = TC.gate_c_tests


def gate_c_tests_dev():
    p = C3.RES / "C_tests.json"
    c = C3.read_json(p) if p.is_file() else {}
    if c.get("all_pass"):
        return _orig_gate()
    dev = CD.check()
    rec = c.get("trainer_code_sha256") or {}
    diff = [f for f in TC.TRAINER_FILES if rec.get(f) != TC.CODE_SHA[f]]
    if diff:
        C3.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}: rerun the C tests")
    print(f"[run_dev] C_tests.json all_pass false; recorded C-3′ deviation accepted "
          f"(C_deviation.json {dev['c_deviation_sha256'][:12]})", flush=True)
    return dict(c_tests_sha256=C3.sha256(p), c_deviation_sha256=dev["c_deviation_sha256"])


TC.gate_c_tests = gate_c_tests_dev

if __name__ == "__main__":
    a = sys.argv[1:]
    if not (a[:1] == ["run"] and len(a) == 3) and a != ["final"]:
        print(__doc__)
        sys.exit(2)
    sys.exit(TC.main(["train_ch3.py"] + a))
