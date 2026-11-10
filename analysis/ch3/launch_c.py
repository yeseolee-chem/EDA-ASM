#!/usr/bin/env python3
"""launch_c.py — CH3-MODEL 4-7 submission, run as its own small CPU job after the T stage of the regeneration chain
(CLAUDE.md: automated submission from a job, never from the login node; one queue slot while L / F / T hold the rest).

  python launch_c.py      -> $CH3/launch_c.json (job ids)

Chain (submitted from analysis/ch3):
  VC   cpu, CH3ENV=d1fu2           verify_inputs.py ::: commit_phase.py V ::: tests_ch3.py cpu      (V, C-Y, C-D, C-5)
  C8   gpu1 RTX 3090, afterok:VC   train_ch3.py smoke                                              (C-8, C-T, C-4)
  C3r  gpu1 RTX 3090, afterok:VC   train_ch3.py c3r                                                (C-3′, C-4)
  Cm   cpu, afterany:C8:C3r        tests_ch3.py merge ::: commit_phase.py C                         (C_tests.json)
  R    cpu, afterok:Cm, espley     eval_ch3.py selftest ::: prereg_ch3.py write ::: commit_phase.py PREREG
                                   ::: launch_k.py                                               (PREREG, then K chain)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402
from launch_k import CPU, GPU, sbatch  # noqa: E402


def main():
    ids = {}
    ids["VC"] = sbatch(CPU + ["-J", "ch3_VC", "--export=ALL,CH3ENV=d1fu2", "jobs/cpu_ch3.slurm", "verify_inputs.py",
                              ":::", "commit_phase.py", "V", ":::", "tests_ch3.py", "cpu"])
    dep = ["--dependency=afterok:" + ids["VC"], "--kill-on-invalid-dep=yes"]
    ids["C8"] = sbatch(GPU + ["-J", "ch3_C8", *dep, "jobs/gpu_ch3.slurm", "train_ch3.py", "smoke"])
    ids["C3r"] = sbatch(GPU + ["-J", "ch3_C3r", *dep, "jobs/gpu_ch3.slurm", "train_ch3.py", "c3r"])
    ids["Cm"] = sbatch(CPU + ["-J", "ch3_Cm", "--export=ALL,CH3ENV=d1fu2",
                              f"--dependency=afterany:{ids['C8']}:{ids['C3r']}", "jobs/cpu_ch3.slurm", "tests_ch3.py",
                              "merge", ":::", "commit_phase.py", "C"])
    ids["R"] = sbatch(CPU + ["-J", "ch3_R", "--export=ALL,CH3ENV=espley", f"--dependency=afterok:{ids['Cm']}",
                             "--kill-on-invalid-dep=yes", "jobs/cpu_ch3.slurm", "eval_ch3.py", "selftest", ":::",
                             "prereg_ch3.py", "write", ":::", "commit_phase.py", "PREREG", ":::", "launch_k.py"])
    rec = dict(written=C3.now(), jobs=ids)
    C3.write_json(C3.CH3 / "launch_c.json", rec)
    print(json.dumps(rec, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
