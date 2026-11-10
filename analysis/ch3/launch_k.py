#!/usr/bin/env python3
"""launch_k.py — CH3-MODEL 8-11 submission, run on a compute node at the end of the PREREG job (CLAUDE.md: automated
submission from a job, never from the login node). Exit 3 unless the PREREG gate passes (prereg_ch3.check: both files
committed and unchanged), so K is submitted only after the PREREG commit (spec 13).

  python launch_k.py      -> $CH3/launch_k.json (job ids)

Chain (submitted from analysis/ch3, SLURM_SUBMIT_DIR of every child = analysis/ch3):
  K    gpu1 RTX 3090, --array=0-4%5          jobs/gpu_ch3.slurm train_ch3.py run @I 5
  Kc   cpu, afterany:K, CH3ENV=d1fu2          train_ch3.py collect ::: commit_phase.py K
  E    cpu, afterok:Kc, CH3ENV=espley         eval_ch3.py run ::: commit_phase.py E
  WI   gpu1 RTX 3090, afterok:E                train_ch3.py final ::: commit_phase.py W ::: predict_ch3.py check-i1
                                               ;;; predict_ch3.py check-i2 ;;; commit_phase.py I
       (W and I-2 in one job = the same GPU, spec 11 I-2)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402
import prereg_ch3 as PR  # noqa: E402

LOGS = str(C3.LOGS)
GPU = ["-p", "gpu1", "--gres=gpu:rtx3090:1", "--time=48:00:00"]
CPU = ["-p", "cpu1,cpu2", "--time=48:00:00"]


def sbatch(args):
    r = subprocess.run(["sbatch", "--parsable", *args], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0:
        C3.stop(f"sbatch {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip().split(";")[0]


def main():
    pre = PR.check()
    ids = {}
    ids["K"] = sbatch(GPU + ["-J", "ch3_K", "--array=0-4%5", f"--output={LOGS}/%x_%A_%a.out",
                             "jobs/gpu_ch3.slurm", "train_ch3.py", "run", "@I", "5"])
    ids["Kc"] = sbatch(CPU + ["-J", "ch3_Kc", "--export=ALL,CH3ENV=d1fu2", f"--dependency=afterany:{ids['K']}",
                              "jobs/cpu_ch3.slurm", "train_ch3.py", "collect", ":::", "commit_phase.py", "K"])
    ids["E"] = sbatch(CPU + ["-J", "ch3_E", "--export=ALL,CH3ENV=espley", f"--dependency=afterok:{ids['Kc']}",
                             "--kill-on-invalid-dep=yes", "jobs/cpu_ch3.slurm", "eval_ch3.py", "run", ":::",
                             "commit_phase.py", "E"])
    ids["WI"] = sbatch(GPU + ["-J", "ch3_WI", f"--dependency=afterok:{ids['E']}", "--kill-on-invalid-dep=yes",
                              "jobs/gpu_ch3.slurm", "train_ch3.py", "final", ":::", "commit_phase.py", "W", ":::",
                              "predict_ch3.py", "check-i1", ";;;", "predict_ch3.py", "check-i2", ";;;",
                              "commit_phase.py", "I"])
    rec = dict(written=C3.now(), prereg_git_head=pre.get("git_head"), prereg_md_sha256=pre.get("md_sha256"), jobs=ids)
    C3.write_json(C3.CH3 / "launch_k.json", rec)
    print(json.dumps(rec, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
