#!/usr/bin/env python3
"""launch_abs.py — ABS3 job chains, run on a compute node (CLAUDE.md: automated submission from a job, never from the
login node). Each stage writes $ABS3/launch_<stage>.json (job ids, one record per submission). Max parallel within
MaxSubmit 20 (array elements counted; the queue is checked before every submission).

  python launch_abs.py c    (after LS) C phase: Cg = tests_abs gpu (gpu1 RTX 3090), Cx = tests_abs xgb (CPU 32),
                            Cm = tests_abs merge ::: commit C (afterany Cg Cx), R = eval_abs selftest ::: prereg_abs write
                            ::: commit PREREG ::: launch_abs k (afterok Cm)
  python launch_abs.py k    (PREREG gate) K: GNN and TF arrays (gpu1), XGB array (CPU 32), KCn = train_abs collect
                            (afterany GNN TF), KCx = xgb_abs collect ::: commit K (afterok KCn, afterany XGB),
                            E = eval_abs run ::: commit E (afterok KCx)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402

LOGS = str(C.LOGS)
GPU = ["-p", "gpu1", "--gres=gpu:rtx3090:1", "--time=48:00:00"]
CPU = ["-p", "cpu1,cpu2", "--time=48:00:00"]
XCPU = CPU + ["--cpus-per-task=32", "--mem=64G", "--export=ALL,ABS3ENV=espley"]
D1 = CPU + ["--export=ALL,ABS3ENV=d1fu2"]
ES = CPU + ["--export=ALL,ABS3ENV=espley"]
MAX_SUBMIT = 20
_STAGE: dict = {}
_HIST: dict = {}


def sbatch(args):
    r = None
    for _ in range(3):                                   # transient controller errors: retry before STOP
        r = subprocess.run(["sbatch", "--parsable", *args], cwd=str(HERE), capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip().split(";")[0]
        time.sleep(30)
    C.stop(f"sbatch {' '.join(args)}: {r.stderr.strip()}")


def queued():
    """tasks of this user in the queue (pending + running, array tasks expanded) = what MaxSubmit counts; a failing
    squeue is retried and then a STOP (never read as an empty queue)."""
    for _ in range(5):
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", "yeseo1ee"), "-h", "-r"], capture_output=True, text=True)
        if r.returncode == 0:
            return len([ln for ln in r.stdout.splitlines() if ln.strip()])
        time.sleep(30)
    C.stop(f"squeue failed: {r.stderr.strip()}")


def wait_room(n, max_wait_s=6 * 3600):
    """block (on the compute node) until n more tasks fit under MaxSubmit 20."""
    t0 = time.time()
    while queued() + n > MAX_SUBMIT:
        if time.time() - t0 > max_wait_s:
            C.stop(f"no room for {n} tasks under MaxSubmit {MAX_SUBMIT} after {max_wait_s} s")
        time.sleep(60)


def alive(ids):
    """the job ids still known to the controller (pending / running); 'Invalid job id' = gone; other squeue errors are
    retried, then a STOP (a transient error must not look like a finished job)."""
    out = []
    for j in ids:
        for attempt in range(5):
            r = subprocess.run(["squeue", "-h", "-j", str(j), "-o", "%i"], capture_output=True, text=True)
            if r.returncode == 0:
                if r.stdout.strip():
                    out.append(str(j))
                break
            if "invalid job id" in (r.stderr or "").lower():
                break
            if attempt == 4:
                C.stop(f"squeue -j {j} failed: {r.stderr.strip()}")
            time.sleep(30)
    return out


def dep(kind, ids):
    ids = alive(ids)
    return [f"--dependency={kind}:{':'.join(ids)}"] if ids else []


def arr(name):
    return ["-J", name, "--array=0-4%5", f"--output={LOGS}/%x_%A_%a.out"]


def load_stage(stage):
    """the stage record of an earlier run ($ABS3/launch_<stage>.json) -> resubmitting the launcher is safe (review 11-11)."""
    if stage not in _STAGE:
        p = C.ABS3 / f"launch_{stage}.json"
        old = C.read_json(p) if p.is_file() else {}
        _STAGE[stage] = dict(old.get("jobs") or {})
        _HIST[stage] = dict(old.get("history") or {})
    return _STAGE[stage]


def save(stage, ids, **extra):
    rec = dict(written=C.now(), stage=stage, jobs=ids, history=_HIST.get(stage, {}), **extra)
    C.write_json(C.ABS3 / f"launch_{stage}.json", rec)
    print(json.dumps(rec, indent=1), flush=True)


def submit(stage, key, n_tasks, args):
    """one submission: a job of an earlier launcher run that is still queued / running is kept (never a second copy of a
    unit while the first trains); a finished one is resubmitted (idempotent steps: cached units return at once)."""
    rec = load_stage(stage)
    if key in rec:
        if alive([rec[key]]):
            print(f"[launch] {stage}/{key}: job {rec[key]} still in the queue, kept", flush=True)
            return rec[key]
        _HIST[stage].setdefault(key, []).append(rec[key])
    wait_room(n_tasks)
    rec[key] = sbatch(args)
    save(stage, rec)
    return rec[key]


def stage_c():
    cg = submit("c", "Cg", 1, GPU + ["-J", "abs3_Cg", "jobs/gpu_abs.slurm", "tests_abs.py", "gpu"])
    cx = submit("c", "Cx", 1, XCPU + ["-J", "abs3_Cx", "jobs/cpu_abs.slurm", "tests_abs.py", "xgb"])
    cm = submit("c", "Cm", 1, D1 + ["-J", "abs3_Cm"] + dep("afterany", [cg, cx]) + ["jobs/cpu_abs.slurm", "tests_abs.py",
                                                                                    "merge", ":::", "commit_abs.py", "C"])
    submit("c", "R", 1, ES + ["-J", "abs3_R", f"--dependency=afterok:{cm}", "--kill-on-invalid-dep=yes",
                              "jobs/cpu_abs.slurm", "eval_abs.py", "selftest", ":::", "prereg_abs.py", "write", ":::",
                              "commit_abs.py", "PREREG", ":::", "launch_abs.py", "k"])
    save("c", load_stage("c"))


def stage_k():
    import prereg_abs as PR
    pre = PR.check()
    g = submit("k", "G", 5, GPU + arr("abs3_G") + ["jobs/gpu_abs.slurm", "train_abs.py", "run", "GNN", "@I", "5"])
    t = submit("k", "T", 5, GPU + arr("abs3_T") + ["jobs/gpu_abs.slurm", "train_abs.py", "run", "TF", "@I", "5"])
    x = submit("k", "X", 5, XCPU + arr("abs3_X") + ["jobs/cpu_abs.slurm", "xgb_abs.py", "run", "@I", "5"])
    kcn = submit("k", "KCn", 1, D1 + ["-J", "abs3_KCn"] + dep("afterany", [g, t]) + ["jobs/cpu_abs.slurm", "train_abs.py",
                                                                                      "collect"])
    xa = alive([x])
    d = f"--dependency=afterok:{kcn}" + (f",afterany:{':'.join(xa)}" if xa else "")
    kcx = submit("k", "KCx", 1, ES + ["-J", "abs3_KCx", d, "--kill-on-invalid-dep=yes", "jobs/cpu_abs.slurm", "xgb_abs.py",
                                      "collect", ":::", "commit_abs.py", "K"])
    submit("k", "E", 1, ES + ["-J", "abs3_E", f"--dependency=afterok:{kcx}", "--kill-on-invalid-dep=yes",
                              "jobs/cpu_abs.slurm", "eval_abs.py", "run", ":::", "commit_abs.py", "E"])
    save("k", load_stage("k"), prereg_git_head=pre.get("git_head"), prereg_md_sha256=pre.get("md_sha256"))


if __name__ == "__main__":
    a = sys.argv[1:]
    fn = dict(c=stage_c, k=stage_k).get(a[0] if a else "")
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn() or 0)
