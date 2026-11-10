#!/usr/bin/env python3
"""launch_m.py — CH3M job chains, run on a compute node (CLAUDE.md: automated submission from a job, never from the
login node). Each stage writes $CH3M/launch_<stage>.json (job ids). Max parallel, MaxSubmit 20 respected by staging.

  python launch_m.py c    (after XP) C phase: Ccpu (commit XP ::: tests_m cpu), C-RT, C-RG(+RG2), C-8 GNN (gpu1 RTX 3090),
                          C-RX, C-8 XGB (CPU 32), merge (::: commit C), R (eval selftest ::: prereg write ::: commit PREREG
                          ::: launch_m.py k)
  python launch_m.py k    (PREREG gate) K wave 1: GNN10, TF10 arrays (gpu1), XGB array (CPU 32: run 10 ::: run 3), and
                          launch_m.py k2 (afterany TF10)
  python launch_m.py k2   K wave 2: GNN3 array (gpu1), collect nn (afterany GNN10 GNN3 TF10), collect xgb + commit K
                          (afterok nn, afterany XGB), E (eval run ::: commit E ::: launch_m.py w)
  python launch_m.py w    spec 9 from results/EM_report.json choice.final_model: TF10 / GNN10 / GNN3 -> WI job
                          (final ::: commit W ::: I-1 ;;; I-2 ;;; commit I); none -> record only
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402

LOGS = str(C.LOGS)
GPU = ["-p", "gpu1", "--gres=gpu:rtx3090:1", "--time=48:00:00"]
CPU = ["-p", "cpu1,cpu2", "--time=48:00:00"]
XCPU = CPU + ["--cpus-per-task=32", "--mem=64G", "--export=ALL,CH3MENV=espley"]
D1 = CPU + ["--export=ALL,CH3MENV=d1fu2"]
ES = CPU + ["--export=ALL,CH3MENV=espley"]
RE = CPU + ["--cpus-per-task=1", "--mem=2G", "--export=ALL,CH3MENV=reactot"]


MAX_SUBMIT = 20
_STAGE: dict = {}


def sbatch(args):
    r = subprocess.run(["sbatch", "--parsable", *args], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0:
        C.stop(f"sbatch {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip().split(";")[0]


def queued():
    """tasks of this user in the queue (pending + running, array tasks expanded) = what MaxSubmit counts (spec 14)."""
    import os
    r = subprocess.run(["squeue", "-u", os.environ.get("USER", "yeseo1ee"), "-h", "-r"], capture_output=True, text=True)
    return len([ln for ln in r.stdout.splitlines() if ln.strip()])


def wait_room(n, max_wait_s=6 * 3600):
    """block (on the compute node) until n more tasks fit under MaxSubmit 20 (review 11-10)."""
    import time
    t0 = time.time()
    while queued() + n > MAX_SUBMIT:
        if time.time() - t0 > max_wait_s:
            C.stop(f"no room for {n} tasks under MaxSubmit {MAX_SUBMIT} after {max_wait_s} s")
        time.sleep(60)


def alive(ids):
    """the job ids still known to the controller (finished / purged ones are dropped from dependency lists: the next
    step checks completeness itself; review 11-10)."""
    out = []
    for j in ids:
        r = subprocess.run(["squeue", "-h", "-j", str(j), "-o", "%i"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            out.append(str(j))
    return out


def dep(kind, ids):
    ids = alive(ids)
    return [f"--dependency={kind}:{':'.join(ids)}"] if ids else []


def arr(name):
    return ["-J", name, "--array=0-4%5", f"--output={LOGS}/%x_%A_%a.out"]


def submit(stage, key, n_tasks, args):
    """one submission with the queue-room check; the stage record is written after every submission."""
    rec = _STAGE.setdefault(stage, {})
    if key in rec:
        return rec[key]
    wait_room(n_tasks)
    rec[key] = sbatch(args)
    save(stage, rec)
    return rec[key]


def save(stage, ids, **extra):
    rec = dict(written=C.now(), stage=stage, jobs=ids, **extra)
    C.write_json(C.CH3M / f"launch_{stage}.json", rec)
    print(json.dumps(rec, indent=1), flush=True)


def stage_c():
    ids = {}
    ids["Ccpu"] = sbatch(D1 + ["-J", "ch3m_Ccpu", "jobs/cpu_m.slurm", "commit_m.py", "XP", ":::", "tests_m.py", "cpu"])
    ids["CRT"] = sbatch(GPU + ["-J", "ch3m_CRT", "jobs/gpu_m.slurm", "train_m.py", "cr", "TF"])
    ids["CRG"] = sbatch(GPU + ["-J", "ch3m_CRG", "jobs/gpu_m.slurm", "train_m.py", "cr", "GNN"])
    ids["C8nn"] = sbatch(GPU + ["-J", "ch3m_C8nn", "jobs/gpu_m.slurm", "train_m.py", "smoke"])
    ids["CRX"] = sbatch(XCPU + ["-J", "ch3m_CRX", "jobs/cpu_m.slurm", "xgb_m.py", "cr"])
    ids["C8x"] = sbatch(XCPU + ["-J", "ch3m_C8x", "jobs/cpu_m.slurm", "xgb_m.py", "smoke"])
    cdep = ":".join(ids[k] for k in ("Ccpu", "CRT", "CRG", "C8nn", "CRX", "C8x"))
    ids["Cm"] = sbatch(D1 + ["-J", "ch3m_Cm", f"--dependency=afterany:{cdep}", "jobs/cpu_m.slurm", "tests_m.py", "merge",
                             ":::", "commit_m.py", "C"])
    ids["R"] = sbatch(ES + ["-J", "ch3m_R", f"--dependency=afterok:{ids['Cm']}", "--kill-on-invalid-dep=yes",
                            "jobs/cpu_m.slurm", "eval_m.py", "selftest", ":::", "prereg_m.py", "write", ":::",
                            "commit_m.py", "PREREG", ":::", "launch_m.py", "k"])
    save("c", ids)


def stage_k():
    """K wave 1 (max parallel, user 11-10: GNN10 and TF10 together instead of the spec 7 afterany chain; XGB 10 and 3 in
    one array, the 3 mode run regardless of the 10 mode exit)."""
    import prereg_m as PR
    pre = PR.check()
    submit("k", "G10", 5, GPU + arr("ch3m_G10") + ["jobs/gpu_m.slurm", "train_m.py", "run", "GNN", "10", "@I", "5"])
    t10 = submit("k", "T10", 5, GPU + arr("ch3m_T10") + ["jobs/gpu_m.slurm", "train_m.py", "run", "TF", "10", "@I", "5"])
    submit("k", "X", 5, XCPU + arr("ch3m_X") + ["jobs/cpu_m.slurm", "xgb_m.py", "run", "10", "@I", "5", ";;;", "xgb_m.py",
                                                "run", "3", "@I", "5"])
    submit("k", "L2", 1, RE + ["-J", "ch3m_L2"] + dep("afterany", [t10]) + ["jobs/cpu_m.slurm", "launch_m.py", "k2"])
    save("k", _STAGE["k"], prereg_git_head=pre.get("git_head"), prereg_md_sha256=pre.get("md_sha256"))


def stage_k2():
    k = C.read_json(C.CH3M / "launch_k.json")["jobs"]
    g3 = submit("k2", "G3", 5, GPU + arr("ch3m_G3") + ["jobs/gpu_m.slurm", "train_m.py", "run", "GNN", "3", "@I", "5"])
    kcn = submit("k2", "KCn", 1, D1 + ["-J", "ch3m_KCn"] + dep("afterany", [k["G10"], g3, k["T10"]])
                 + ["jobs/cpu_m.slurm", "train_m.py", "collect"])
    xa = alive([k["X"]])
    d = f"--dependency=afterok:{kcn}" + (f",afterany:{':'.join(xa)}" if xa else "")
    kcx = submit("k2", "KCx", 1, ES + ["-J", "ch3m_KCx", d, "--kill-on-invalid-dep=yes", "jobs/cpu_m.slurm", "xgb_m.py",
                                       "collect", ":::", "commit_m.py", "K"])
    submit("k2", "E", 1, ES + ["-J", "ch3m_E", f"--dependency=afterok:{kcx}", "--kill-on-invalid-dep=yes",
                               "jobs/cpu_m.slurm", "eval_m.py", "run", ":::", "commit_m.py", "E", ":::", "launch_m.py", "w"])
    save("k2", _STAGE["k2"])


def stage_w():
    rep = C.read_json(C.RES / "EM_report.json")
    ch = rep.get("choice") or {}
    fm = ch.get("final_model")
    if fm in ("TF10", "GNN10", "GNN3"):
        submit("w", "WI", 1, GPU + ["-J", "ch3m_WI", "jobs/gpu_m.slurm", "final_m.py", fm, ":::", "commit_m.py", "W",
                                    ":::", "predict_m.py", "check-i1", fm, ";;;", "predict_m.py", "check-i2", fm, ";;;",
                                    "commit_m.py", "I"])
    save("w", _STAGE.get("w", {}), chosen=ch.get("chosen"), final_model=fm, action=ch.get("action"))


if __name__ == "__main__":
    a = sys.argv[1:]
    fn = dict(c=stage_c, k=stage_k, k2=stage_k2, w=stage_w).get(a[0] if a else "")
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn() or 0)
