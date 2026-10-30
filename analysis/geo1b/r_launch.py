#!/usr/bin/env python3
"""Phase R launcher (spec 9-4; CLAUDE.md option 2: it runs as a 1-CPU sbatch job, never on the login node).
After the smoke job it computes the GPU budget (decision R-smoke timing rule), applies the pre-approved 9-4 reductions
in order if needed, then submits tune 20 -> select 5 -> final 15 + ablation (15 or 9) -> evaluate 1 in waves that keep
`squeue -u $USER -h -r | wc -l` <= 20 (MaxSubmit; MaxJobs 10 is enforced by SLURM). Idempotent: a task whose output
file exists or whose job name is queued is never submitted again; a job that ended without its output is resubmitted
once (infra), a second loss blocks its dependants and writes results/STOP_R.md.
  sbatch -p cpu1,cpu2 --time=48:00:00 --cpus-per-task=1 --mem=2G --dependency=afterok:<smoke> \
         --wrap "source .../conda.sh; conda activate reactot; cd analysis/geo1b; python r_launch.py"
Writes results/R_budget.json, results/HPC_efficiency.csv, $GEO/run_r/state.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))
RUNS, PREDS, LOGS, STATE = GEO / "runs", GEO / "preds", GEO / "logs", GEO / "run_r" / "state.json"
GPU_P, CPU_P = "gpu1,gpu2,gpu3,gpu5,gpu6", "cpu1,cpu2"
MAX_SUBMIT, GPU_H_MAX, POLL_S, STOP_AFTER_H = 20, 200.0, 120, 46.0
FOLDS, CFGS, SEEDS, ABL = (0, 1, 2, 3, 4), ("c0", "c1", "c2", "c3"), (0, 1, 2), ()   # GEO1b: no ablation
USER = os.environ.get("USER", "yeseo1ee")


def wjson(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def decision(key, text):
    p = RES / "DECISIONS.md"
    if f"- **{key}**:" not in p.read_text():
        with open(p, "a") as f:
            f.write(f"- **{key}**: {text}\n")


def sh(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def epoch_s(tag):
    h = json.loads((RUNS / f"{tag}.history.json").read_text())["hist"]
    return float(h[1]["wall_s"] - h[0]["wall_s"])        # decision R-smoke: one train epoch + one validation


def budget():
    p = RES / "R_budget.json"
    if p.is_file():
        return json.loads(p.read_text())
    if os.environ.get("GEO_R_MAX_EP"):                    # decision R-user (2026-10-28): the user's epochs / no patience
        me, pa = int(os.environ["GEO_R_MAX_EP"]), int(os.environ.get("GEO_R_PATIENCE", "100000"))
        B = dict(user_override=True, limit_gpu_h=None, stop=False,
                 chosen=dict(step="user 2026-10-28", max_ep=me, patience=pa, abl_folds=list(FOLDS), gpu_h=None))
        wjson(p, B)
        return B
    t1, t3 = epoch_s("smoke_f0_c0"), epoch_s("smoke_f0_c1")
    est = lambda ep, n_abl: ep * (10 * t1 + 10 * t3 + 15 * t3 + n_abl * t3) / 3600.0   # noqa: E731  spec 9-4
    steps = [dict(step="none", max_ep=100, patience=15, abl_folds=list(FOLDS), gpu_h=est(100, 15)),
             dict(step="1", max_ep=60, patience=10, abl_folds=list(FOLDS), gpu_h=est(60, 15)),
             dict(step="1+2", max_ep=60, patience=10, abl_folds=[0, 1, 2], gpu_h=est(60, 9))]
    pick = next((s for s in steps if s["gpu_h"] <= GPU_H_MAX), None)
    B = dict(t1_s=t1, t3_s=t3, limit_gpu_h=GPU_H_MAX, steps=steps, chosen=pick, stop=pick is None)
    wjson(p, B)
    if pick is None:
        (RES / "STOP_R.md").write_text(
            f"# STOP R (spec 9-4): GPU budget > {GPU_H_MAX} h even after both pre-approved reductions\n\n"
            f"t1 = {t1:.1f} s, t3 = {t3:.1f} s per epoch (smoke, fold 0); estimates: "
            + ", ".join(f"{s['step']}: {s['gpu_h']:.0f} h" for s in steps) + "\n")
    else:
        decision("R-budget", f"smoke(fold 0) epoch 시간 t1 {t1:.1f} s(c0, R=1), t3 {t3:.1f} s(c1, R=3). 명령서 9-4 상한 "
                 f"예상 = 축소 없음 {steps[0]['gpu_h']:.0f}, ① {steps[1]['gpu_h']:.0f}, ①+② {steps[2]['gpu_h']:.0f} "
                 f"GPU시간 → 적용: '{pick['step']}' (MAX_EP {pick['max_ep']}, PATIENCE {pick['patience']}, ablation "
                 f"fold {pick['abl_folds']}); GEO_MAX_EP / GEO_PATIENCE 환경 변수로 train.py에 넘긴다.")
    return B


def tasks(B):
    c = B["chosen"]
    env = f"ALL,GEO_MAX_EP={c['max_ep']},GEO_PATIENCE={c['patience']}"
    T = []
    for k in FOLDS:
        for cfg in CFGS:
            T.append(dict(name=f"gb_tune_f{k}_{cfg}", gpu=True, args=["train.py", "tune", str(k), cfg], env=env,
                          out=RUNS / f"tune_f{k}_{cfg}.history.json", deps=[]))
    for k in FOLDS:
        T.append(dict(name=f"gb_sel_f{k}", gpu=False, args=["train.py", "select", str(k)], env=env,
                      out=RES / f"R_select_f{k}.json", deps=[f"gb_tune_f{k}_{cfg}" for cfg in CFGS]))
    for k in FOLDS:
        for s in SEEDS:
            T.append(dict(name=f"gb_final_f{k}_s{s}", gpu=True, args=["train.py", "final", str(k), str(s)], env=env,
                          out=PREDS / f"final_f{k}_s{s}.pkl", deps=[f"gb_sel_f{k}"]))
        if k in c["abl_folds"]:
            for a in ABL:
                T.append(dict(name=f"gb_abl_{a}_f{k}", gpu=True, args=["train.py", "ablation", str(k), a], env=env,
                              out=PREDS / f"abl_{a}_f{k}.pkl", deps=[f"gb_sel_f{k}"]))
    T.append(dict(name="gb_eval", gpu=False, args=["evaluate.py"], env=env, out=RES / "E_report.json",
                  deps=[t["name"] for t in T if t["name"].startswith(("gb_final", "gb_abl"))]))
    return T


def submit(t):
    res = ["-p", GPU_P, "--gres=gpu:1"] if t["gpu"] else ["-p", CPU_P]
    out = sh(["sbatch", "--parsable", *res, "-J", t["name"], "--time=48:00:00", "--cpus-per-task=4", "--mem=32G",
              f"--output={LOGS}/%x_%j.out", f"--export={t['env']}", "jobs/gpu.slurm", *t["args"]])
    return out.strip().split(";")[0]


def hpc_csv(st):
    ids = sorted({j for v in st.values() for j in v.get("jobids", [])})        # every attempt (spec 14)
    if not ids:
        return
    try:
        out = sh(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o",
                  "JobID,JobName,State,Elapsed,ElapsedRaw,CPUTimeRAW,AllocTRES"])
    except subprocess.CalledProcessError as e:
        print(f"[r_launch] sacct failed, HPC_efficiency.csv not written: {e.stderr}", flush=True)
        return
    tmp = RES / f"HPC_efficiency.csv.{os.getpid()}.tmp"
    tmp.write_text("job_id|job_name|state|elapsed|elapsed_s|cputime_raw_s|alloc_tres\n" + out)
    os.replace(tmp, RES / "HPC_efficiency.csv")


RELAUNCH = ("source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh && conda activate reactot && "
            f"cd {HERE} && PYTHONDONTWRITEBYTECODE=1 python r_launch.py")
ONCE = os.environ.get("GEO_R_ONCE") == "1"      # decision R-once: one pass, then again in ONCE_MIN min (frees the slot)
ONCE_MIN = 10
PEND_MAX = 2                                    # decision R-lpt: ONCE mode keeps <= 2 R tasks pending (see cost)
NPASS = dict(c0=2, c1=4, c2=2, c3=4)            # train.py CFGS: n_pass = R + 1


def cost(t):
    """decision R-lpt: relative GPU cost of a ready final / ablation task, (best_ep + 1) x (passes + 2), x 0.5 without
    D0, so the longest ready jobs are submitted first; select and evaluate go before everything."""
    a = t["args"]
    if len(a) < 3 or a[1] not in ("final", "ablation"):
        return float("inf")
    sel = json.loads((RES / f"R_select_f{a[2]}.json").read_text())
    p = 1 if a[1] == "ablation" and a[3] == "r0" else NPASS[sel["chosen"]]
    return (sel["best_ep"] + 1) * (p + 2) * (0.5 if a[1] == "ablation" and a[3] == "no_d0" else 1.0)


def relaunch(begin=False):
    """46 h (or ONCE mode after every pass): the same launcher again (idempotent); never leaves the next waves
    unsubmitted. ONCE mode waits with --begin, so between passes it holds a submit slot but no running slot."""
    me = os.environ.get("SLURM_JOB_ID")
    dep = [f"--begin=now+{ONCE_MIN}minutes"] if begin else ([f"--dependency=afterany:{me}"] if me else [])
    env = "--export=ALL,GEO_R_ONCE=1" if ONCE else "--export=ALL"
    try:
        out = sh(["sbatch", "--parsable", "-p", CPU_P, "-J", "gb_rlaunch", "--time=48:00:00", "--cpus-per-task=1",
                  "--mem=2G", f"--output={LOGS}/%x_%j.out", env, *dep, "--wrap", RELAUNCH])
        print(f"[r_launch] relaunched as {out.strip()}", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[r_launch] relaunch failed (resubmit by hand): {e.stderr}", flush=True)
        return False


def main():
    os.chdir(HERE)
    t0 = time.time()
    B = budget()
    if B["stop"]:
        print("STOP R: budget", flush=True)
        return 3
    T = tasks(B)
    byname = {t["name"]: t for t in T}
    st = json.loads(STATE.read_text()) if STATE.is_file() else {}
    while True:
        try:
            rows = [ln.split("|") for ln in sh(["squeue", "-u", USER, "-h", "-r", "-o", "%j|%T"]).splitlines()
                    if ln.strip()]
        except subprocess.CalledProcessError as e:                 # transient slurmctld error: try again later
            print(f"[r_launch] squeue failed: {e.stderr}", flush=True)
            time.sleep(POLL_S)
            continue
        queued = [r[0] for r in rows]
        npend = sum(1 for r in rows if r[0] in byname and r[-1] == "PENDING")      # R tasks only (not the launcher)
        active, nq, todo, dead = set(queued), len(queued), [], []
        for t in T:                                                # T is in dependency order
            n, rec = t["name"], st.setdefault(t["name"], dict(tries=0))
            if t["out"].exists() or n in active:
                continue
            if rec.get("tries", 0) >= 2 or any(d in dead for d in t["deps"]):
                dead.append(n)                                     # lost twice, or depends on such a task
                continue
            if any(not byname[d]["out"].exists() for d in t["deps"]):
                continue
            todo.append(t)
        if ONCE:                                                   # R-lpt: longest ready job first
            todo.sort(key=lambda t: -cost(t))
        for t in todo:
            if nq >= MAX_SUBMIT - ONCE:                            # R-once: nq counts this launcher; keep 1 for the next
                break
            if ONCE and npend >= PEND_MAX:                         # R-lpt: the launcher gets a slot every few job ends
                break
            try:
                jid = submit(t)
            except subprocess.CalledProcessError as e:             # e.g. MaxSubmit hit by another job: next round
                print(f"[r_launch] sbatch {t['name']} failed: {e.stderr}", flush=True)
                break
            rec = st[t["name"]]
            rec.update(jobid=jid, tries=rec.get("tries", 0) + 1, submitted=time.strftime("%F %T"))
            rec.setdefault("jobids", []).append(jid)
            nq += 1
            npend += 1
            print(f"[{time.strftime('%F %T')}] submitted {t['name']} -> {jid}", flush=True)
            wjson(STATE, st)
        wjson(STATE, st)
        left = [t["name"] for t in T if not t["out"].exists()]
        if not left:
            hpc_csv(st)
            print("[r_launch] all R tasks and the evaluation finished", flush=True)
            return 0
        if dead and not any(n in active for n in left) and not todo:
            (RES / "STOP_R.md").write_text("# STOP R: jobs lost twice without output (and their dependants)\n\n"
                                           + "\n".join(f"- {b}" for b in dead) + "\n")
            hpc_csv(st)
            print(f"STOP R: dead {dead}", flush=True)
            return 3
        if ONCE and relaunch(begin=True):                  # R-once: next pass in ONCE_MIN min, no running slot meanwhile
            return 0                                       # (if the resubmit failed, keep polling as before)
        if time.time() - t0 > STOP_AFTER_H * 3600:
            hpc_csv(st)
            relaunch()
            print("[r_launch] 46 h: relaunched (idempotent)", flush=True)
            return 2
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
