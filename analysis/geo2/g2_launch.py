#!/usr/bin/env python3
r"""g2_launch.py — GEO2 pipeline launcher, Y -> Z -> T -> G / F / H -> P -> C -> R -> M3 -> E (docs/specs/GEO2.md 8 order;
role ORCH). The ONCE-mode design of analysis/geo/r_launch.py (GEO1 decisions R-once, R-lpt): every pass is one short
1-CPU sbatch job on cpu1,cpu2 (CLAUDE.md option 2: never on the login node). A pass reads squeue / sacct, judges every
task of the DAG from its done-marker files, submits the ready tasks within the caps, writes the state / status files,
submits the next pass with --begin=now+10minutes and exits (between passes it holds one submit slot, no running slot).

  start (login node, one-shot sbatch line):  analysis/geo2/RUNBOOK.md 3
  python g2_launch.py            one pass, then the next pass (inside a SLURM job only)
  python g2_launch.py status     read only: judge every task, print the table and the sbatch line of each ready task
  python g2_launch.py selftest   read only: DAG, job files, scripts, unit order vs train_atoms.py, parsers (exit 0 / 3)

Task = (name, phase, sbatch argv from RUNBOOK_feat / RUNBOOK_lad / RUNBOOK_tests, done marker, deps, stop check).
Kinds: job (submitted), gate (a human / git step: the launcher only waits for it and never commits), check (computed
here: the Y GPU hours). Per pass, in DAG order, each task is
  done     its done marker holds once no dependency is still in progress (sticky in the state from then on; M3 atom
           workers also: last attempt exit 0; the C merges: and their own job ended exit 0) — O-36, O-37
  active   a job with the task's name is in squeue (a job submitted by hand under that name is adopted)
  stop     its report says stop (report json key; Y > 30 GPU-h, H projection > 2,500 core-h, smoke projection > 80
           GPU-h), or its last job ended with exit 3 / with a non-zero exit and a log line starting with 'STOP'
           (a job cancelled with scancel is a stop too: never resubmitted by the launcher)
  dead     lost twice (ended without its done marker and without a STOP: crash, exit 2, OOM, timeout, node failure)
  blocked  a dependency is stop / dead / blocked: never submitted, listed in results/STOP_<phase>.md
  todo     every dependency done -> submitted (a first loss is resubmitted once; an M3 atom worker goes to the GPU
           type of its previous attempt, which its unit fingerprints record)
Caps: `squeue -u $USER -h -r` lines (array elements and this launcher included) stay <= MAX_SUBMIT - 1 after each
submission (one slot for the next pass); <= 2 pending own tasks per class (cpu / gpu); MaxJobs 10 is SLURM's. Every
task: --time=48:00:00, --output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%j.out, -p cpu1,cpu2 or gpu1,gpu2,gpu3,gpu5,gpu6.
Writes: $GEO2/run_g2/{state.json, status.json, STATUS.txt}; analysis/geo2/results/STOP_<phase>.md while a STOP holds
(renamed STOP_<phase>.resolved_<time>.md once it no longer holds); results/HPC_efficiency.csv at the end.
Controls: $GEO2/run_g2/HALT (no submission, no next pass); $GEO2/run_g2/retry/<task> (forget that task's attempts and
sticky done once; the file is then removed). Decisions O-1 .. : results/DECISIONS_orch.md. Stdlib only; runs with the
reactot python by absolute path without conda activation, so the tasks inherit the environment of the starting shell.
"""
from __future__ import annotations

import ast
import errno
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent                    # analysis/geo2
REPO = HERE.parents[1]
AN = REPO / "analysis"
RES = HERE / "results"
GEO2 = Path("/gpfs/tmp_cpu2/yeseo1ee/geo2")               # O-3: the wrappers' fixed roots (geo2.slurm GEO2_SCRATCH,
GEO1 = Path("/gpfs/tmp_cpu2/yeseo1ee/geo")                # analysis/geo/jobs GEO_SCRATCH, d1_ladder/jobs
LAD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")           # D1LADDER_SCRATCH); never read from the environment
LOGS = GEO2 / "logs"
RUN = GEO2 / "run_g2"
STATE, STATUS_JSON, STATUS_TXT = RUN / "state.json", RUN / "status.json", RUN / "STATUS.txt"
LOCK, HALT, RETRY = RUN / "launch.lock", RUN / "HALT", RUN / "retry"
USER = os.environ.get("USER") or "yeseo1ee"
GPU_P, CPU_P, WALL = "gpu1,gpu2,gpu3,gpu5,gpu6", "cpu1,cpu2", "48:00:00"
MAX_SUBMIT, PEND_MAX, MAX_TRIES = 20, 2, 2               # O-5 (submit cap counts this launcher), O-6, O-7
ONCE_MIN, POLL_S, STOP_AFTER_H, UNKNOWN_AFTER_S = 10, 120, 46.0, 1800
Y_GPU_H_MAX, H_COREH_MAX, M3_GPU_H_MAX = 30.0, 2500.0, 80.0          # spec 12 / 14 limits
Y_SINCE = "2026-10-30T07:00:00"                          # sacct window of the Y jobs (decision Y-early, 07:5x)
PY_LAUNCH = "/home1/yeseo1ee/miniconda3/envs/reactot/bin/python"
LAUNCH_NAME = "geo2_launch"
MARK = "<!-- g2_launch STOP file -->"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
            "DEADLINE", "REVOKED"}

# ---------------------------------------------------------------- files the judges read (all read only)
D_GEO, D_PAIR, D_HF, D_LAD = AN / "geo", AN / "d1_pair", AN / "d1_hf", AN / "d1_ladder"
GEO1_RES, LAD_RES, STEPS, UNITS = D_GEO / "results", D_LAD / "results", RES / "F_steps", LAD / "units"
GEO1_COMMIT = [GEO1_RES / "E_report.json", GEO1_RES / "E_rxn.csv"] + [GEO1_RES / f"R_select_f{k}.json" for k in range(5)]
GEO1_PREDS = [GEO1 / "preds" / f"final_f{k}_s{s}.pkl" for k in range(5) for s in range(3)]
START_FILES = [LAD_RES / n for n in ("L_predictions.parquet", "L_report.json", "L_xgb_report.json", "E2_report.json",
                                     "E2_metrics.csv")] + [AN / "d1_g3" / "feat.py", AN / "d1_g3" / "g3_common.py",
                                                           AN / "d1_g2" / "results" / "labels_d1_hf_g2.parquet"]
SELFTEST_E = RES / "selftest" / "E_report.json"
Y_PLAN = GEO1_RES / "Y_plan.json"
SEEDS = (0, 1, 2)
Y_PKL = [GEO1 / "preds" / f"final_fm1_s{s}.pkl" for s in SEEDS]
Z_REPORT, T2_JSON, T3_JSON = RES / "Z_report.json", RES / "T_2.json", RES / "T_3.json"
H_PILOT, H_REPORT, H_PLAN = RES / "H_pilot.json", RES / "H_report.json", RES / "hf_geo" / "H_batches" / "plan.json"
G_REPORT, F_REPORT = RES / "G_report.json", RES / "F_report.json"
P_BUILD, P_GEO_REPORT = RES / "P_build_geo.json", RES / "P_geo_report.json"
C_REG = {p: RES / f"C_tests_geo_regress_{p}.json" for p in ("gpu", "cpu")}
C_FULL = {p: RES / f"C_tests_geo_{p}.json" for p in ("gpu", "cpu")}
C_REG_M, C_M = RES / "C_tests_geo_regress.json", RES / "C_tests_geo.json"
SMOKE_ATOMS = LAD_RES / "C_smoke_atoms_geo.json"
E3_SELF = RES / "e3_selftest" / "E3_selftest.json"
PREREG_MD, PREREG_JS, R_CHECK = RES / "PREREG_GEO2.md", RES / "prereg_geo2.json", RES / "R_check_geo2.json"
ARMS, MODEL_RANK, NF = ("S", "SH"), {"GNN": 0, "AW": 1, "TF": 2}, 5       # = train_atoms.py (selftest compares)
M3_UNITS = [f"L__geo__{a}__{m}__f{k}" for m, a, k in sorted(((m, a, k) for a in ARMS for m in MODEL_RANK
                                                             for k in range(NF)),
                                                            key=lambda u: (MODEL_RANK[u[0]], ARMS.index(u[1]), u[2]))]
N_M3, N_G, W_H = 4, 4, 5                                 # atom workers, G / F slices, H workers (= the runbooks)
XGB_UNITS = [UNITS / f"L__geo__{a}__XGB__f{k}.json" for a in ARMS for k in range(NF)]
L_XGB_GEO = LAD_RES / "L_xgb_report_geo.json"
COLLECT = [LAD_RES / "L_predictions_geo.parquet", LAD_RES / "L_val_geo.csv", LAD_RES / "L_report_geo.json"]
E_REPORT, E3_REPORT = RES / "E_report.json", RES / "E3_report.json"
FINAL_TASKS = ("geo2_e", "geo2_e3")

# job files: key -> (submit dir, job file relative to it, --export)  (RUNBOOK_feat 1, RUNBOOK_lad 0, RUNBOOK_tests 0)
W = dict(
    geo_gpu=(D_GEO, "jobs/gpu.slurm", "ALL"),
    geo2_gpu=(HERE, "../geo/jobs/gpu.slurm", "ALL"),                # Z: GPU env on a CPU partition (spec 4)
    geo2_d1=(HERE, "jobs/geo2.slurm", "ALL,GEO2ENV=d1"),
    geo2_reactot=(HERE, "jobs/geo2.slurm", "ALL,GEO2ENV=reactot"),
    pair_g=(D_PAIR, "jobs/g.slurm", "ALL,PYENV=venv5,D1PAIR_PHASE=F"),
    pair_py=(D_PAIR, "jobs/py.slurm", "ALL,PYENV=espley"),
    hf=(D_HF, "jobs/hf.slurm", "ALL"),
    lad_gpu=(D_LAD, "jobs/gpu.slurm", "ALL"),
    lad_reactot=(D_LAD, "jobs/lad.slurm", "ALL,LADENV=reactot"),
)
SPEC_ROW = {"START": "14 '0' (1 시작 조건)", "Y": "14 'Y' (3; 12 Y 30 GPU시간)", "Z": "14 'Z' (4-3)",
            "T": "14 'T' (5)", "G": "14 'G·F·H·P' (6-2 G)", "F": "14 'G·F·H·P' (6-2 F)",
            "H": "14 'G·F·H·P' (6-2 H; 6-1 pilot 2,500코어시간)", "P": "14 'P' / 'G·F·H·P' (6-2 P, 7)",
            "C": "14 'R·M3' (8-2 검사; 8-4 smoke 예상 80 GPU시간)", "R": "14 'R·M3' (8-3 사전 등록)",
            "M3": "14 'R·M3' (8-4 학습, collect, LADDER 결과 sha256 전후 비교)", "E": "14 'E' (9; selftest)"}


# ---------------------------------------------------------------- small helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def rj(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def short(x, n=600):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + " ..."


def write_atomic(path, text):
    """O-10: writes only under $GEO2/run_g2 and analysis/geo2/results."""
    path = Path(os.path.abspath(path))
    if not any(str(path).startswith(str(r) + os.sep) for r in (RUN, RES)):
        raise SystemExit(f"STOP: write path {path} outside {RUN} / {RES}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def wjson(path, obj):
    write_atomic(path, json.dumps(obj, indent=1, default=str))


def sh(args, cwd=None, timeout=300):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True, timeout=timeout).stdout


def git(*a):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")       # O-12: read-only git, never takes index.lock
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, env=env, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(a, 128, "", f"{type(e).__name__}: {e}")


def jobkey(j):
    head = str(j).split("_")[0].split(";")[0]
    return int(head) if head.isdigit() else 0


def parse_exit(s):
    """sacct ExitCode '3:0' -> (3, 0); anything else -> (None, None)."""
    try:
        a, b = str(s).strip().split(":")
        return int(a), int(b)
    except ValueError:
        return None, None


def gpu_type(tres):
    m = re.search(r"gres/gpu:([A-Za-z0-9_.\-]+)=", str(tres or ""))
    return m.group(1) if m else None


def n_gpu(tres):
    m = re.search(r"gres/gpu=(\d+)", str(tres or ""))
    return int(m.group(1)) if m else None


def stop_lines_of(text, n=6):
    """lines that start with 'STOP' (C.stop / L.stop / SystemExit('STOP: ...') messages) and the last 15 lines."""
    lines = text.splitlines()
    return [ln.strip()[:400] for ln in lines if ln.strip().startswith("STOP")][-n:], [ln[:300] for ln in lines[-15:]]


def log_of(name, jid):
    return LOGS / f"{name}_{jid}.out"


def read_log(name, jid):
    try:
        with open(log_of(name, jid), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 262144))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ---------------------------------------------------------------- judges: () -> ("done" | "stop" | None, reason)
def j_files(*ps):
    def f():
        miss = [rel(p) for p in ps if not Path(p).exists()]
        return ("done", "") if not miss else (None, f"missing {short(miss, 300)}")
    return f


def j_json(p, done, stop=None, why=None, newer_than=()):
    """report json p: stop(d) -> 'stop', done(d) -> 'done'. With newer_than, p is judged only when every one of those
    files exists and p is newer than all of them; else it is not judged (None) — O-9, O-38."""
    def f():
        d = rj(p)
        if d is None:
            return None, f"{rel(p)} missing"
        miss = [rel(q) for q in newer_than if not Path(q).exists()]
        if miss:                                         # O-38: no anchor yet = p belongs to an earlier step
            return None, f"{rel(p)} not judged: {short(miss, 300)} missing (it must be newer than them)"
        try:
            mt = Path(p).stat().st_mtime
            old = [rel(q) for q in newer_than if Path(q).stat().st_mtime >= mt]
        except OSError as e:
            return None, f"{rel(p)} not judged: stat failed ({e})"
        if old:
            return None, f"{rel(p)} older than {old} (stale)"
        w = short(why(d)) if why else ""
        if stop is not None and stop(d):
            return "stop", f"{rel(p)}: {w or 'stop'}"
        if done(d):
            return "done", ""
        return None, f"{rel(p)} not done yet{': ' + w if w else ''}"
    return f


def j_step(name, neutral=False):
    """results/F_steps/<name>.json (feat_geo.step): rc 0 done, rc 3 stop. neutral: only the stop part counts."""
    def f():
        d = rj(STEPS / f"{name}.json")
        rc = None if d is None else d.get("rc")
        if rc == 3:
            return "stop", f"F_steps/{name}.json rc 3: {d.get('fail_stage')}: {short(d.get('fail'), 500)}"
        if neutral or rc == 0:
            return "done", ""
        return None, f"F_steps/{name}.json " + ("missing" if d is None else f"rc {rc}")
    return f


def j_all(*js):
    def f():
        out = [j() for j in js]
        st = [w for s, w in out if s == "stop"]
        if st:
            return "stop", " | ".join(st)
        nd = [w for s, w in out if s != "done"]
        return ("done", "") if not nd else (None, nd[0])
    return f


def j_t(p):
    """T_2.json / T_3.json (feat_geo t_check): verdict pass -> done, fail -> stop (spec 5 / 14 T)."""
    return j_json(p, done=lambda d: d.get("pass") is True and d.get("verdict") == "pass",
                  stop=lambda d: d.get("verdict") == "fail",
                  why=lambda d: f"verdict {d.get('verdict')}; {d.get('fail_stage')}: {d.get('fail') or d.get('summary')}")


def j_part(p):
    """C_tests_geo{,_regress}_{gpu,cpu}.json (run_tests finish_geo): pass + final inputs -> done, fail -> stop."""
    def f():
        d = rj(p)
        if d is None:
            return None, f"{rel(p)} missing"
        if d.get("running"):
            return None, f"{rel(p)} says running (the part was killed?)"
        s = d.get("status")
        if s == "fail":
            bad = [c for c, v in (d.get("checks") or {}).items() if (v or {}).get("status") == "fail"]
            return "stop", f"{rel(p)} status fail: checks {bad}"
        if s == "pass" and d.get("final_inputs"):
            return "done", ""
        return None, f"{rel(p)} status {s}, final_inputs {d.get('final_inputs')}, abort {d.get('abort')}"
    return f


def j_merged(p, gpu_limit=False):
    """C_tests_geo{,_regress}.json (run_tests merge_geo): all_pass -> done, failed -> stop; with gpu_limit also the
    spec 8-4 smoke projection: gpu_estimate_geo.geo_gpu_hours (or C_smoke_atoms_geo.json estimate_geo) > 80 -> stop."""
    def f():
        d = rj(p)
        if d is None:
            return None, f"{rel(p)} missing"
        if d.get("failed"):
            return "stop", f"{rel(p)}: failed checks {d.get('failed')} (8-2 / 14 R·M3)"
        if gpu_limit:
            est = d.get("gpu_estimate_geo") or (rj(SMOKE_ATOMS) or {}).get("estimate_geo") or {}
            h = est.get("geo_gpu_hours")
            if isinstance(h, (int, float)) and h > M3_GPU_H_MAX:
                return "stop", (f"smoke GPU-hour projection {h:.1f} > {M3_GPU_H_MAX:.0f} GPU-h (spec 8-4 / 12 / 14 "
                                f"R·M3; {rel(p)} gpu_estimate_geo)")
        if d.get("all_pass") is True:
            return "done", ""
        return None, f"{rel(p)} status {d.get('status')}: {short(d.get('waiting_reasons'), 300)}"
    return f


def j_hip():
    pilot = j_all(j_step("h_inputs", neutral=True),
                  j_json(H_PILOT, done=lambda d: d.get("pass") is True and d.get("complete") is True,
                         stop=lambda d: bool(d.get("stop")), why=lambda d: d.get("stop_reasons")),
                  j_files(H_PLAN))

    def f():
        d = rj(H_PILOT) or {}
        pr = d.get("projection_core_h")
        if isinstance(pr, (int, float)) and pr > H_COREH_MAX:
            return "stop", (f"H pilot projection {pr:.0f} core-h > {H_COREH_MAX:.0f} (spec 6-1 / 12 / 14; "
                            f"{rel(H_PILOT)} projection_core_h)")
        return pilot()
    return f


def j_m3(k):
    us = M3_UNITS[k::N_M3]                               # = train_atoms unit_list(3)[k::4] (selftest compares)

    def f():
        miss = [u for u in us if not (UNITS / f"{u}.json").is_file()]
        return ("done", "") if not miss else (None, f"{len(miss)}/{len(us)} unit json missing ({miss[0]} ...)")
    return f


def committed_pushed(paths):
    """O-11: every file tracked, unchanged since HEAD (= the trainers' gate_prereg_geo2 rule), and its last commit on
    a remote-tracking branch (pushed; spec 1 0-1 / 8-3 'commit·push')."""
    why = []
    for p in paths:
        r = rel(p)
        if not Path(p).is_file():
            why.append(f"{r} missing")
            continue
        if git("ls-files", "--error-unmatch", "--", r).returncode != 0:
            why.append(f"{r} not committed")
            continue
        g = git("diff", "--quiet", "HEAD", "--", r)
        if g.returncode != 0:
            why.append(f"{r} modified since its commit" if g.returncode == 1 else f"{r}: git diff error {g.stderr[:200]}")
            continue
        h = git("log", "-1", "--format=%H", "--", r).stdout.strip()
        b = git("branch", "-r", "--contains", h) if h else None
        if not h or b.returncode != 0 or not b.stdout.strip():
            why.append(f"{r}: commit {h[:10] or '?'} not pushed (no remote-tracking branch contains it)")
    return why


def j_gate_geo1():
    miss = [rel(p) for p in START_FILES if not p.is_file()]
    if miss:
        return "stop", f"start conditions 0-2 .. 0-4: missing {miss}"
    st = ((rj(SELFTEST_E) or {}).get("selftest") or {}).get("all_pass")
    if st is not True:
        return "stop", f"start condition 0-6: {rel(SELFTEST_E)} selftest.all_pass = {st!r}"
    miss = [rel(p) for p in GEO1_PREDS if not p.is_file()]
    if miss:
        return None, f"0-1: GEO1 prediction files missing: {miss[:3]} ... ({len(miss)})"
    why = committed_pushed(GEO1_COMMIT)
    if why:
        return None, "0-1 human step: commit + push the GEO1 results: " + "; ".join(why[:4])
    return "done", ""


def j_gate_prereg():
    why = committed_pushed([PREREG_MD, PREREG_JS])
    if why:
        return None, ("R-2 human step (RUNBOOK_lad R-2): review, commit + push PREREG_GEO2.md + prereg_geo2.json "
                      "(+ the gated code): " + "; ".join(why))
    return "done", ""


def y_gpu_hours():
    """GPU hours of every Y attempt (sacct by job name since Y_SINCE; running jobs count their elapsed time)."""
    names = ",".join(f"geo2_y_s{s}" for s in SEEDS)
    try:
        out = sh(["sacct", "-X", "-n", "-P", "-S", Y_SINCE, f"--name={names}", "-o",
                  "JobID,JobName,State,ElapsedRaw,AllocTRES"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None, 0
    tot, n = 0.0, 0
    for ln in out.splitlines():
        f = ln.split("|")
        if len(f) < 5 or not f[3].strip().isdigit():
            continue
        tot += int(f[3]) * (n_gpu(f[4]) or 1) / 3600.0
        n += 1
    return tot, n


def j_y_budget():
    h, n = y_gpu_hours()
    if h is None:
        return None, "sacct failed (Y GPU hours unknown this pass)"
    if h > Y_GPU_H_MAX:
        return "stop", (f"Y used {h:.1f} GPU-h over {n} attempts > {Y_GPU_H_MAX:.0f} (spec 12 / 14 Y); the launcher never "
                        f"cancels jobs (ask the user)")
    miss = [rel(p) for p in Y_PKL if not p.is_file()]
    if miss:
        return None, f"Y running: {h:.1f} GPU-h so far ({n} attempts), {len(miss)} pkl missing"
    return "done", f"Y {h:.1f} GPU-h <= {Y_GPU_H_MAX:.0f} ({n} attempts)"


# ---------------------------------------------------------------- the DAG (RUNBOOK_feat / RUNBOOK_lad / RUNBOOK_tests)
def build_tasks():
    T = []

    def job(name, phase, w, args, deps=(), res="cpu", cpus=4, mem="16G", judge=None, prio=5, env="", exit0_done=False,
            pin_gpu=False, own_run=False):
        where, jobfile, exp = W[w]
        T.append(dict(name=name, phase=phase, kind="job", w=w, where=where, jobfile=jobfile, export=exp + env,
                      args=[str(a) for a in args], deps=list(deps), res=res, cpus=int(cpus), mem=mem, judge=judge,
                      prio=prio, exit0_done=exit0_done, pin_gpu=pin_gpu, own_run=own_run))

    def other(kind, name, phase, judge, deps=()):
        T.append(dict(name=name, phase=phase, kind=kind, deps=list(deps), judge=judge, prio=0, res=None))

    FG_HF = FG_PAIR = "../geo2/feat_geo.py"
    # Y (spec 3; DECISIONS Y-1: the GEO1 final env GEO_MAX_EP=300, GEO_PATIENCE=100000)
    job("geo2_y_plan", "Y", "geo_gpu", ["train_m1.py", "plan"], cpus=4, mem="32G", judge=j_files(Y_PLAN), prio=0)
    for s in SEEDS:
        job(f"geo2_y_s{s}", "Y", "geo_gpu", ["train_m1.py", "final", s], deps=["geo2_y_plan"], res="gpu", cpus=4,
            mem="32G", env=",GEO_MAX_EP=300,GEO_PATIENCE=100000", judge=j_files(Y_PKL[s]), prio=4)
    other("check", "geo2_y_budget", "Y", j_y_budget, deps=[f"geo2_y_s{s}" for s in SEEDS])
    # start conditions (spec 1; 0-1 = GEO1 committed + pushed is a human step)
    other("gate", "geo2_gate_geo1", "START", j_gate_geo1)
    # Z (spec 4)
    job("geo2_z", "Z", "geo2_gpu", ["build_geo.py", "build"], deps=["geo2_gate_geo1", "geo2_y_budget"], cpus=4,
        mem="32G", prio=0,
        judge=j_json(Z_REPORT, done=lambda d: d.get("stop") is False and (d.get("n_folders") or 0) > 0,
                     stop=lambda d: bool(d.get("stop")), why=lambda d: d.get("stop_reasons")))
    # T (spec 5; RUNBOOK_feat 3). O-13: T, the early C regressions and the E3 selftest wait for the spec 1 start
    # conditions (only Y was moved before them, by the user); they may run while Y trains (spec 8 order note).
    g0 = ["geo2_gate_geo1"]
    job("geo2_t2g", "T", "geo2_d1", ["feat_geo.py", "t2_g"], deps=g0, cpus=3, mem="8G", judge=j_step("t2_g"), prio=0)
    job("geo2_t2f", "T", "pair_g", [FG_PAIR, "t2_f"], deps=["geo2_t2g"], cpus=8, mem="32G", judge=j_t(T2_JSON), prio=0)
    job("geo2_t3", "T", "hf", [FG_HF, "t3"], deps=g0, cpus=3, mem="16G", judge=j_t(T3_JSON), prio=0)
    zt = ["geo2_z", "geo2_t2f", "geo2_t3"]
    # H (spec 6-1 / 6-2 H; RUNBOOK_feat 4)
    job("geo2_hip", "H", "hf", [FG_HF, "h_inputs", W_H, ":::", FG_HF, "h_pilot"], deps=zt, cpus=48, mem="180G",
        judge=j_hip(), prio=1)
    for k in range(W_H):
        job(f"geo2_hb{k}", "H", "hf", [FG_HF, "h_batch", k, W_H], deps=["geo2_hip"], cpus=48, mem="180G",
            judge=j_step(f"h_batch_w{k}of{W_H}"), prio=2)
    job("geo2_hr", "H", "hf", [FG_HF, "h_build", ":::", FG_HF, "h_report"], deps=[f"geo2_hb{k}" for k in range(W_H)],
        cpus=4, mem="32G", prio=0,
        judge=j_all(j_step("h_build", neutral=True),
                    j_json(H_REPORT, done=lambda d: d.get("complete") is True and d.get("stop") is False,
                           stop=lambda d: bool(d.get("stop")), why=lambda d: d.get("stop_reasons"))))
    # G, F (spec 6-2 G / F; RUNBOOK_feat 5)
    for i in range(N_G):
        job(f"geo2_g{i}", "G", "geo2_d1", ["feat_geo.py", "g", i, N_G], deps=zt, cpus=16, mem="32G",
            judge=j_step(f"g_{i:02d}of{N_G:02d}"), prio=6)
    job("geo2_gr", "G", "pair_g", [FG_PAIR, "g_report"], deps=[f"geo2_g{i}" for i in range(N_G)], cpus=16, mem="32G",
        prio=0, judge=j_all(j_step("g_report"),
                            j_json(G_REPORT, done=lambda d: d.get("complete") is True and d.get("stop") is False,
                                   stop=lambda d: bool(d.get("stop")), why=lambda d: d.get("stop_reasons"))))
    for i in range(N_G):
        job(f"geo2_fs{i}", "F", "pair_g", [FG_PAIR, "f_slice", i, N_G], deps=["geo2_gr"], cpus=32, mem="64G",
            judge=j_step(f"f_slice_{i:02d}of{N_G:02d}"), prio=6)
    job("geo2_fr", "F", "pair_g", [FG_PAIR, "f_rest"], deps=[f"geo2_fs{i}" for i in range(N_G)], cpus=48, mem="96G",
        prio=0, judge=j_all(j_step("f_report"), *[j_step(s, neutral=True) for s in ("f_merge", "f_atoms_00of01",
                                                                                       "f_gate", "f_tokens")]))
    # P + pair report + F report (spec 6-1 P, 7; RUNBOOK_feat 6)
    job("geo2_p", "P", "geo2_reactot", ["feat_geo.py", "p", ";;;", "feat_geo.py", "p_report", ";;;", "feat_geo.py",
                                        "report"], deps=["geo2_fr", "geo2_hr"], cpus=4, mem="32G", prio=0,
        judge=j_all(j_json(P_BUILD, done=lambda d: d.get("all_pass") is True, stop=lambda d: bool(d.get("stop")),
                           why=lambda d: d.get("stop_reason")),
                    j_json(P_GEO_REPORT, done=lambda d: d.get("stop") is False, stop=lambda d: bool(d.get("stop")),
                           why=lambda d: d.get("stop_reasons")),
                    j_json(F_REPORT, done=lambda d: d.get("complete") is True and d.get("stop") is False,
                           stop=lambda d: bool(d.get("stop")),
                           why=lambda d: d.get("stop_reasons") or d.get("incomplete"))))
    # C: regressions now (T-R), full tests after P (spec 8-2; RUNBOOK_tests 2)
    job("geo2_creg_g", "C", "lad_gpu", ["tests/run_tests.py", "geo", "gpu", "--regress-only"], deps=g0, res="gpu",
        cpus=4, mem="32G", judge=j_part(C_REG["gpu"]), prio=5)
    job("geo2_creg_c", "C", "pair_py", ["../d1_ladder/tests/run_tests.py", "geo", "cpu", "--regress-only"], deps=g0,
        cpus=32, mem="64G", judge=j_part(C_REG["cpu"]), prio=5)
    job("geo2_creg_m", "C", "lad_reactot", ["tests/run_tests.py", "geo", "merge", "--regress-only"],
        deps=["geo2_creg_g", "geo2_creg_c"], cpus=1, mem="4G", judge=j_merged(C_REG_M), prio=0, own_run=True)
    job("geo2_ctest_g", "C", "lad_gpu", ["tests/run_tests.py", "geo", "gpu"], deps=["geo2_p", "geo2_creg_m"],
        res="gpu", cpus=4, mem="32G", judge=j_part(C_FULL["gpu"]), prio=5)
    job("geo2_ctest_c", "C", "pair_py", ["../d1_ladder/tests/run_tests.py", "geo", "cpu"],
        deps=["geo2_p", "geo2_creg_m"], cpus=32, mem="64G", judge=j_part(C_FULL["cpu"]), prio=5)
    job("geo2_ctest_m", "C", "lad_reactot", ["tests/run_tests.py", "geo", "merge"],       # O-36: a fresh merge,
        deps=["geo2_ctest_g", "geo2_ctest_c"], cpus=1, mem="4G", judge=j_merged(C_M, gpu_limit=True), prio=0,
        own_run=True)                                    # right before the PREREG write (RUNBOOK_tests C)
    # E3 selftest now (RUNBOOK_tests E-T)
    job("geo2_e3t", "E", "pair_py", ["../d1_ladder/evaluate.py", "e3-selftest"], deps=g0, cpus=4, mem="32G", prio=5,
        judge=j_json(E3_SELF, done=lambda d: d.get("ok") is True, stop=lambda d: d.get("ok") is False,
                     why=lambda d: {k: v for k, v in (d.get("checks") or {}).items() if not v}))
    # R (spec 8-3; RUNBOOK_lad R-1 .. R-3): write -> GATE commit + push (human) -> check
    job("geo2_R", "R", "lad_reactot", ["../geo2/prereg_geo2.py", "write"],
        deps=["geo2_ctest_m", "geo2_p", "geo2_z", "geo2_y_budget"], cpus=1, mem="8G",
        judge=j_files(PREREG_MD, PREREG_JS), prio=0)
    other("gate", "geo2_gate_prereg", "R", j_gate_prereg, deps=["geo2_R"])
    job("geo2_Rc", "R", "lad_reactot", ["../geo2/prereg_geo2.py", "check"], deps=["geo2_gate_prereg"], cpus=1,
        mem="4G", prio=0, judge=j_json(R_CHECK, done=lambda d: d.get("ok") is True, stop=lambda d: d.get("ok") is False,
                                       why=lambda d: d.get("bad"), newer_than=[PREREG_JS]))
    # M3 (spec 8-4; RUNBOOK_lad M-1 .. M-4) + the spec 14 before / after sha256 comparison (prereg check again)
    job("geo2_x3", "M3", "pair_py", ["../d1_ladder/train_xgb.py", "run", 0, 1, "--stage", 3], deps=["geo2_Rc"],
        cpus=32, mem="64G", judge=j_files(*XGB_UNITS), prio=3)
    job("geo2_x3c", "M3", "pair_py", ["../d1_ladder/train_xgb.py", "check", "--stage", 3], deps=["geo2_x3"], cpus=2,
        mem="16G", prio=0, judge=j_json(L_XGB_GEO, done=lambda d: d.get("ok") is True, why=lambda d: d.get("bad"),
                                        newer_than=XGB_UNITS))
    for k in range(N_M3):
        job(f"geo2_m3_{k}", "M3", "lad_gpu", ["train_atoms.py", "run", k, N_M3, "--stage", 3], deps=["geo2_Rc"],
            res="gpu", cpus=4, mem="16G", judge=j_m3(k), prio=3, exit0_done=True, pin_gpu=True)
    job("geo2_m3col", "M3", "lad_gpu", ["train_atoms.py", "collect", "--stage", 3, "--with-xgb"],
        deps=[f"geo2_m3_{k}" for k in range(N_M3)] + ["geo2_x3c"], cpus=2, mem="24G", judge=j_files(*COLLECT), prio=0)
    job("geo2_Rc2", "M3", "lad_reactot", ["../geo2/prereg_geo2.py", "check"], deps=["geo2_m3col"], cpus=1, mem="4G",
        prio=0, judge=j_json(R_CHECK, done=lambda d: d.get("ok") is True, stop=lambda d: d.get("ok") is False,
                             why=lambda d: d.get("bad"), newer_than=[COLLECT[2]]))
    # E (spec 9; eval_geo2 run first so that e3 records its cross-check, RUNBOOK_tests E3); O-39: collect is a dep too
    em = ["geo2_m3col", "geo2_Rc2"]
    job("geo2_e", "E", "pair_py", ["../geo2/eval_geo2.py", "run"], deps=em, cpus=4, mem="32G", prio=0,
        judge=j_json(E_REPORT, done=lambda d: d.get("mode") == "run", newer_than=[COLLECT[0]]))
    job("geo2_e3", "E", "pair_py", ["../d1_ladder/evaluate.py", "e3"], deps=["geo2_e", "geo2_e3t"] + em,
        cpus=4, mem="32G", prio=0, judge=j_json(E3_REPORT, done=lambda d: True, newer_than=[COLLECT[0]]))
    return T


# ---------------------------------------------------------------- SLURM
def squeue():
    out = sh(["squeue", "-u", USER, "-h", "-r", "-o", "%i|%j|%T"])
    return [tuple(ln.strip().split("|")[:3]) for ln in out.splitlines() if ln.count("|") >= 2]


def sacct_ids(ids):
    """{jobid: dict(state, exit, signal, elapsed_s, tres, end, terminal)}; None when sacct fails."""
    out = {}
    ids = sorted(set(ids), key=jobkey)
    for i in range(0, len(ids), 80):
        try:
            txt = sh(["sacct", "-X", "-n", "-P", "-j", ",".join(ids[i:i + 80]), "-o",
                      "JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,End"])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            print(f"[g2_launch] sacct failed: {getattr(e, 'stderr', e)}", flush=True)
            return None
        for ln in txt.splitlines():
            f = ln.split("|")
            if len(f) < 7:
                continue
            state = f[2].split()[0] if f[2].strip() else ""
            ex, sig = parse_exit(f[3])
            out[f[0].strip()] = dict(jobid=f[0].strip(), name=f[1], state=state, exit=ex, signal=sig,
                                     elapsed_s=int(f[4]) if f[4].strip().isdigit() else None, tres=f[5], end=f[6],
                                     terminal=state in TERMINAL)
    return out


def partitions_with(gt):
    try:
        out = sh(["sinfo", "-h", "-p", GPU_P, "-o", "%P|%G"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    have = {ln.split("|")[0].strip().rstrip("*") for ln in out.splitlines()
            if "|" in ln and re.search(rf"gpu:{re.escape(gt)}(:|\(|$)", ln.split("|")[1].strip())}
    return ",".join(p for p in GPU_P.split(",") if p in have) or None


def sbatch_argv(t, rec, resolve_pin=True):
    if t["res"] == "gpu":
        parts, gres = GPU_P, "gpu:1"
        gt = rec.get("gpu_type") if t.get("pin_gpu") else None
        if gt:                                           # O-8: the unit fingerprint records the GPU name
            parts, gres = (partitions_with(gt) if resolve_pin else None) or GPU_P, f"gpu:{gt}:1"
        res = ["-p", parts, f"--gres={gres}"]
    else:
        res = ["-p", CPU_P]
    return ["sbatch", "--parsable", *res, "-J", t["name"], f"--time={WALL}", f"--cpus-per-task={t['cpus']}",
            f"--mem={t['mem']}", f"--output={LOGS}/%x_%j.out", f"--export={t['export']}", t["jobfile"], *t["args"]]


def shell_line(t, rec=None, resolve_pin=False):
    a = sbatch_argv(t, rec or {}, resolve_pin=resolve_pin)
    a.remove("--parsable")
    return f"cd {t['where']} && " + " ".join(shlex.quote(x) for x in a)


def submit(t, rec):
    out = sh(sbatch_argv(t, rec), cwd=str(t["where"]))
    return out.strip().split(";")[0]


RELAUNCH = f"cd {HERE} && PYTHONDONTWRITEBYTECODE=1 exec {PY_LAUNCH} g2_launch.py"


def relaunch(rows):
    """O-2: the next pass in ONCE_MIN minutes (--begin), unless another launcher pass is already pending."""
    me = os.environ.get("SLURM_JOB_ID", "")
    other = [j for j, n, s in rows if n == LAUNCH_NAME and j != me and s == "PENDING"]
    if other:
        print(f"[g2_launch] next pass already queued ({other}); not submitting another", flush=True)
        return True
    try:
        out = sh(["sbatch", "--parsable", "-p", CPU_P, "-J", LAUNCH_NAME, f"--time={WALL}", "--cpus-per-task=1",
                  "--mem=2G", f"--output={LOGS}/%x_%j.out", "--export=ALL", f"--begin=now+{ONCE_MIN}minutes",
                  "--wrap", RELAUNCH], cwd=str(HERE))
        print(f"[g2_launch] next pass {out.strip()} (begins in {ONCE_MIN} min)", flush=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[g2_launch] relaunch failed: {getattr(e, 'stderr', e)}", flush=True)
        return False


# ---------------------------------------------------------------- state and evaluation
def load_state():
    st = rj(STATE) or {}
    st.setdefault("tasks", {})
    st.setdefault("meta", {})
    return st


def apply_retries(st, names):
    if not RETRY.is_dir():
        return
    for f in sorted(RETRY.iterdir()):
        n = f.name
        if n in names:
            rec = st["tasks"].get(n) or {}
            hist = (rec.pop("history", None) or []) + [dict(rec, reset=now())]
            st["tasks"][n] = dict(tries=0, jobids=list(rec.get("jobids") or []), history=hist[-10:],
                                  retry_requested=now(), **({"gpu_type": rec["gpu_type"]} if rec.get("gpu_type") else {}))
            print(f"[g2_launch] retry file: {n} forgotten (attempts / done reset)", flush=True)
        else:
            print(f"[g2_launch] retry file {f.name}: no such task (removed)", flush=True)
        try:
            f.unlink()
        except OSError:
            pass


def outcome(rec, acct, t_now):
    """the latest attempt: a terminal record, 'open' (not final in sacct yet) or None (never submitted)."""
    jid = rec.get("jobid")
    if not jid:
        return None
    last = rec.get("last") or {}
    if last.get("jobid") == jid and last.get("terminal"):
        return last
    info = (acct or {}).get(jid)
    if info:
        rec["last"] = info
        gt = gpu_type(info.get("tres"))
        if gt:
            rec["gpu_type"] = gt
        return info if info["terminal"] else "open"
    if acct is not None and t_now - float(rec.get("submitted_ts") or t_now) > UNKNOWN_AFTER_S:
        rec["last"] = dict(jobid=jid, state="UNKNOWN", exit=None, signal=None, terminal=True,
                           note="neither in squeue nor in sacct")
        return rec["last"]
    return "open"


def evaluate(T, st, rows, acct, t_now):
    active = {}
    for jid, name, state in rows:
        active.setdefault(name, []).append((jid, state))
    S = {}
    for t in T:
        n = t["name"]
        rec = st["tasks"].setdefault(n, dict(tries=0))
        deps = t["deps"]
        s = dict(status=None, reason="")
        if rec.get("done"):
            s.update(status="done", reason=rec.get("done_note", ""))
        elif t["kind"] == "job" and n in active:
            for jid, _ in sorted(active[n], key=lambda x: jobkey(x[0])):
                if jid not in rec.setdefault("jobids", []):          # O-4: adopted (submitted by hand / before)
                    rec["jobids"].append(jid)
                    rec.update(jobid=jid, tries=int(rec.get("tries", 0)) + 1, submitted_ts=t_now)
                    rec.setdefault("adopted", []).append(jid)
            s.update(status="active", reason=", ".join(f"{j} {x}" for j, x in active[n]))
        else:
            try:
                j, why = t["judge"]()
            except Exception as e:                               # noqa: BLE001 — a judge must never stop the pass
                j, why = None, f"judge error {type(e).__name__}: {e}"
            if j == "done":
                early = [d for d in deps if S[d]["status"] not in ("done", "stop", "dead", "blocked")]
                if early:                                        # O-37: a marker ahead of its dependencies
                    j, why = None, f"marker holds, not counted while {why_deps(S, early)} in progress (O-37)"
                elif t.get("own_run"):                           # O-36: the marker alone is not enough
                    a0 = outcome(rec, acct, t_now)
                    if not (isinstance(a0, dict) and a0.get("state") == "COMPLETED" and a0.get("exit") == 0):
                        j, why = None, "marker holds, but done also needs this task's own job ended exit 0 (O-36)"
            if j == "done":
                rec.update(done=now(), done_note=why)
                s.update(status="done", reason=why)
            elif j == "stop":
                s.update(status="stop", reason=why, kind_of_stop="report")
            elif any(S[d]["status"] in ("stop", "dead", "blocked") for d in deps):
                s.update(status="blocked", reason="dependency " + ", ".join(
                    d for d in deps if S[d]["status"] in ("stop", "dead", "blocked")))
            elif t["kind"] in ("gate", "check"):
                ready = all(S[d]["status"] == "done" for d in deps)
                s.update(status=t["kind"] if ready or t["kind"] == "check" else "wait", reason=why)
            else:
                a = outcome(rec, acct, t_now)
                ready = all(S[d]["status"] == "done" for d in deps)
                if a == "open":
                    s.update(status="ending", reason=f"job {rec.get('jobid')} left squeue, sacct not final yet")
                elif a is None:
                    s.update(status="todo" if ready else "wait", reason="" if ready else f"needs {why_deps(S, deps)}")
                else:
                    ex, state = a.get("exit"), a.get("state")
                    stops, tail = stop_lines_of(read_log(n, a["jobid"]))
                    lastw = f"job {a['jobid']} {state} exit {ex}"
                    if t.get("exit0_done") and ex == 0 and state == "COMPLETED":
                        rec.update(done=now(), done_note=f"{lastw} (exit 0 = done for this task)")
                        s.update(status="done", reason=rec["done_note"])
                    elif ex == 3 or (ex not in (0, None) and stops):
                        s.update(status="stop", reason=f"{lastw}; {why}", stop_lines=stops, tail=tail,
                                 kind_of_stop="exit")
                    elif state == "CANCELLED":                       # O-7: scancel is a human decision, never undone
                        s.update(status="stop", reason=f"{lastw} (cancelled: not resubmitted; touch {RETRY}/{n} to "
                                                       f"run it again); marker: {why}", stop_lines=stops, tail=tail,
                                 kind_of_stop="cancelled")
                    elif int(rec.get("tries", 0)) >= MAX_TRIES:
                        s.update(status="dead", reason=f"lost {rec.get('tries')} times; last {lastw}; marker: {why}",
                                 stop_lines=stops, tail=tail)
                    elif ready:
                        s.update(status="todo", reason=f"retry after {lastw} (marker: {why})")
                    else:
                        s.update(status="wait", reason=f"needs {why_deps(S, deps)}")
        S[n] = s
    return S


def why_deps(S, deps):
    return ", ".join(f"{d} ({S[d]['status']})" for d in deps if S[d]["status"] != "done")


def descendants(T, roots):
    kids = {}
    for t in T:
        for d in t["deps"]:
            kids.setdefault(d, []).append(t["name"])
    out, todo = [], list(roots)
    while todo:
        for k in kids.get(todo.pop(), []):
            if k not in out:
                out.append(k)
                todo.append(k)
    return out


# ---------------------------------------------------------------- reports
def write_stops(T, S, st):
    by_phase = {}
    for t in T:
        if S[t["name"]]["status"] in ("stop", "dead"):
            by_phase.setdefault(t["phase"], []).append(t)
    since = st["meta"].setdefault("stop_since", {})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for ph, ts in by_phase.items():
        since.setdefault(ph, now())
        L = [f"# STOP {ph} — GEO2 launcher", "", MARK, "",
             f"Since {since[ph]} (analysis/geo2/g2_launch.py; docs/specs/GEO2.md {SPEC_ROW.get(ph, '14')}). The "
             "launcher submits none of the halted dependants below and keeps the independent branches running. Report "
             "what went wrong (task, file and line, options) to the user and wait (spec 14). Never edit labels or "
             "existing result files.", "", "## Stopped tasks"]
        for t in ts:
            s, rec = S[t["name"]], st["tasks"].get(t["name"]) or {}
            L.append(f"- `{t['name']}` — {s['status']} ({s.get('kind_of_stop', 'lost twice')}): {s['reason']}")
            if rec.get("jobids"):
                L.append(f"  - jobs {', '.join(rec['jobids'])}; log `{log_of(t['name'], rec.get('jobid'))}`")
            for x in s.get("stop_lines") or []:
                L.append(f"  - log: `{x}`")
            if s["status"] == "dead":
                L += ["  - log tail:"] + [f"    `{x}`" for x in (s.get("tail") or [])]
            if t["kind"] == "job":
                L.append(f"  - by hand, after the fix only: `{shell_line(t, rec)}`")
        halted = [d for d in descendants(T, [t["name"] for t in ts]) if S[d]["status"] == "blocked"]
        L += ["", "## Halted dependants", ", ".join(f"`{d}`" for d in halted) or "(none)", "", "## Resume",
              f"- After the fix (and a user decision where spec 14 asks for one): move the STOP report aside if a report "
              f"json says stop, then `touch {RETRY}/<task>`; the next pass forgets the task's attempts and judges it "
              f"again. Or resubmit the line above by hand with the same `-J` name: the launcher adopts that job.",
              f"- `touch {HALT}` stops the chain (no submission, no next pass); `rm` it and resubmit the start line "
              f"(analysis/geo2/RUNBOOK.md 3) to continue."]
        text = "\n".join(L) + "\n"
        p = RES / f"STOP_{ph}.md"
        try:
            same = p.is_file() and p.read_text(encoding="utf-8") == text
        except (OSError, ValueError):
            same = False
        if not same:
            write_atomic(p, text)
            print(f"[g2_launch] wrote {rel(p)}", flush=True)
    for p in sorted(RES.glob("STOP_*.md")):
        ph = p.name[len("STOP_"):-len(".md")]
        if ".resolved_" in ph or ph in by_phase:
            continue
        try:
            mine = MARK in p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            mine = False
        if mine:
            os.replace(p, p.with_name(f"STOP_{ph}.resolved_{stamp}.md"))
            since.pop(ph, None)
            print(f"[g2_launch] {p.name} no longer holds -> STOP_{ph}.resolved_{stamp}.md", flush=True)


def hpc_csv(st):
    ids = sorted({j for r in st["tasks"].values() for j in (r.get("jobids") or [])}, key=jobkey)
    if not ids:
        return
    rows = []
    for i in range(0, len(ids), 80):
        try:
            rows.append(sh(["sacct", "-X", "-n", "-P", "-j", ",".join(ids[i:i + 80]), "-o",
                            "JobID,JobName,State,Elapsed,ElapsedRaw,CPUTimeRAW,AllocTRES"]))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            print(f"[g2_launch] sacct failed, HPC_efficiency.csv not written: {getattr(e, 'stderr', e)}", flush=True)
            return
    write_atomic(RES / "HPC_efficiency.csv",
                 "job_id|job_name|state|elapsed|elapsed_s|cputime_raw_s|alloc_tres\n" + "".join(rows))


def table(T, S, st):
    out = [f"{'task':<17} {'phase':<5} {'status':<8} {'tries':>5} {'job':>9}  reason"]
    for t in T:
        s, rec = S[t["name"]], st["tasks"].get(t["name"]) or {}
        out.append(f"{t['name']:<17} {t['phase']:<5} {s['status']:<8} {rec.get('tries', 0):>5} "
                   f"{str(rec.get('jobid') or ''):>9}  {short(s['reason'] or '', 160)}")
    return "\n".join(out)


def write_status(T, S, st, info):
    counts = {}
    for s in S.values():
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    doc = dict(written=now(), launcher_job=os.environ.get("SLURM_JOB_ID"), counts=counts, **info,
               gates_waiting={t["name"]: S[t["name"]]["reason"] for t in T if S[t["name"]]["status"] == "gate"},
               stops={t["name"]: S[t["name"]]["reason"] for t in T if S[t["name"]]["status"] in ("stop", "dead")},
               tasks={t["name"]: dict(phase=t["phase"], status=S[t["name"]]["status"], reason=S[t["name"]]["reason"],
                                      tries=(st["tasks"].get(t["name"]) or {}).get("tries", 0),
                                      jobid=(st["tasks"].get(t["name"]) or {}).get("jobid")) for t in T})
    wjson(STATUS_JSON, doc)
    head = (f"GEO2 launcher status {doc['written']} (pass job {doc['launcher_job']}): {counts}; queue {info.get('nq')} "
            f"/ {MAX_SUBMIT}; pending own {info.get('pending')}; submitted {info.get('submitted')}; "
            f"halt {info.get('halt')}; outcome {info.get('outcome')}")
    gates = "".join(f"\nGATE {k}: {v}" for k, v in doc["gates_waiting"].items())
    stops = "".join(f"\nSTOP {k}: {short(v, 300)}" for k, v in doc["stops"].items())
    write_atomic(STATUS_TXT, head + gates + stops + "\n\n" + table(T, S, st) + "\n")


# ---------------------------------------------------------------- one pass
def one_pass(mutate=True):
    """-> (exit code, run another pass?)"""
    T = build_tasks()
    names = {t["name"] for t in T}
    st = load_state()
    if mutate:
        apply_retries(st, names)
    halt = HALT.exists()
    try:
        rows = squeue()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[g2_launch] squeue failed ({getattr(e, 'stderr', e)}): nothing judged this pass", flush=True)
        return (0, True) if mutate else (2, False)
    t_now = time.time()
    ids = [r["jobid"] for r in st["tasks"].values() if r.get("jobid") and not r.get("done")
           and not ((r.get("last") or {}).get("jobid") == r["jobid"] and (r.get("last") or {}).get("terminal"))]
    acct = sacct_ids(ids) if ids else {}
    S = evaluate(T, st, rows, acct, t_now)
    byname = {t["name"]: t for t in T}
    todo = sorted((t for t in T if S[t["name"]]["status"] == "todo"), key=lambda t: (t["prio"], T.index(t)))
    nq = len(rows)
    pend = {"gpu": 0, "cpu": 0}
    for _, n, s in rows:
        if n in byname and byname[n]["kind"] == "job" and s == "PENDING":
            pend[byname[n]["res"]] += 1
    submitted = []
    if mutate and not halt:
        for t in todo:
            if nq >= MAX_SUBMIT - 1:                         # this pass is counted in nq; one slot for the next pass
                print(f"[g2_launch] submit cap: {nq} queued", flush=True)
                break
            if pend[t["res"]] >= PEND_MAX:                   # O-6: <= 2 pending own tasks per class
                continue
            rec = st["tasks"][t["name"]]
            try:
                jid = submit(t, rec)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                print(f"[g2_launch] sbatch {t['name']} failed: {getattr(e, 'stderr', e)}", flush=True)
                break
            rec.setdefault("jobids", []).append(jid)
            rec.update(jobid=jid, tries=int(rec.get("tries", 0)) + 1, submitted=now(), submitted_ts=time.time())
            rec.pop("last", None)
            S[t["name"]].update(status="active", reason=f"{jid} submitted")
            nq += 1
            pend[t["res"]] += 1
            submitted.append(f"{t['name']}={jid}")
            print(f"[{now()}] submitted {t['name']} -> {jid}" + (f" (gpu {rec['gpu_type']})" if rec.get("gpu_type")
                                                                and t.get("pin_gpu") else ""), flush=True)
            wjson(STATE, st)
    finished = all(S[n]["status"] == "done" for n in FINAL_TASKS)
    live = any(s["status"] in ("active", "ending", "todo", "gate", "check") for s in S.values())
    outcome_ = "finished" if finished else ("halted" if halt else ("running" if live else "stalled"))
    if not mutate:
        print(table(T, S, st), flush=True)
        for t in todo:
            print(f"READY {t['name']}: {shell_line(t, st['tasks'].get(t['name']), resolve_pin=True)}", flush=True)
        print(f"queue {nq}/{MAX_SUBMIT}, pending own {pend}, halt {halt}, outcome {outcome_}", flush=True)
        return 0, False
    st["meta"].update(last_pass=now(), passes=int(st["meta"].get("passes", 0)) + 1,
                      last_job=os.environ.get("SLURM_JOB_ID"), outcome=outcome_)
    write_stops(T, S, st)
    wjson(STATE, st)
    write_status(T, S, st, dict(nq=nq, pending=pend, submitted=submitted, halt=halt, outcome=outcome_))
    counts = {}
    for s in S.values():
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print(f"[{now()}] pass: {counts}; submitted {submitted or 'none'}; outcome {outcome_}", flush=True)
    for t in T:
        if S[t["name"]]["status"] in ("gate", "stop", "dead"):
            print(f"  {S[t['name']]['status'].upper()} {t['name']}: {short(S[t['name']]['reason'], 400)}", flush=True)
    if finished:
        hpc_csv(st)
        print("[g2_launch] E finished (eval_geo2 run + evaluate e3): the orchestrator writes SUMMARY_GEO2.md "
              "(spec 11; RUNBOOK.md 6)", flush=True)
        return 0, False
    if halt:
        print(f"[g2_launch] {HALT} present: nothing submitted, no next pass", flush=True)
        return 0, False
    if not live:
        hpc_csv(st)
        print("[g2_launch] STOP: nothing can progress (see analysis/geo2/results/STOP_*.md); no next pass", flush=True)
        return 3, False
    return 0, True


# ---------------------------------------------------------------- commands
def cmd_run():
    RUN.mkdir(parents=True, exist_ok=True)
    RETRY.mkdir(parents=True, exist_ok=True)
    lk = open(LOCK, "a")
    try:
        fcntl.lockf(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)      # O-2: one pass at a time
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EACCES):
            print("[g2_launch] another pass holds the lock: exit (that pass submits the next one)", flush=True)
            return 0
        print(f"[g2_launch] lock not available on this file system ({e}): continuing without it", flush=True)
    t0 = time.time()
    while True:
        try:
            rc, again = one_pass()
        except Exception:                                    # noqa: BLE001 — a launcher bug: STOP file, no next pass
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            try:
                write_atomic(RES / "STOP_LAUNCHER.md",
                             f"# STOP LAUNCHER — GEO2 launcher\n\n{MARK}\n\nPass job {os.environ.get('SLURM_JOB_ID')} "
                             f"crashed at {now()}; no next pass was submitted. The tasks already queued keep running. "
                             f"Fix analysis/geo2/g2_launch.py (report first), then resubmit the start line "
                             f"(analysis/geo2/RUNBOOK.md 3). This file is renamed STOP_LAUNCHER.resolved_<time>.md by "
                             f"the next good pass.\n\n```\n{tb[-6000:]}```\n")
            except (OSError, SystemExit):
                pass
            return 3
        if not again:
            return rc
        try:
            rows = squeue()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            rows = []
        if relaunch(rows):
            return rc
        if time.time() - t0 > STOP_AFTER_H * 3600:           # 46 h of failed relaunches: give up, by hand
            print("[g2_launch] relaunch failing for 46 h: resubmit the start line by hand (RUNBOOK.md 3)", flush=True)
            return 2
        time.sleep(POLL_S)                                   # r_launch: keep polling (compute node) until it works


def cmd_selftest():
    T = build_tasks()
    bad, seen = [], set()
    for t in T:
        n = t["name"]
        if n in seen:
            bad.append(f"duplicate task {n}")
        for d in t["deps"]:
            if d not in seen:
                bad.append(f"{n}: dependency {d} unknown or later in the list")
        seen.add(n)
        if t["phase"] not in SPEC_ROW or t["kind"] not in ("job", "gate", "check") or not callable(t["judge"]):
            bad.append(f"{n}: phase / kind / judge")
        if not n.startswith("geo2_") or n == LAUNCH_NAME:
            bad.append(f"{n}: name")
        if t["kind"] != "job":
            continue
        if not t["where"].is_dir() or not (t["where"] / t["jobfile"]).is_file():
            bad.append(f"{n}: job file {t['where'] / t['jobfile']} missing")
        for a in t["args"]:
            if a.endswith(".py") and not (t["where"] / a).is_file():
                bad.append(f"{n}: script {t['where'] / a} missing")
        a = sbatch_argv(t, dict(gpu_type="a10"), resolve_pin=False)
        if f"--time={WALL}" not in a or a[a.index("-p") + 1] not in (GPU_P, CPU_P) or \
                f"--output={LOGS}/%x_%j.out" not in a or not t["export"].startswith("ALL") or \
                (t["res"] == "gpu") != any(x.startswith("--gres=gpu:") for x in a):
            bad.append(f"{n}: sbatch line {a}")
        print(f"{n:<17} {shell_line(t)}", flush=True)
    src = (D_LAD / "train_atoms.py").read_text(encoding="utf-8")
    mr = re.search(r"^MODEL_RANK\s*=\s*(\{[^}]*\})", src, re.M)
    ar = re.search(r"^ARMS\s*=\s*(\([^)]*\))", src, re.M)
    if not mr or not ar or ast.literal_eval(mr.group(1)) != MODEL_RANK or tuple(ast.literal_eval(ar.group(1))) != ARMS:
        bad.append("MODEL_RANK / ARMS differ from analysis/d1_ladder/train_atoms.py (M3 worker unit lists)")
    lst = GEO2 / "regress" / "list_stage3.txt"
    if lst.is_file() and [x for x in lst.read_text(encoding="utf-8").split() if x.startswith("L__geo__")] != M3_UNITS:
        bad.append(f"{lst} != the launcher's stage-3 unit order")
    checks = [(parse_exit("3:0"), (3, 0)), (parse_exit("0:15"), (0, 15)), (parse_exit(""), (None, None)),
              (gpu_type("billing=4,cpu=4,gres/gpu:a10=1,gres/gpu=1,node=1"), "a10"), (gpu_type("cpu=1,node=1"), None),
              (n_gpu("billing=4,cpu=4,gres/gpu:a10=1,gres/gpu=1,node=1"), 1),
              (stop_lines_of("x\n  STOP: a b\nok\n")[0], ["STOP: a b"]), (len(M3_UNITS), 30),
              (M3_UNITS[:2], ["L__geo__S__GNN__f0", "L__geo__S__GNN__f1"]), (len(set(M3_UNITS)), 30),
              (jobkey("1053846"), 1053846), (jobkey("12_3"), 12)]
    bad += [f"parser check {i}: {a!r} != {b!r}" for i, (a, b) in enumerate(checks) if a != b]
    for p in (Path(PY_LAUNCH), LOGS):
        if not p.exists():
            bad.append(f"{p} missing")
    print(json.dumps(dict(n_tasks=len(T), n_jobs=sum(t["kind"] == "job" for t in T), problems=bad,
                          all_pass=not bad), indent=1), flush=True)
    return 0 if not bad else 3


def main(argv):
    for s in (sys.stdout, sys.stderr):                       # O-35: logs stay readable in any node locale
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    if not os.environ.get("SLURM_JOB_ID"):
        print("refused: g2_launch.py runs only inside a SLURM job (CLAUDE.md: the login node is read only); "
              "see analysis/geo2/RUNBOOK.md 3", file=sys.stderr)
        return 2
    if argv == ["status"]:
        return one_pass(mutate=False)[0]
    if argv == ["selftest"]:
        return cmd_selftest()
    if argv:
        print(__doc__, file=sys.stderr)
        return 2
    return cmd_run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
