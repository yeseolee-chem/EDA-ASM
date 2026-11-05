#!/usr/bin/env python3
r"""pipeline.py — DEPLOY-V2 (MLTS) automatic stage launcher (docs/specs/DEPLOY_V2_MLTS.md order 0 -> I -> G -> Z -> FT
-> A -> H -> C -> R -> K -> E -> X -> W -> P -> S; operator notes PIPELINE.md). Compute nodes only: `once` runs inside
jobs/pipeline.slurm (a 1-CPU cpu1/cpu2 job that queues its successor with --begin=now+10minutes and
--dependency=singleton: CLAUDE.md option 2); `dry` / `status` through jobs/sbx.sh; `lists` as a small CPU job.
Standard library + mlt_common only at import (pandas / numpy only inside `lists`), so MLTENV=reactot starts in < 1 s.

CLI (from analysis/mlts)
  once     one pass: lock $MLT/pipeline/lock -> snapshot (squeue -r, sacct) -> RESOLVE lines of results/STOP_*.md ->
           reconcile intents -> adopt manual jobs -> stage status / unit retries -> decisions and STOPs -> git commit +
           push of analysis/mlts/results only -> submit ready work within the slot policy -> results/PIPELINE_STATE.json,
           results/PIPELINE_LOG.md, $MLT/PIPELINE_ATTENTION; exit 0
  dry      the same pass read-only (squeue / sacct / git reads only): prints statuses, decisions and the sbatch lines it
           would run; never sbatch, never git writes, never state / log / DECISIONS / STOP writes
  status   one line per stage from results/PIPELINE_STATE.json + the live squeue count per stage
  lists    (CPU job, MLTENV=reactot) $MLT/pipeline/lists/{f<k>_test,dip<d>,a3}.csv + lists.json from mlt_common.targets()
  guard <file ...>   (first segment of every launcher job, any env) exit 0 when every file is tracked and equal to
           HEAD, else exit 4: a job never runs code that was changed after it was submitted (it is resubmitted once the
           code is committed; exit 4 never uses a retry)

Policy (user 2026-11-04, results/DECISIONS.md): take the recommended option at every branch and at spec STOPs, record it
in DECISIONS.md, continue; a STOP without a recommended option = HALT (results/STOP_<stage>.md, cleared by a first line
`RESOLVE: retry | accept | skip | param [<stage>.]<k>=<v>`). The launcher never deletes data and never cancels jobs.
Slots (this cluster schedules first-come-first-served: PriorityType=priority/basic, so --nice is not used):
  queue: every task of the user (launchers, sandbox, manual work counted) + the successor launcher <= 20 - 1 spare;
  eligible pending (not Dependency / BeginTime / held, launchers excluded) <= free running slots (10 - running) + 2 for
  critical work, <= free running slots for filler (filler never waits in the queue ahead of critical work);
  CPUs <= 640, GPUs <= 12 (pending counted); no submissions without a fresh $MLT/QUOTA_FREE_GB (DECISIONS 0-8).

Job contract (every stage; placeholders = code not written yet, see PIPELINE.md 'contracts'):
  python <script> <sub> [args] [<i|@I> <n>] through jobs/mlt.slurm from analysis/mlts (the launcher prepends the
  `pipeline.py guard <files> :::` segment); exit 0 done, 2 incomplete / not ready (resubmit the same command), 3 STOP;
  units atomic + skipped when present; a task stops claiming when 2 h of the 48 h wall remain (MLT_JOB_T0); report
  JSON: complete (bool), n_todo, stop (null | text), est {units_total, unit_wall_h, workers}, + the metrics the stage
  decision reads.
A script counts as ready only when it and every analysis/mlts module it imports (transitively, + mlt_common.py) are
tracked by git and equal to HEAD, and the sub-command token (add_parser("sub") / == "sub" / in (.., "sub", ..) /
"sub": ( / == ["sub"]) is present: code still being written is never run ('waiting: code ...'), and starts on its own
once the lead commits it. The same file set is checked again at job start (guard).
Every job: -J mltp_<stage> --comment=mltp:<stage>:<unit spec>:<intent id>; manual jobs are adopted by the names in
the registry (adopt) or by a comment mltp:<stage>:<unit>:manual."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)

CODE_VERSION = "mltp-2"
LAUNCH = "mltp_launch"

# ---------------------------------------------------------------- policy constants
SUBMIT_CAP = 20                 # MaxSubmit: every task (array element) of the user, running + pending
SPARE = 1                       # one queue place always left for the lead (a manual job / a new sandbox)
MAX_RUN = 10                    # MaxJobs (running tasks; SLURM enforces it)
PEND_BUF = 2                    # eligible pending tasks beyond the free running slots, critical work only
CPU_CAP, GPU_CAP = 640, 12
MAX_TASKS_PER_PASS = SUBMIT_CAP
INELIGIBLE = ("Dependency", "BeginTime", "JobHeld", "DependencyNeverSatisfied")   # pending reasons that do not compete
QUOTA_MIN_GB, QUOTA_STALE_H = 30.0, 24.0
INTENT_VOID_S = 120
LOST_AFTER_S, LOST_ALONE_S = 3600, 12 * 3600
LOCK_STALE_S = 720              # a pass lasts <= 600 s (timeout in jobs/pipeline.slurm)
RETRY = dict(cont=4, infra=1, oom=1, noprog=2, code=6, sbatch=3)
EXIT_GUARD = 4                  # `pipeline.py guard`: code changed after submission
WALL_TARGET_H = 20.0
GEOM_WALL_H = 2.0               # ts_geom arrays (FT-2 / X / recalc-5 re-measure): target wall per task, and the least wall
GEOM_UNIT_MIN_S = 180.0         # per reaction assumed (AIMNet2 Z wall 4.3 s; a fine-tuned model takes ~60-300 s) -> n =
                                # ceil(n_rxn x max(Z wall, 180 s) / 14 workers / 2 h) <= 4 (lead 11-05: f0_test 1,874 ->
                                # 4 tasks per start = 8 GPUs for both starts; 120 s gave 3)
G_BUDGET_DEFAULT = 8000.0      # gen_dft.BUDGET_CORE_H (spec 0-5 5,000 -> user decisions D0-n3 5,500, D0-plus 8,000)
G_BUDGET = G_BUDGET_DEFAULT     # kept for older callers; the live value is g_budget(P) (read from gen_dft.py)
D0_N_NEAR, D0_N_G2_TRAIN, D0_N_HOLDOUT = 2, 200, 100   # user decisions D0-n3 / D0-plus / D0-gauss (DECISIONS.md)
G4_RESERVE_CORE_H = 150.0       # = gen_d0.G4_RESERVE_CORE_H (G-4 ~100 D1 + ~50 D0, spec 3 / D0-plus)
NUR_UPLOAD_DEFAULT = "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"   # = inv_nurion.UPLOAD (the user's upload dir)
NUR_EXTRA = ("d1_extra_02_jobs_unfinished", "d1_extra_03_queue_stale_pilot")   # = inv_nurion.EXTRA_DATA (job trees)
NUR_SETTLE_S = 900              # an upload modified within 15 min counts as still uploading
NUR_PARTIAL_RX = re.compile(r"^- \*\*N partial ok\b", re.M | re.I)   # the lead's DECISIONS line: FT may start early
D0E_MODELS = ("FT_all_s0_al", "FT_all_s1_al", "FT_all_s2_al")        # = ts_geom.D0E_MODELS (D0 supplementary eval)
HISTORY_SINCE = "2026-11-04T00:00:00"                   # first sacct window for adopting the lead's manual jobs
HISTORY_MARGIN_S = 900
PART = dict(C="cpu1,cpu2", C2="cpu2", GA="gpu1,gpu2,gpu3,gpu4,gpu5,gpu6", G24="gpu1,gpu3,gpu4,gpu5", G48="gpu3,gpu4,gpu5")
SQ_FMT = "%i|%j|%T|%C|%b|%P|%r|%k"
SA_FMT = "JobID,JobName,State,ExitCode,Elapsed,Start,End,Submit,Comment"
TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
GIT_ID = ("-c", "user.name=yeseo1ee", "-c", "user.email=yeseolele@gmail.com")
LIVE_STATES = ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING", "RESIZING", "STAGE_OUT")
TRANSIENT_RX = re.compile(r"Socket timed out|Unable to contact slurm controller|Transient|try again|Connection refused|"
                          r"Communication connection failure|Zero Bytes|timed out|TimeoutExpired|Resource temporarily "
                          r"unavailable|slurm_persist|Error connecting|Slurm backup controller|in standby mode",
                          re.I)
MAXSUBMIT_RX = re.compile(r"MaxSubmit|AssocMaxSubmit|QOSMaxSubmit")
DEPENDENCY_RX = re.compile(r"[Dd]ependency")

Z_CANDS = ("MACE-OMOL-0", "MACE-MH-1", "AIMNet2", "MACE-OFF24", "xtb")
Z2FT = {"MACE-MH-1": "MACE-MH-1-omol"}                 # zero_shot name -> finetune.py candidate name
FT2Z = {v: k for k, v in Z2FT.items()}
FT_ALIAS = {"omol": "MACE-OMOL-0", "aim": "AIMNet2", "aimnet2": "AIMNet2", "mh1": "MACE-MH-1-omol"}
SPEED_DEFAULT_TAG = {"MACE-OMOL-0": "MACE-OMOL-0|float64|", "MACE-MH-1": "MACE-MH-1|float64|", "AIMNet2": "AIMNet2||",
                     "MACE-OFF24": "MACE-OFF24|float64|"}               # = zero_shot.SPEED_DEFAULT_TAG
STARTS = ("label", "g2")
FT2_A = dict(dd_median_A=0.010, dd_p90_A=0.030, rmsd_median_A=0.02, same_ts_frac=0.90)   # spec 5 FT-2 (a)
NIMAG1_MIN = 0.95
# FT2-p rules fixed before any fold-0 FT-2 number was seen (user 11-05 requests (2) and (3); decide_ft2p docstring):
FT2_DD_CUT_A = 0.030            # per-reaction |dd| cut of the failure classes (= FT2_A dd_p90_A; = ts_geom.DD_CUT_A)
FAIL_CLASSES = ("not_converged", "n_imag", "other_saddle")   # (i) / (ii) / (iii); = ts_geom.FAIL_CLASSES
NOT_ATTEMPTED = ("no_start", "no_label", "not_covered")
RECALC_K = 5                    # re-measure of the failed reactions: ts_geom --recalc 5 (analytic:5 = G2 / G3L Recalc_Hess 5;
RECALC_SUFFIX = f"_recalc{RECALC_K}"   # never less frequent than D1 config_run.yaml Recalc_Hess 30) -> <start>_recalc5/
RECALC_WALL_X = 2.0             # wall per re-measured reaction ~ 2 x the Z wall (more Hessians, harder reactions)
EZ_SAME_DIFF, EZ_DD_DIFF_A = 0.05, 0.002    # E/Z rule: 'with' vs 'without sibling' differ by > 0.05 same-TS fraction OR
                                            # > 0.002 A median |dd| (label start) -> gate on 'without' only
EZ_LIST_TMPL = "f{k}_test_ez"               # fold k (FT2-r2: k = 0..4, made by ez_groups.py run --folds 0,1,2,3,4)
GATE_FOLD = 4                               # FT2-p gate fold: user 11-05 15:5x "바꿔줘" (fold 0 -> 4, before any FT-2
                                            # number was seen): FTpilot_<c>_4 trained first, FT-p (fold 0) still trains
GATE_LIST = f"f{GATE_FOLD}_test"            # its test list (2,195 reactions)
EZ_LIST_NAME = EZ_LIST_TMPL.format(k=GATE_FOLD)   # its E/Z sibling list (f4_test_ez: 0 'with sibling' of 2,195)
FT2_SPEC_RX = rf"ft2_spec_f{GATE_FOLD}_(?P<s>label|g2)"   # the lead's manual FT-2 arrays of the gate fold (1063318 /
FT2_SPEC_CAND = "AIMNet2"                   # 1063319, --array=0-1, `ts_geom.py run FTpilot_AIMNet2_4 ... @I 2`): adopted
F0_REPORT = "FT2_f0_report.json"            # FT2-f0: the fold-0 FT-2 numbers split by f0_test_ez (report only)
TIE_SAME, TIE_DD = 0.025, 0.002
# FT-abl (user 11-05 17:0x (2) "비교 결과로 자동 선택"): the lead's manual fold-4 force-weight ablations (ft_abl_<tag>
# `finetune.py train FTabl_AIMNet2_4_<tag> AIMNet2 --fw <x> --patience 50 ::: ft1`, then afterok ft2_abl_<tag> `ts_geom.py
# run FTabl_AIMNet2_4_<tag> f4_test label ... @I 2`, first pass only) against the baseline FTpilot_AIMNet2_4 (fw 0.2,
# patience 25) pick the `finetune.py train` arguments of FT-r2 / FT-fin; rule fixed before any ablation number (decide_ftabl)
ABL_CAND = FT2_SPEC_CAND                    # AIMNet2: the ablated candidate (the arguments apply only to that base)
ABL_BASE_RUN = f"FTpilot_{ABL_CAND}_{GATE_FOLD}"   # = gate_run(ABL_CAND), first-pass records $MLT/ft2/<run>/label/recs
ABL_TAGS = ("fw1", "fw5")
ABL_ARGS = {"fw1": ["--fw", "1.0", "--patience", "50"], "fw5": ["--fw", "5.0", "--patience", "50"]}
ABL_FW = {"baseline": 0.2, "fw1": 1.0, "fw5": 5.0}
ABL_PATIENCE = {"baseline": 25, "fw1": 50, "fw5": 50}
ABL_MIN_GAIN = 0.02                         # an ablation must beat the baseline same-TS fraction by MORE than this
ABL_DD_TIE_A = 0.002                        # ablations within ABL_MIN_GAIN of each other -> lower median |dd|; within
ABL_START = "label"                         # this too -> the smaller fw (closer to the default)
GEOM_PARTIAL = 0.995           # coverage accepted after 2 rounds without progress (records missing = pipeline errors)
Z_TOL = 0.01                    # Z: up to 1 % of the list without a record counts as finished
A3_RULE = dict(stop_below=0.70, limit_below=0.85, app_core_h=1.0)
A3_BUDGET = {"A": (0.6, 3.0), "Ap": (0.6, 3.0), "App": (1.0, 3.0)}    # arm: (core-h, GPU-min) per reaction, median
N_D1 = 9266
A_WORKERS, A_CORES = 6, 2                               # mlts_autode A2-workers (before A-3 measures its own)
DISK_GB_PER_RXN_DEFAULT = 0.002                         # A-3 before its own measurement (A2 test: 0.22 MB / reaction)
H_DISK_GB_PER_RXN_DEFAULT = 0.02
APP_DISK_GB_PER_RXN_DEFAULT = 0.01
DISK_GB_DEFAULT = 0.2                                   # per task, for the free-disk estimate between quota readings
DEFAULT_OPTIONS = ["`retry` — 시도 횟수를 0으로 되돌리고 다시 낸다(원인을 고친 뒤) / reset attempts and resubmit",
                   "`accept` — 이 단계를 끝난 것으로 보고 다음 단계로 간다 / treat the stage as done",
                   "`skip` — 이 단계를 건너뛴다(뒤 단계가 이 단계 산출물 없이 시작될 수 있다) / skip the stage",
                   "`param [<단계>.]<이름>=<값>` — 단계 매개변수를 바꾸고 다시 낸다 / set a stage parameter and retry"]


class SnapshotError(RuntimeError):
    pass


# ================================================================ small helpers
def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def parse_iso(s):
    try:
        return time.mktime(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError, OverflowError):
        return None


def _int(x, default=0):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return default


def gpu_count(tres):
    t = str(tres or "")
    if "gpu" not in t:
        return 0
    n = [int(x) for x in re.findall(r"gpu(?::[A-Za-z0-9_.-]+)?:(\d+)", t)]
    return sum(n) if n else 1


def parse_sq(line):
    p = line.rstrip("\n").split("|", 7)
    p += [""] * (8 - len(p))
    jid, name, state, cpus, tres, part, reason, comment = (x.strip() for x in p)
    return dict(id=jid, name=name, state=state, cpus=_int(cpus), gpus=gpu_count(tres), part=part, reason=reason,
                comment="" if comment in ("(null)", "") else comment)


def parse_sa(line):
    p = line.rstrip("\n").split("|", 8)
    p += [""] * (9 - len(p))
    jid, name, state, exitc, elapsed, start, end, submit, comment = (x.strip() for x in p)
    rc, _, sig = exitc.partition(":")
    return dict(id=jid, name=name, state=state, exit=_int(rc, None), sig=_int(sig), elapsed=elapsed, start=start,
                end=end, submit=submit, comment=comment)


def expand_ids(jid):
    """'123_[2-3,5%2]' -> ['123_2', '123_3', '123_5']; anything else -> [jid]."""
    m = re.fullmatch(r"(\d+)_\[([^\]]+)\]", str(jid))
    if not m:
        return [str(jid)]
    base, spec, out = m.group(1), m.group(2).split("%")[0], []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")[:2]
            out += [f"{base}_{i}" for i in range(int(a), int(b) + 1)]
        elif part.strip():
            out.append(f"{base}_{int(part)}")
    return out


def parse_comment(c):
    """'mltp:<stage>:<unit spec>:<intent>' -> (stage, spec, intent) or None."""
    if not c or not str(c).startswith("mltp:"):
        return None
    p = str(c).split(":")
    if len(p) < 4:
        return None
    return p[1], ":".join(p[2:-1]), p[-1]


def spec_keys(spec, line_id=None):
    """unit spec of a comment -> unit keys. 'G1[0+1+2]' with line id '123_1' -> ['G1[1]']; 'collect' -> ['collect']."""
    m = re.fullmatch(r"(.+)\[([\d+]+)\]", spec or "")
    if not m:
        return [spec] if spec else []
    akey, idx = m.group(1), [int(x) for x in m.group(2).split("+") if x]
    if line_id and "_" in str(line_id):
        i = _int(str(line_id).split("_", 1)[1], None)
        if i is not None:
            return [f"{akey}[{i}]"]
    return [f"{akey}[{i}]" for i in idx]


def classify(end):
    s = str(end.get("state") or "").split()[0] if end.get("state") else ""
    rc = end.get("exit")
    if s == "COMPLETED":
        return "ok" if rc in (0, None) else {2: "cont", 3: "stop", EXIT_GUARD: "code"}.get(rc, "infra")
    if s == "FAILED":
        return {2: "cont", 3: "stop", EXIT_GUARD: "code"}.get(rc, "infra")
    if s == "TIMEOUT":
        return "cont"
    if s == "OUT_OF_MEMORY":
        return "oom"
    if s == "CANCELLED":
        return "cancelled"
    return "infra"                     # NODE_FAIL BOOT_FAIL PREEMPTED DEADLINE LOST ...


def count_dir(p):
    try:
        with os.scandir(p) as it:
            return sum(1 for _ in it)
    except OSError:
        return 0


def count_lines(p):
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def n_tasks(units_total, unit_wall_h, workers, nmax, wall=WALL_TARGET_H):
    """n = clamp(ceil(units x unit wall / workers / 20 h), 1, n_max)."""
    try:
        x = float(units_total) * float(unit_wall_h) / max(1.0, float(workers)) / wall
    except (TypeError, ValueError):
        return 1
    return int(max(1, min(int(nmax), math.ceil(x)))) if x > 0 else 1


def pct(v, q):
    """numpy-default (linear) percentile of a list."""
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def child_env(base=None):
    """environment for sbatch from inside a job: drop the launcher's SLURM_* / SBATCH_* and MLT_JOB_T0 (env.sh would
    otherwise inherit the launcher's start time)."""
    e = dict(os.environ if base is None else base)
    for k in list(e):
        if k.startswith(("SLURM_", "SBATCH_", "SRUN_", "SALLOC_")) or k in ("MLT_JOB_T0", "MLTENV",
                                                                              "PIPELINE_FAKE_SLURM"):
            e.pop(k, None)
    return e


def token_re(sub):
    s = re.escape(sub)
    q = r"""["']"""
    return re.compile(rf"(add_parser\(\s*{q}{s}{q}|==\s*{q}{s}{q}|\bin\s*\([^)]*{q}{s}{q}[^)]*\)|{q}{s}{q}\s*:\s*\(|"
                      rf"==\s*\[\s*{q}{s}{q}\s*\])")


def eligible(line):
    """a PENDING task that competes for a running slot now (not waiting on a dependency / begin time / hold)."""
    r = str(line.get("reason") or "")
    return line.get("state") == "PENDING" and not any(r.startswith(x) for x in INELIGIBLE)


def sbatch_error_kind(rc, err):
    """'maxsubmit' (the submit cap), 'transient' (controller / timeout: the job may or may not exist), 'dependency'
    (a dependency job left the controller between the snapshot and sbatch: nothing created, recomputed next pass),
    'fixed' (anything else: counted, HALT after RETRY['sbatch'] passes in a row)."""
    e = str(err or "")
    if MAXSUBMIT_RX.search(e):
        return "maxsubmit"
    if rc == 255 or TRANSIENT_RX.search(e):
        return "transient"
    if DEPENDENCY_RX.search(e):
        return "dependency"
    return "fixed"


IMPORT_RX = re.compile(r"^[ \t]*(?:from[ \t]+([A-Za-z_]\w*)[ \t]+import|import[ \t]+([A-Za-z_][\w \t,.]*))", re.M)


def local_closure(code_dir, scripts):
    """the scripts + every analysis/mlts module they import (transitively, any indentation) + mlt_common.py, as
    file names relative to code_dir (sorted). A module counts when <code_dir>/<name>.py exists."""
    code_dir = Path(code_dir)
    seen, todo = set(), [s for s in scripts if s]
    while todo:
        s = todo.pop()
        if s in seen:
            continue
        p = code_dir / s
        if not p.is_file():
            seen.add(s)
            continue
        seen.add(s)
        try:
            txt = p.read_text()
        except OSError:
            continue
        for a, b in IMPORT_RX.findall(txt):
            names = [a] if a else [x.strip().split()[0].split(".")[0] for x in b.split(",") if x.strip()]
            for n in names:
                if n and (code_dir / f"{n}.py").is_file() and f"{n}.py" not in seen:
                    todo.append(f"{n}.py")
    if (code_dir / "mlt_common.py").is_file():
        seen.add("mlt_common.py")
    return sorted(seen)


def cmd_scripts(cmd):
    """the script of every ' ::: ' / ' ;;; ' segment of a jobs/mlt.slurm command."""
    out, first = [], True
    for t in cmd:
        if t in (":::", ";;;"):
            first = True
            continue
        if first:
            out.append(t)
            first = False
    return [s for s in out if s.endswith(".py")]


# ================================================================ SLURM interfaces
class Slurm:
    """real SLURM client commands (compute node)."""
    fake = False

    def __init__(self, user=None):
        self.user = user or os.environ.get("USER") or os.environ.get("LOGNAME") or "yeseo1ee"

    @staticmethod
    def _run(argv, timeout=90, cwd=None, env=None):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
            return r.returncode, r.stdout, r.stderr
        except (OSError, subprocess.TimeoutExpired) as e:
            return 255, "", f"{type(e).__name__}: {e}"

    def squeue(self):
        rc, out, err = self._run(["squeue", "-r", "-u", self.user, "-h", "-o", SQ_FMT])
        if rc != 0:
            raise SnapshotError(f"squeue rc {rc}: {err.strip()[:300]}")
        return [parse_sq(x) for x in out.splitlines() if x.strip()]

    def sacct(self, ids=None, names=None, since=None):
        rows = []
        chunks = [list(ids)[i:i + 100] for i in range(0, len(ids), 100)] if ids else [None]
        for ch in chunks:
            argv = ["sacct", "-X", "-n", "-P", "-o", SA_FMT]
            argv += ["-j", ",".join(ch)] if ch else ["-u", self.user]
            if names:
                argv += ["--name", ",".join(names)]
            if since:
                argv += ["-S", since]
            rc, out, err = self._run(argv)
            if rc != 0:
                return None
            rows += [parse_sa(x) for x in out.splitlines() if x.strip()]
        return rows

    def sbatch(self, argv, cwd, env):
        rc, out, err = self._run(["sbatch", "--parsable"] + list(argv), timeout=120, cwd=cwd, env=env)
        return rc, out.strip(), err.strip()


class ReadOnlySlurm:
    """dry mode: squeue / sacct pass through, sbatch refuses."""
    def __init__(self, inner):
        self.inner, self.fake = inner, getattr(inner, "fake", False)

    def squeue(self):
        return self.inner.squeue()

    def sacct(self, *a, **k):
        return self.inner.sacct(*a, **k)

    def sbatch(self, *a, **k):
        raise RuntimeError("sbatch called in dry mode")


def parse_sbatch(argv):
    """the options pipeline.py passes (both '--x=v' and '-x v' forms) + the script and its arguments."""
    o = dict(name=None, cpus=1, gpus=0, mem=None, part=None, array=None, comment="", dep=None, nice=0, begin=None,
             env=None, output=None, script=None, cmd=[])
    i, a = 0, list(argv)
    short = {"-J": "--job-name", "-p": "--partition", "-c": "--cpus-per-task", "-o": "--output", "-a": "--array"}
    while i < len(a):
        t = a[i]
        if t in ("--parsable",):
            i += 1
            continue
        if t in short:
            k, v = short[t], a[i + 1]
            i += 2
        elif t.startswith("--") and "=" in t:
            k, v = t.split("=", 1)
            i += 1
        elif t.startswith("-"):
            k, v = t, (a[i + 1] if i + 1 < len(a) else "")
            i += 2
        else:
            o["script"], o["cmd"] = t, a[i + 1:]
            break
        if k == "--job-name":
            o["name"] = v
        elif k == "--partition":
            o["part"] = v
        elif k == "--cpus-per-task":
            o["cpus"] = _int(v, 1)
        elif k == "--gres":
            o["gpus"] = gpu_count(v)
        elif k == "--mem":
            o["mem"] = v
        elif k == "--array":
            o["array"] = [int(x) for x in v.split("%")[0].split(",") if x.strip().isdigit()]
        elif k == "--comment":
            o["comment"] = v
        elif k == "--dependency":
            o["dep"] = v
        elif k == "--nice":
            o["nice"] = _int(v)
        elif k == "--begin":
            o["begin"] = v
        elif k == "--export":
            m = re.search(r"MLTENV=([A-Za-z0-9_]+)", v)
            o["env"] = m.group(1) if m else None
        elif k == "--output":
            o["output"] = v
    return o


class FakeSlurm:
    """in-memory SLURM for tests (PIPELINE_FAKE_SLURM=<json> persists it between processes)."""
    fake = True

    def __init__(self, path=None, clock=None, max_submit=20):
        self.path = Path(path) if path else None
        self.jobs = {}
        self.next_id = 1062500
        self.calls = []
        self.max_submit = max_submit
        self.clock = clock or time.time
        self.fail = None             # None | 'maxsubmit' | 'error' | 'transient' | 'transient_created' | 'crash_after' | 'crash_before'
        self.sacct_fail = False
        if self.path and self.path.is_file():
            d = json.loads(self.path.read_text())
            self.jobs, self.next_id, self.calls = d["jobs"], d["next_id"], d.get("calls", [])

    def _save(self):
        if self.path:
            M.write_json(self.path, dict(jobs=self.jobs, next_id=self.next_id, calls=self.calls))

    def add(self, jid, name, state="RUNNING", cpus=1, gpus=0, part="cpu2", comment="", reason="None", submit=None,
            start=None, exit=0, cmd=None):
        self.jobs[str(jid)] = dict(id=str(jid), name=name, state=state, cpus=cpus, gpus=gpus, part=part,
                                   comment=comment, reason=reason, submit=submit or iso(self.clock()),
                                   start=start or (iso(self.clock()) if state == "RUNNING" else "Unknown"),
                                   end="Unknown", exit=exit, elapsed="00:00:00", cmd=cmd or [])
        self._save()

    def add_line(self, line):
        d = parse_sq(line)
        self.add(d["id"], d["name"], d["state"], d["cpus"], d["gpus"], d["part"], d["comment"], d["reason"])

    def ids(self, jid):
        jid = str(jid)
        return [k for k in self.jobs if k == jid or k.split("_")[0] == jid]

    def set(self, jid, state, exit=0, end=None, start=None):
        for k in self.ids(jid):
            j = self.jobs[k]
            if start:
                j["start"] = start
            elif j["start"] == "Unknown" and state != "PENDING":
                j["start"] = iso(self.clock())
            j["state"], j["exit"] = state, exit
            if state not in LIVE_STATES:
                j["end"] = end or iso(self.clock())
        self._save()

    def squeue(self):
        return [dict(id=j["id"], name=j["name"], state=j["state"], cpus=j["cpus"], gpus=j["gpus"], part=j["part"],
                     reason=j["reason"], comment=j["comment"]) for j in self.jobs.values()
                if j["state"] in ("PENDING", "RUNNING")]

    def sacct(self, ids=None, names=None, since=None):
        if self.sacct_fail:
            return None
        rows = []
        want = set(map(str, ids)) if ids else None
        for j in self.jobs.values():
            if want is not None and j["id"] not in want and j["id"].split("_")[0] not in want:
                continue
            if names and j["name"] not in names:
                continue
            if since and j["submit"] < since[:19]:
                continue
            rows.append(dict(id=j["id"], name=j["name"], state=j["state"], exit=j["exit"], sig=0, elapsed=j["elapsed"],
                             start=j["start"], end=j["end"], submit=j["submit"], comment=j["comment"]))
        return rows

    def sbatch(self, argv, cwd=None, env=None):
        self.calls.append(list(argv))
        a = parse_sbatch(argv)
        n = len(a["array"]) if a["array"] is not None else 1
        if self.fail == "crash_before":
            raise RuntimeError("fake crash before sbatch")
        if self.fail == "error":
            self._save()
            return 1, "", "sbatch: error: Batch job submission failed: Invalid partition name specified"
        if self.fail == "transient":
            self._save()
            return 1, "", "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"
        if self.fail == "maxsubmit" or len(self.squeue()) + n > self.max_submit:
            self._save()
            return 1, "", "sbatch: error: QOSMaxSubmitJobPerUserLimit"
        jid = str(self.next_id)
        self.next_id += 1
        ids = [f"{jid}_{i}" for i in a["array"]] if a["array"] is not None else [jid]
        if a["name"] is None and str(a["script"] or "").endswith("pipeline.slurm"):
            a["name"] = LAUNCH                               # the script's own #SBATCH --job-name
        for t in ids:
            self.jobs[t] = dict(id=t, name=a["name"] or "sbatch", state="PENDING", cpus=a["cpus"], gpus=a["gpus"],
                                part=a["part"] or "", comment=a["comment"],
                                reason="Dependency" if a["dep"] else ("BeginTime" if a["begin"] else "Priority"),
                                submit=iso(self.clock()), start="Unknown", end="Unknown", exit=0, elapsed="00:00:00",
                                cmd=a["cmd"], script=a["script"], env=a["env"], mem=a["mem"], dep=a["dep"],
                                nice=a["nice"], begin=a["begin"])
        self._save()
        if self.fail == "crash_after":
            raise RuntimeError("fake crash after sbatch")
        if self.fail == "transient_created":                 # the controller created the job but the reply timed out
            return 255, "", "TimeoutExpired: sbatch did not answer in 120 s"
        return 0, jid, ""


def default_slurm():
    p = os.environ.get("PIPELINE_FAKE_SLURM")
    return FakeSlurm(p) if p else Slurm()


# ================================================================ git interfaces
class Git:
    fake = False

    def __init__(self, cwd, readonly=False):
        self.cwd, self.readonly = str(cwd), readonly

    def _g(self, *args, timeout=60):
        """git with a timeout that ends it with SIGTERM first (git removes its index.lock on SIGTERM; the SIGKILL of
        subprocess.run(timeout=) would leave .git/index.lock behind and block every later commit)."""
        try:
            p = subprocess.Popen(["git", *args], cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True)
        except OSError as e:
            return 255, f"{type(e).__name__}: {e}"
        try:
            out, _ = p.communicate(timeout=timeout)
            return p.returncode, out or ""
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                out, _ = p.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
            return 255, f"git {args[0] if args else ''} timed out after {timeout} s: {(out or '')[-200:]}"

    def tracked(self, path):
        return self._g("ls-files", "--error-unmatch", "--", str(path))[0] == 0

    def clean(self, path):
        return self._g("diff", "--quiet", "HEAD", "--", str(path))[0] == 0

    def branch(self):
        rc, out = self._g("rev-parse", "--abbrev-ref", "HEAD")
        return out.strip().splitlines()[0] if rc == 0 and out.strip() else None

    def _lock_path(self):
        rc, out = self._g("rev-parse", "--git-path", "index.lock")
        if rc != 0 or not out.strip():
            return None
        p = Path(out.strip().splitlines()[0])
        return p if p.is_absolute() else Path(self.cwd) / p

    def index_locked(self):
        p = self._lock_path()
        return True if p is None else p.exists()

    def index_lock_age(self):
        p = self._lock_path()
        try:
            return time.time() - p.stat().st_mtime if p is not None else None
        except OSError:
            return None

    def head(self):
        rc, out = self._g("rev-parse", "--short", "HEAD")
        return out.strip() if rc == 0 else None

    def add(self, paths):
        if self.readonly:
            raise RuntimeError("git add in read-only mode")
        return self._g("add", "--", *map(str, paths))

    def commit(self, paths, title, body):
        if self.readonly:
            raise RuntimeError("git commit in read-only mode")
        return self._g(*GIT_ID, "commit", "--only", "-m", title, "-m", body, "--", *map(str, paths), timeout=120)

    def push(self):
        if self.readonly:
            raise RuntimeError("git push in read-only mode")
        return self._g("push", "origin", "mlts", timeout=60)


class FakeGit:
    fake = True

    def __init__(self, tracked=None, dirty=(), branch="mlts", push_ok=True, locked=False):
        self.tracked_names = None if tracked is None else set(tracked)
        self.dirty, self.br, self.push_ok, self.locked = set(dirty), branch, push_ok, locked
        self.commits, self.pushes, self.committed, self.n = [], [], set(), 0
        self.lock_age = 0.0

    def tracked(self, path):
        n = Path(path).name
        return self.tracked_names is None or n in self.tracked_names or n in self.committed

    def clean(self, path):
        return Path(path).name not in self.dirty

    def branch(self):
        return self.br

    def index_locked(self):
        return self.locked

    def index_lock_age(self):
        return self.lock_age if self.locked else None

    def head(self):
        return f"fake{self.n:04d}"

    def add(self, paths):
        return 0, ""

    def commit(self, paths, title, body):
        self.n += 1
        self.commits.append(dict(paths=[str(p) for p in paths], title=title, body=body))
        self.committed |= {Path(p).name for p in paths}
        return 0, ""

    def push(self):
        self.pushes.append(self.push_ok)
        return (0, "") if self.push_ok else (1, "fake push failure")


# ================================================================ context
class Ctx:
    """roots + injectable SLURM / git / clock (tests point every root under $MLT/tmp/pipetest)."""

    def __init__(self, mlt=None, res=None, pdir=None, ctl=None, code=None, slurm=None, git=None, clock=None, job=None,
                 launcher=None, env=None, nur_upload=None):
        self.nur_upload = Path(nur_upload or os.environ.get("MLT_NUR_UPLOAD") or NUR_UPLOAD_DEFAULT)
        self.mlt = Path(mlt) if mlt else Path(M.MLT)
        self.res = Path(res) if res else Path(M.RES)
        self.pdir = Path(pdir) if pdir else self.mlt / "pipeline"
        self.ctl = Path(ctl) if ctl else self.mlt
        self.code = Path(code) if code else HERE
        self.slurm = slurm if slurm is not None else default_slurm()
        self.git = git if git is not None else Git(self.code)
        self.clock = clock or time.time
        self.job = str(job or os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}")
        self.launcher = (os.environ.get("SLURM_JOB_NAME") == LAUNCH) if launcher is None else bool(launcher)
        self.env = env

    @property
    def state_path(self):
        return self.res / "PIPELINE_STATE.json"

    @property
    def log_path(self):
        return self.res / "PIPELINE_LOG.md"

    @property
    def dec_path(self):
        return self.res / "DECISIONS.md"

    @property
    def lists(self):
        return self.pdir / "lists"

    def list_path(self, name):
        return self.lists / f"{name}.csv"


# ================================================================ registry building blocks
class Unit:
    __slots__ = ("key", "akey", "idx", "cmd", "env", "part", "c", "mem", "gpu", "deps", "done", "sink", "disk_gb")

    def __init__(self, key, cmd, env="reactot", part="C", c=4, mem=16, gpu=0, deps=(), done=None, sink=False,
                 akey=None, idx=None, disk_gb=None):
        self.key, self.cmd, self.env, self.part, self.c, self.mem, self.gpu = key, [str(x) for x in cmd], env, part, c, mem, gpu
        self.deps, self.done, self.sink, self.akey, self.idx = list(deps), done, sink, akey, idx
        self.disk_gb = float(DISK_GB_DEFAULT if disk_gb is None else disk_gb)     # GB this task may write (estimate)


def one(key, cmd, **kw):
    return [Unit(key, cmd, **kw)]


def arr(akey, cmd, n, **kw):
    """array units akey[0..n-1]; cmd carries '@I' (mlt_common.array_task) and n."""
    return [Unit(f"{akey}[{i}]", cmd, akey=akey, idx=i, **kw) for i in range(int(n))]


def keys(us):
    return [u.key for u in us]


class Stage:
    """one registry entry. after: stages that must be done (or skipped) first; code: [(script, sub[, regex])];
    done(P): the outputs are complete; decide(P, S, st) -> {key, line | halt, numbers, options}; guard(P) -> HALT text;
    gate(P) -> ('pass' | 'fail' | 'wait', why, numbers) (no jobs); active(P) -> False marks the stage skipped (an
    optional branch, judged once its prerequisites are done); partial(P) -> (ok, text): outputs accepted with a few
    records missing after the no-progress rounds; filler: rank only (never waits in the queue ahead of critical work,
    its HALT is local: only its dependants stop); rearm: finishes an input set and waits for more (H-early)."""

    def __init__(self, name, phase, crit, units=None, *, after=(), code=(), need=(), done=None, guard=None, gate=None,
                 progress=None, adopt=(), decide=None, on_start=None, on_exit3="halt", filler=False, results=(),
                 commit_gate=False, placeholder=False, summary="", rearm=False, active=None, partial=None,
                 halt_scope=None):
        self.name, self.phase, self.crit, self.units_fn, self.rearm = name, phase, crit, units, rearm
        self.after, self.need, self.done, self.guard, self.gate = list(after), list(need), done, guard, gate
        self.code = [tuple(c) + (None,) * (3 - len(c)) for c in code]
        self.progress, self.adopt, self.decide, self.on_start, self.on_exit3 = progress, list(adopt), decide, on_start, on_exit3
        self.filler, self.results, self.commit_gate, self.placeholder, self.summary = filler, list(results), commit_gate, placeholder, summary
        self.active, self.partial = active, partial
        self.halt_scope = halt_scope or ("local" if filler else "global")

    def units(self, P, st):
        return self.units_fn(P, self, st) if self.units_fn else []


# ---------------------------------------------------------------- shared predicates / estimates
def fixed_n(P, st, key, compute):
    """task count fixed in the state at its first use (per epoch), so slices stay stable."""
    nm = st.setdefault("params", {}).setdefault("n", {})
    k = f"{key}@{st.get('epoch', 0)}"
    if k not in nm:
        try:
            nm[k] = max(1, int(compute()))
        except Exception:                                                   # noqa: BLE001
            nm[k] = 1
    return int(nm[k])


def adopted_n(st, akey):
    """1 + the largest index i of the units akey[i] that hold an adopted (manual) job of this round, else 0: an array
    the lead submitted by hand fixes the task count of that array key (its `@I n` slicing)."""
    n, ep = 0, int(st.get("epoch") or 0)
    rx = re.compile(re.escape(akey) + r"\[(\d+)\]")
    for k, us in (st.get("units") or {}).items():
        m = rx.fullmatch(k)
        if m and any(j.get("adopted") and int(j.get("epoch") or 0) == ep for j in us.get("jobs", [])):
            n = max(n, int(m.group(1)) + 1)
    return n


def rep_complete(name):
    return lambda P: (P.rj(name) or {}).get("complete") is True


BUDGET_RX = re.compile(r"^BUDGET_CORE_H\s*=\s*([0-9][0-9_.]*)", re.M)


def g_budget(P=None):
    """the G budget cap = gen_dft.BUDGET_CORE_H, read from the file (pipeline.py imports stdlib + mlt_common only);
    G_BUDGET_DEFAULT when the file or the line is missing (test stubs)."""
    code = Path(P.c.code if P is not None else HERE) / "gen_dft.py"
    try:
        m = BUDGET_RX.search(code.read_text())
        return float(m.group(1).replace("_", "")) if m else G_BUDGET_DEFAULT
    except (OSError, ValueError):
        return G_BUDGET_DEFAULT


def lists_info(P):
    return P.rjp(P.c.lists / "lists.json") or {}


def n_list(P, name):
    return int(((lists_info(P).get("files") or {}).get(name) or {}).get("n") or 0)


def lists_done(P):
    info = lists_info(P)
    want = [f"f{k}_test" for k in range(5)] + [f"dip{d}" for d in range(5)] + ["a3"]
    f = info.get("files") or {}
    return all(w in f and Path(f[w]["path"]).is_file() for w in want)


def geom_cov(P, out_dir):
    """records summarised in <out_dir>/summary.json (every status, incl. no_start / not_covered)."""
    S = P.rjp(Path(out_dir) / "summary.json") or {}
    sc = (S.get("all") or {}).get("status_counts") or {}
    try:
        return sum(int(v) for v in sc.values())
    except (TypeError, ValueError):
        return 0


def geom_done(P, out_dir, nl):
    return nl > 0 and geom_cov(P, out_dir) >= nl


def geom_partial(dirs_fn):
    """Stage.partial for ts_geom stages: after the no-progress rounds, accept when every (out_dir, list size) has
    >= 99.5 % of its reactions recorded (ts_geom writes no record for a reaction that raises a pipeline error; such
    reactions count as failures in the pooled FT-2 numbers, decide_ft2r2)."""
    def f(P):
        rows = [(str(d), nl, geom_cov(P, d)) for d, nl in dirs_fn(P)]
        bad = [r for r in rows if r[1] <= 0 or r[2] < math.ceil(GEOM_PARTIAL * r[1])]
        miss = sum(max(0, nl - c) for _, nl, c in rows)
        if not rows or bad:
            return False, f"{len(bad)} output folders below {GEOM_PARTIAL:.1%} coverage"
        return True, (f"ts_geom: {miss} reactions without a record after 2 rounds without progress (pipeline errors) — "
                      f"coverage >= {GEOM_PARTIAL:.1%} in every folder, accepted; they count as failures (not same TS) in "
                      f"the pooled FT-2 numbers")
    return f


def geom_all(P, out_dir):
    return (P.rjp(Path(out_dir) / "summary.json") or {}).get("all") or {}


def data_ok(P, run, newer_than=("G_report.json", "G_report_d0.json")):
    p = P.c.res / "ft" / f"{run}_data.json"
    J = P.rjp(p)
    if not J or not J.get("inputs_complete") or not (J.get("leakage_C2") or {}).get("pass_"):
        return False
    t = P.mt(p)
    for n in newer_than:                          # a report that does not exist (stage skipped by RESOLVE) is ignored
        tn = P.mt(P.c.res / n)
        if tn is not None and t < tn:
            return False
    return True


def train_done(P, run):
    T = P.rjp(P.c.res / "ft" / f"{run}_train.json") or {}
    if T.get("status") != "done" or not (P.c.mlt / "models" / f"{run}.json").is_file():
        return False
    D = P.rjp(P.c.res / "ft" / f"{run}_data.json")
    if D and T.get("data_fingerprint") and D.get("fingerprint") and T["data_fingerprint"] != D["fingerprint"]:
        return False
    tf, tt = P.mt(P.c.res / "ft" / f"{run}_ft1.json"), P.mt(P.c.res / "ft" / f"{run}_train.json")
    return tf is not None and tt is not None and tf >= tt


def is_aimnet(cand):
    return str(cand).lower().startswith("aimnet")


def train_unit(key, run, cand, al=False, deps=(), args=()):
    """`finetune.py train <run> <cand> [--with-al] [args] ::: finetune.py ft1 <run> [--with-al]`; args (the FT-abl
    choice, ft_args) go to train only."""
    x = ["--with-al"] if al else []
    cmd = ["finetune.py", "train", run, cand] + x + [str(a) for a in args] + [":::", "finetune.py", "ft1", run] + x
    rd = run + ("_al" if al else "")
    return one(key, cmd, env="mlts", part="GA" if is_aimnet(cand) else "G24", c=8, mem=64 if is_aimnet(cand) else 96,
               gpu=1, deps=deps, done=lambda P, r=rd: train_done(P, r), disk_gb=3.0)


def pilots(P):
    """Z pilots minus the candidates put on hold ($MLT/pipeline/pilots_held.json {"held": [...]}; user 11-05:
    MACE-OMOL-0 'setting failure, on hold' — NaN after 1 epoch at lr 1e-3, not a PES-quality verdict)."""
    held = set((rj_abs(P.c.mlt / "pipeline" / "pilots_held.json") or {}).get("held") or [])
    return [c for c in (P.state.get("params", {}).get("pilots") or []) if c not in held]


def rj_abs(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def base_cand(P):
    return P.state.get("params", {}).get("base")


def r1_run(P, k):
    """first-round model of fold k: FT_0 = the base pilot; FT_1-4 = FTpilot_<base>_<k> when that speculative run
    (decision FT-spec: both pilots' folds 1-4 trained during the pilot) is trained, else FT_<k>."""
    if k == 0:
        return P.state.get("params", {}).get("base_run")
    b = P.state.get("params", {}).get("base")
    if b and train_done(P, f"FTpilot_{b}_{k}"):
        return f"FTpilot_{b}_{k}"
    return f"FT_{k}"


def spec_live(P, cand, k):
    """a manual speculative run of fold k (job name ft_spec_<cand>_<k>, decision FT-spec) is queued or running."""
    name = f"ft_spec_{cand}_{k}"
    return any(j.get("name") == name for j in (getattr(P, "sq", None) or []))


def gate_run(c):
    """the FT2-p gate model of pilot c = FTpilot_<c>_<GATE_FOLD> (fold 4, user 11-05 15:5x). FTpilot_<c>_0 keeps its
    fold-0 meaning: FT-p trains it, params base_run = FT_0 = r1_run(0), G-4 --models, C-1 (tests.py)."""
    return f"FTpilot_{c}_{GATE_FOLD}"


def gate_trained(P):
    """every pilot's gate-fold model trained (FT2-p need; FT2-p does not wait for FT-p / fold 0)."""
    p = pilots(P)
    return bool(p) and all(train_done(P, gate_run(c)) for c in p)


def geom_spec_live(P, k, s):
    """a manual FT-2 array of fold k / start s (job name ft2_spec_f<k>_<s>: the lead's 1063318 / 1063319) is queued or
    running. FT2-p adopts it (FT2_SPEC_RX); while it is in squeue the start gets no unit at all (neither its first pass
    nor its recalc-5 re-measure is submitted beside it)."""
    name = f"ft2_spec_f{k}_{s}"
    return any(j.get("name") == name for j in (getattr(P, "sq", None) or []))


def z_wall_h(P, cand):
    Z = P.rj("Z_report.json") or {}
    g = ((Z.get("candidates") or {}).get(FT2Z.get(cand, cand)) or {}).get("geometry") or {}
    w = g.get("wall_median_s")
    return (float(w) if w else 120.0) / 3600.0


def geom_unit_h(P, cand):
    """wall per reaction for the ts_geom task count: the Z wall of the candidate, at least GEOM_UNIT_MIN_S."""
    return max(z_wall_h(P, cand), GEOM_UNIT_MIN_S / 3600.0)


def geom_units(P, st, run, cand, list_name, out_root, akey_prefix="", starts=STARTS):
    """ts_geom array units for the starts of one model (skips a start whose summary covers the list); tasks n =
    n_tasks(n_rxn, max(Z wall, GEOM_UNIT_MIN_S = 180 s), 14 workers, nmax 4, wall GEOM_WALL_H = 2 h) (lead 11-05:
    f0_test 1,874 -> 4 per start), fixed at first use; an adopted manual array of this key fixes n to its size instead
    (adopted_n: ft2_spec_f4_<start> has n = 2), so a retried task keeps the manual `@I n` slicing."""
    nl = n_list(P, list_name)
    out = []
    for s in starts:
        od = out_root / run / s
        if geom_done(P, od, nl):
            continue
        ak = f"{akey_prefix}{run}/{s}"
        n = fixed_n(P, st, ak, lambda: adopted_n(st, ak) or n_tasks(nl, geom_unit_h(P, cand), 14, 4,
                                                                    wall=GEOM_WALL_H))
        out += arr(ak, ["ts_geom.py", "run", run, str(P.c.list_path(list_name)), s, str(od), "@I", str(n)], n,
                   env="mlts", part="GA", c=16, mem=64, gpu=1, disk_gb=0.3)
    return out


def geom_progress(P, runs, root):
    return sum(count_dir(root / r / s / "recs") for r in runs for s in STARTS)


def geom_dirs(runs_lists, root):
    """[(run, list name)] -> [(out_dir, list size)] for both starts (Stage.partial input)."""
    def f(P):
        return [(root(P) / r / s, n_list(P, ln)) for r, ln in runs_lists(P) if r for s in STARTS]
    return f


def rank_key_cmp(a, b):
    """zero_shot.rank comparator: same-TS higher; within 0.025 -> |dd| lower; within 0.002 A -> faster."""
    sa, sb = a.get("same") or 0.0, b.get("same") or 0.0
    if abs(sa - sb) > TIE_SAME:
        return -1 if sa > sb else 1
    da, db = a.get("dd") if a.get("dd") is not None else 9.9, b.get("dd") if b.get("dd") is not None else 9.9
    if abs(da - db) > TIE_DD:
        return -1 if da < db else 1
    ta, tb = a.get("ef") if a.get("ef") is not None else 1e9, b.get("ef") if b.get("ef") is not None else 1e9
    return -1 if ta < tb else (1 if ta > tb else 0)


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# ================================================================ stage functions: G
GCMD = ["gen_dft.py"]


def g_run_units(akey, cmd, n, env="reactot"):
    return arr(akey, cmd + ["@I", str(n)], n, env=env, part="C2", c=64, mem=240, disk_gb=2.0)


def units_gd1(P, S, st):
    ep = st.get("epoch", 0)
    if ep == 0:                                       # the lead's manual layout (DECISIONS G-0 / G-helper), adopted
        u = g_run_units("G1", GCMD + ["run", "G1"], 3)
        u += one("G1rev", GCMD + ["run", "G1", "0", "1", "--rev"], env="reactot", part="C2", c=64, mem=240)
        u += g_run_units("G3", GCMD + ["run", "G3"], 1)
        u += g_run_units("G2", GCMD + ["run_g2"], 3, env="d1")
    else:
        R = P.rj("G_report.json")
        if R is None:
            return one("collect", GCMD + ["collect"], env="reactot", part="C", c=4, mem=32, sink=True)
        u = []
        G0 = P.rj("G0_estimate.json") or {}
        by = ((G0.get("predicted_core_h") or {}).get("by_group") or {})
        for g in ("G1", "G3", "G2"):
            gr = (R.get("groups") or {}).get(g) or {}
            m = int(gr.get("n_missing") or 0)
            if m <= 0 or g == "G2":                   # G-2: stopped by the user at 317 / 400 (G-D1-trunc), never again
                continue
            planned = max(1, int(gr.get("n_planned") or 1))
            n = fixed_n(P, st, g, lambda: n_tasks(m, float(by.get(g) or 0) / planned, 64, 6))
            u += g_run_units(g, GCMD + (["run_g2"] if g == "G2" else ["run", g]), n, env="d1" if g == "G2" else "reactot")
    return u + one("collect", GCMD + ["collect"], env="reactot", part="C", c=4, mem=32, deps=keys(u), sink=True)


G_D1_G2_USER_STOP = 317     # user decision 'Nurion' (DECISIONS 11-04 20:4x): D1 G-2 (1061710) cancelled at 317 / 400


def gd1_g2_user_stopped(R):
    """decision 'Nurion' (DECISIONS 11-04 20:4x, decision G-D1-trunc): the user cancelled the D1 G-2 paths (job 1061710)
    at 317 of 400 units (the Nurion label-run trajectories replace the rest). True when G1 and G3 have nothing missing
    and G-2 has >= 317 units recorded; the missing G-2 units are never resubmitted."""
    g = (R or {}).get("groups") or {}
    ok13 = all(isinstance(g.get(x), dict) and _int(g[x].get("n_missing"), 1) == 0 for x in ("G1", "G3"))
    return ok13 and _int((g.get("G2") or {}).get("n_units"), 0) >= G_D1_G2_USER_STOP


def done_gd1(P):
    """G_report complete, or complete except the D1 G-2 paths the user stopped at 317 / 400 (decision G-D1-trunc)."""
    R = P.rj("G_report.json") or {}
    if R.get("stop_file") is not None:
        return False
    return R.get("complete") is True or gd1_g2_user_stopped(R)


def decide_gd1(P, S, st):
    R = P.rj("G_report.json") or {}
    if R.get("complete") is True:
        return dict(key="complete", line=f"D1 G 끝: {core_h_of(R):,.0f} core-h (G_report complete).")
    g2 = (R.get("groups") or {}).get("G2") or {}
    return dict(key="G2 user stop", numbers=dict(G2={k: g2.get(k) for k in ("n_planned", "n_units", "n_missing")},
                                                 core_h=core_h_of(R)),
                line=f"D1 G 끝(사용자 결정 'Nurion'): G1 · G3 빠짐 없음, G-2 경로는 사용자가 {g2.get('n_units')}/"
                     f"{g2.get('n_planned')}반응에서 멈춤(1061710 취소; Nurion 라벨 실행 궤적이 대신함) → G_report "
                     f"complete=false이지만 끝난 것으로 보고 G-2를 다시 내지 않는다 ({core_h_of(R):,.0f} core-h).")


def guard_dft_stop(report):
    def g(P):
        f = P.c.mlt / "dft" / "STOP"
        if f.is_file():
            try:
                txt = f.read_text().strip()[:300]
            except OSError:
                txt = ""
            return f"$MLT/dft/STOP (라벨 수준 검사 실패: {txt})"
        R = P.rj(report) or {}
        if R.get("stop_file"):
            return f"{report} stop_file: {str(R['stop_file'])[:300]}"
        return None
    return g


def prog_gd1(P):
    r = P.c.mlt / "dft" / "runs"
    return count_dir(r / "G1") + count_dir(r / "G3") + count_dir(r / "G2")


# D0 extension (gen_dft.py + gen_d0.py, user decisions 'D0', 'D0-n3', 'D0-plus', 'D0-gauss'): plan_d0 (auto) -> g0_d0
# (G-0 on D0: cost fits) -> the budget check of the user's D0 selection (G total <= gen_dft.BUDGET_CORE_H) ->
# `gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100` (200 D0G2 training paths; 100 evaluation-only reactions
# that are NOT refined, D0-gauss) -> run D0G1 / D0G3 / run_g2 --d0 (g2_plan_d0.json rxns = the training units only)
# -> collect --d0.
def d0_plan(P):
    return P.rj("G_plan_d0.json") or {}


def d0_est(P):
    return P.rj("G0_estimate_d0.json") or {}


def d0_plan_n(J):
    b = J.get("budget") or {}
    return J.get("n_g2") if J.get("n_g2") is not None else b.get("n_g2_selected")


def d0_plan_h(J):
    b = J.get("budget") or {}
    v = J.get("n_holdout") if J.get("n_holdout") is not None else b.get("n_holdout")
    return _int(v, 0) if v is not None else 0


def fp_no_near(fp):
    """plan fingerprint without n_near (decision D0-nnear: --n-near may change after the G-0 estimate)."""
    return {k: v for k, v in fp.items() if k != "n_near"} if isinstance(fp, dict) else fp


def units_gd0p(P, S, st):
    return one("plan_d0", GCMD + ["plan_d0"], env="reactot", part="C", c=16, mem=32)


def d0_plan_full(J):
    """a production plan: written with a fingerprint and without --limit (a test `plan_d0 --limit N` never counts;
    gen_d0 records limit both at the top level and inside the fingerprint dict)."""
    fp = J.get("fingerprint")
    if not fp or J.get("limit"):
        return False
    return not (isinstance(fp, dict) and fp.get("limit"))


def done_gd0p(P):
    return d0_plan_full(d0_plan(P))


def units_gd0e(P, S, st):
    return one("g0_d0", GCMD + ["g0_d0"], env="reactot", part="C", c=16, mem=64)


def done_gd0e(P):
    E = d0_est(P)
    return (E.get("predicted_core_h") or {}).get("total") is not None and \
        fp_no_near(E.get("plan_fingerprint")) == fp_no_near(d0_plan(P).get("fingerprint"))


def d0_cost_basis(P):
    """(numbers, total) of the G budget check for the user's D0 selection (decision 'G-D0-budget'): D1 = the larger of
    the measured D1 G core-h (G_report core_h_measured.total) and the plan's budget basis (projection of the full D1
    plan); D0 G1 + G3 = the plan's prediction at its --n-near (G_plan_d0 budget.d0_g1_g3; else the estimate's by_group);
    D0G2 = n_train x the mean predicted core-h per D0G2 unit (plan budget rule, else the estimate) + the measured units
    on disk; + the G-4 reserve (150 core-h, gen_d0)."""
    E, J = d0_est(P), d0_plan(P)
    b = J.get("budget") or {}
    pc = E.get("predicted_core_h") or {}
    by = pc.get("by_group") or {}
    meas = core_h_of(P.rj("G_report.json"))
    cc = b.get("cap_check") or {}
    n_tr = d0f_param(P, "n_g2", D0_N_G2_TRAIN)
    if cc.get("total") is not None and _int(cc.get("n_train"), -1) == n_tr:
        # a gen_d0 D0-plus plan for this selection: its own cap check (D0-cap: D1 basis + D0 G1/G3 + measured D0G2 +
        # the predicted D0G2 of exactly these training units + the G-4 reserve), with the measured D1 if larger
        d1b = float(cc.get("d1_total") or 0.0)
        d1, d1_src = (meas, "D1 실측") if meas > d1b else (d1b, "D1 계획 기준(예상)")
        d0g13 = float(cc.get("d0_g1_g3") or 0.0)
        d0g2 = float(cc.get("d0g2_train_pred") or 0.0) + float(cc.get("d0g2_measured_on_disk") or 0.0)
        res = float(cc.get("g4_reserve") if cc.get("g4_reserve") is not None else G4_RESERVE_CORE_H)
        total = d1 + d0g13 + d0g2 + res
        nums = dict(d1_core_h=round(d1, 1), d1_source=d1_src, d1_measured=round(meas, 1), d1_basis=round(d1b, 1),
                    d0_g1_g3=round(d0g13, 1), d0g2_train=n_tr, d0g2_per_rxn=round(d0g2 / n_tr, 2) if n_tr else 0.0,
                    d0g2=round(d0g2, 1), g4_reserve=res, total=round(total, 1), budget=g_budget(P),
                    n_holdout=d0f_param(P, "n_holdout", D0_N_HOLDOUT), source="G_plan_d0 budget.cap_check (D0-cap)",
                    estimate_stop=E.get("stop"), estimate_budget=pc.get("budget"),
                    suggested_tasks=E.get("suggested_tasks"))
        return nums, total
    basis = float(((b.get("basis") or {}).get("d1_total")) or b.get("d1_total") or pc.get("d1_total") or 0.0)
    d1, d1_src = (meas, "D1 실측") if meas >= basis else (basis, "D1 계획 기준(예상)")
    d0g13 = b.get("d0_g1_g3")
    d0g13 = float(d0g13) if d0g13 is not None else float(by.get("D0G1") or 0.0) + float(by.get("D0G3") or 0.0)
    per = (((b.get("rule") or {}).get("g2_core_h_per_rxn") or {}).get("mean"))
    if per is None:
        n_r, g2p = _int(E.get("n_g2_rule"), 0), float(by.get("D0G2") or 0.0)
        per = (g2p / n_r) if n_r > 0 and g2p > 0 else 6.0
    d0g2 = n_tr * float(per) + float(b.get("d0g2_measured_core_h") or 0.0)
    total = d1 + d0g13 + d0g2 + G4_RESERVE_CORE_H
    nums = dict(d1_core_h=round(d1, 1), d1_source=d1_src, d1_measured=round(meas, 1), d1_basis=round(basis, 1),
                d0_g1_g3=round(d0g13, 1), d0g2_train=n_tr, d0g2_per_rxn=round(float(per), 2), d0g2=round(d0g2, 1),
                g4_reserve=G4_RESERVE_CORE_H, total=round(total, 1), budget=g_budget(P),
                n_holdout=d0f_param(P, "n_holdout", D0_N_HOLDOUT),
                source="n_train x mean predicted D0G2 core-h per pool unit (plan budget rule, else the estimate)",
                estimate_stop=E.get("stop"),
                estimate_budget=pc.get("budget"), suggested_tasks=E.get("suggested_tasks"))
    return nums, total


def d0f_param(P, k, dflt):
    v = (P.st("G-D0f").get("params") or {}).get(k)
    return _int(v, dflt) if v is not None else int(dflt)


def decide_gd0e(P, S, st):
    """the user's D0 selection (D0-n3: 3 structures per D0 reaction; D0-plus: 200 D0G2 training paths + 100
    evaluation-only reactions; D0-gauss: the evaluation reactions are not refined) against the G budget cap
    gen_dft.BUDGET_CORE_H (8,000, user decision). The estimate's own stop flag is not used: it was judged against the
    cap of its time (5,000 for g0_d0 1061949)."""
    nums, total = d0_cost_basis(P)
    bud = g_budget(P)
    n_tr, n_h = nums["d0g2_train"], nums["n_holdout"]
    if total > bud:
        return dict(key="over budget", numbers=nums,
                    halt=f"G 예산: {nums['d1_source']} {nums['d1_core_h']:,.0f} + D0 G1·G3 {nums['d0_g1_g3']:,.0f} + D0G2 "
                         f"{n_tr} × {nums['d0g2_per_rxn']} = {nums['d0g2']:,.0f} + G-4 {G4_RESERVE_CORE_H:.0f} = "
                         f"{total:,.0f} core-h > {bud:,.0f} (명령서 0-5 STOP; 사용자 상한, 남은 추천안 없음)",
                    options=["`param G-D0f.n_g2=<수>` — D0 경로 학습 반응 수를 줄여 plan_d0를 다시 하고 D0 본 실행으로 간다",
                             "`accept` — 예산 초과를 받아들이고 그대로 진행", "`skip` — D0 없이 진행(사용자 D0 결정과 다름)"])
    P.st("G-D0f").setdefault("params", {}).update(n_g2=int(n_tr), n_holdout=int(n_h))
    return dict(key=f"go n_g2={n_tr} holdout={n_h}", numbers=nums,
                line=f"D0 G-0 예산: {nums['d1_source']} {nums['d1_core_h']:,.0f} + D0 G1·G3 {nums['d0_g1_g3']:,.0f} + "
                     f"D0G2 {n_tr} × {nums['d0g2_per_rxn']} = {nums['d0g2']:,.0f} + G-4 {G4_RESERVE_CORE_H:.0f} = "
                     f"{total:,.0f} core-h ≤ {bud:,.0f}(gen_dft.BUDGET_CORE_H) → 사용자 결정 D0-plus · D0-gauss대로 "
                     f"`gen_d0.py plan_d0 --n-near {D0_N_NEAR} --n-g2 {n_tr} --n-holdout {n_h}` (학습용 D0G2 {n_tr}반응, "
                     f"평가 전용 {n_h}반응은 정제하지 않음). g0_d0의 stop 표시는 그때 상한({nums['estimate_budget']})으로 "
                     f"판정한 것이라 쓰지 않는다.")


def gd0f_n(P):
    return d0f_param(P, "n_g2", D0_N_G2_TRAIN)


def gd0f_h(P):
    return d0f_param(P, "n_holdout", D0_N_HOLDOUT)


def units_gd0f(P, S, st):
    return one("plan_d0", ["gen_d0.py", "plan_d0", "--n-near", str(D0_N_NEAR), "--n-g2", str(gd0f_n(P)), "--n-holdout",
                           str(gd0f_h(P))], env="reactot", part="C", c=16, mem=32)


def done_gd0f(P):
    J = d0_plan(P)
    return d0_plan_full(J) and _int(d0_plan_n(J), -1) == gd0f_n(P) and d0_plan_h(J) == gd0f_h(P)


def units_gd0(P, S, st):
    E = d0_est(P)
    sugg = E.get("suggested_tasks") or {}
    nf = gd0f_n(P)
    groups = [("D0G1", GCMD + ["run", "D0G1"], "reactot"), ("D0G3", GCMD + ["run", "D0G3"], "reactot"),
              ("D0G2", GCMD + ["run_g2"], "d1")]
    ep = st.get("epoch", 0)
    R = P.rj("G_report_d0.json") if ep > 0 else None
    u = []
    for g, cmd, env in groups:
        base = _int(sugg.get(g), 1)
        if g == "D0G2":                       # the training units of g2_plan_d0.json (holdout never run, D0-gauss)
            if nf <= 0:
                continue
            per = d0_cost_basis(P)[0]["d0g2_per_rxn"]
            base = max(2, n_tasks(nf, per / 8.0, 8, 4))      # 8 OptTS x 8 ranks per 64-core task (gen_dft G2_NPROCS)
        if R is not None and isinstance(R.get("groups"), dict):
            gr = R["groups"].get(g) or {}
            m = _int(gr.get("n_missing"), 0)
            if m <= 0:
                continue
            base = math.ceil(base * m / max(1, _int(gr.get("n_planned"), m)))
        # round 0: the lead's manual arrays are adopted unit by unit (g_run_d0g1 0-1, g_run_d0g3 0, g_run_d0g2 0-2):
        # n = their array size, so `@I n` slices and any resubmission match the adopted tasks (decision G-D0-adopt)
        n = fixed_n(P, st, g, lambda b=base, g_=g: max(1, min(8, max(b, adopted_n(st, g_)))))
        u += arr(g, cmd + ["@I", str(n)] + (["--d0"] if g == "D0G2" else []), n, env=env, part="C2", c=64, mem=240,
                 disk_gb=2.0)
    return u + one("collect", GCMD + ["collect", "--d0"], env="reactot", part="C", c=4, mem=32, deps=keys(u), sink=True,
                   disk_gb=2.0)


def guard_d0(P):
    for f in (P.c.mlt / "dft" / "STOP_D0", P.c.mlt / "dft" / "STOP"):
        if f.is_file():
            try:
                txt = f.read_text().strip()[:300]
            except OSError:
                txt = ""
            return f"$MLT/dft/{f.name} (D0 / 라벨 수준 검사 실패: {txt})"
    R = P.rj("G_report_d0.json") or {}
    return f"G_report_d0.json stop_file: {str(R['stop_file'])[:300]}" if R.get("stop_file") else None


def done_gd0(P):
    R = P.rj("G_report_d0.json") or {}
    return R.get("complete") is True and R.get("stop_file") is None


def prog_gd0(P):
    r = P.c.mlt / "dft" / "runs"
    return count_dir(r / "D0G1") + count_dir(r / "D0G3") + count_dir(r / "D0G2")


# ---------------------------------------------------------------- N: the Nurion D1 run tars (user decision 'Nurion')
# inv_nurion.py: per uploaded tar `scan <tar> ::: verify <tar>` (cpu1/cpu2), then `collect` -> $MLT/dft/nurion.extxyz.gz
# + results/N_report.json. The user deletes the tar copies by hand (decision 'tar 삭제 방식' 11-04 23:1x: inv_nurion
# release is not used). The FT data stages wait for N_report complete (all 23 result tars) or for the lead's DECISIONS
# line '- **N partial ok ...' (decision N-gate).
def nur_short(stem):
    """= inv_nurion.short: d1_20261010_g01_002 -> g01_002 (its manual job names nur_<short>); extras drop 'd1_'."""
    m = re.fullmatch(r"d1_\d{8}_(g\d+_\d+)", str(stem))
    return m.group(1) if m else str(stem).replace("d1_", "")


def nur_out(P):
    return P.c.mlt / "dft" / "nurion"


def nur_tars(P):
    """[{tar, file, state}] — the stdlib mirror of inv_nurion.tar_states (cached per pass). Registry = the result
    manifests D1_build/nurion/d1_*.manifest.tsv (<stem>.tar) + the NUR_EXTRA bundles (file name from their .sha256).
    state: released (<out>/<stem>/.done and the upload copy gone: the user deleted it), done (.done, copy still there),
    uploading (size != index/bundles.tsv bytes, or modified < 15 min ago), pending (uploaded, not scanned), deleted
    (deleted_tars.tsv without .done), not_uploaded."""
    if getattr(P, "_nur", None) is not None:
        return P._nur
    up = Path(P.c.nur_upload)
    mani, out = up / "D1_build" / "nurion", nur_out(P)
    sizes = {}
    try:
        lines = (mani / "index" / "bundles.tsv").read_text().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            r = dict(zip(hdr, ln.split("\t")))
            sizes[r.get("bundle")] = _int(r.get("bytes"), None)
    except (OSError, IndexError):
        pass
    reg = {m.name[:-len(".manifest.tsv")]: m.name[:-len(".manifest.tsv")] + ".tar"
           for m in sorted(mani.glob("d1_*.manifest.tsv"))}
    for s in sorted(mani.glob("*.sha256")):
        try:
            t = s.read_text().split()
        except OSError:
            continue
        if len(t) >= 2:
            fn = t[1].lstrip("*")
            stem = re.sub(r"\.tar(\.gz|\.xz)?$", "", fn)
            if stem in NUR_EXTRA:
                reg.setdefault(stem, fn)
    dele = set()
    try:
        for ln in (out / "deleted_tars.tsv").read_text().splitlines():
            t = ln.split("\t")
            if t and t[0] and not t[0].startswith("#"):
                dele.add(re.sub(r"\.tar(\.gz|\.xz)?$", "", Path(t[0]).name))
    except OSError:
        pass
    rows = []
    for stem, fn in reg.items():
        try:
            s = (up / fn).stat()
            size, age = s.st_size, P.t - s.st_mtime
        except OSError:
            size, age = None, None
        if (out / stem / ".done").is_file():
            state = "released" if size is None else "done"
        elif size is not None:
            state = "uploading" if ((sizes.get(stem) and size != sizes[stem]) or age < NUR_SETTLE_S) else "pending"
        else:
            state = "deleted" if stem in dele else "not_uploaded"
        rows.append(dict(tar=stem, file=fn, state=state))
    P._nur = rows
    return rows


def nur_report_ok(P, n_done):
    """N_report.json written by a collect that saw every processed tar (tars_processed >= n_done, newer than each
    .done)."""
    p = P.c.res / "N_report.json"
    J, t = P.rjp(p) or {}, P.mt(p)
    if not J or t is None or _int(J.get("tars_processed"), -1) < n_done:
        return False
    out = nur_out(P)
    return all((P.mt(out / r["tar"] / ".done") or 0) <= t for r in nur_tars(P) if r["state"] in ("done", "released"))


def units_nscan(P, S, st):
    """one unit per uploaded, not yet scanned tar: `inv_nurion.py scan <tar> --workers 8 ::: inv_nurion.py verify
    <tar>` (cpu1/cpu2, 8 cores, 24 GB; done = <out>/<tar>/.done); a collect after them whenever more tars are processed
    than the last N_report counted. Tars still uploading or not uploaded wait (the stage re-arms, decision N-scan).
    After a successful verify the uploaded copy is deleted by `release` (user 2026-11-05 00:1x: "업로드가 끝나면 …
    파일들을 뽑아내고 … tar 파일들 전부 삭제"; release re-checks sha256, .done, report and the extxyz read-back first)."""
    rows = nur_tars(P)
    u = []
    for r in rows:
        if r["state"] == "pending":
            stem = r["tar"]
            u += one(f"scan/{nur_short(stem)}", ["inv_nurion.py", "scan", stem, "--workers", "8", ":::", "inv_nurion.py",
                                                 "verify", stem, ":::", "inv_nurion.py", "release", stem, "--reviewed-by",
                                                 "user-auto-delete-2026-11-05"], env="reactot", part="C", c=8, mem=24, disk_gb=1.0,
                     done=lambda P_, s=stem: (nur_out(P_) / s / ".done").is_file())
    nd = sum(r["state"] in ("done", "released") for r in rows)
    goal = sum(r["state"] in ("pending", "done", "released") for r in rows)      # processed once these scans end
    st.setdefault("params", {})["cur_tag"] = f"{nd} tars processed"
    if u or (nd and not nur_report_ok(P, nd)):
        u += one(f"collect@{goal}", ["inv_nurion.py", "collect"], env="reactot", part="C", c=4, mem=64, deps=keys(u),
                 sink=True, disk_gb=5.0)
    return u


def done_nscan(P):
    """N_report complete (inv_nurion: all 23 result tars processed; the optional extras are not required) and up to
    date with every processed tar."""
    J = P.rj("N_report.json") or {}
    nd = sum(r["state"] in ("done", "released") for r in nur_tars(P))
    return bool(J.get("complete") is True and nd and nur_report_ok(P, nd))


def prog_nscan(P):
    return sum(r["state"] in ("done", "released") for r in nur_tars(P))


def n_partial_ok(P):
    """the lead's recorded decision: a DECISIONS.md line starting '- **N partial ok' (or RESOLVE param
    N-scan.partial_ok=1) lets the FT data stages start before every Nurion tar is processed."""
    if (P.st("N-scan").get("params") or {}).get("partial_ok"):
        return True
    try:
        return bool(NUR_PARTIAL_RX.search(P.c.dec_path.read_text()))
    except OSError:
        return False


def n_ready(P):
    """decision N-gate: FT data may be built once the Nurion data are final (N-scan done / skipped, or N_report
    complete) or the lead recorded 'N partial ok'."""
    return P.st("N-scan").get("status") in ("done", "skipped") or done_nscan(P) or n_partial_ok(P)


N_NEED = ("Nurion data final (N_report complete) or the lead's DECISIONS line '- **N partial ok'", n_ready)


# ---------------------------------------------------------------- D0 supplementary geometry evaluation (D0-plus)
def units_d0eval(P, S, st):
    """`ts_geom.py d0_eval run @I n` (GPU, model-major slices of FT_all_s<i>_al x {label = Coley TS, g2 = G2c xTB TS}
    x the D0 evaluation reactions) -> `ts_geom.py d0_eval report` -> results/D0_eval_report.json (report only)."""
    n = fixed_n(P, st, "run", lambda: len(D0E_MODELS))
    run = arr("run", ["ts_geom.py", "d0_eval", "run", "@I", str(n)], n, env="mlts", part="GA", c=16, mem=64, gpu=1,
              disk_gb=0.2)
    return run + one("report", ["ts_geom.py", "d0_eval", "report"], env="mlts", part="C", c=4, mem=16, deps=keys(run),
                     sink=True)


# ---------------------------------------------------------------- Z
def z_done(P, c):
    """unavailable, or finished with records for (almost) the whole list: zero_shot sets 'finished' even when its
    ts_geom runner stopped on the wall or after 20 pipeline errors, so the record count is checked too (<= 1 % of
    the list may lack a record: a reaction that keeps raising a pipeline error)."""
    J = P.rjp(P.c.mlt / "z" / c / "z.json") or {}
    if J.get("available") is False:
        return True
    if not J.get("finished"):
        return False
    n = _int(J.get("n_rxn"), -1)
    sc = (((J.get("geometry") or {}).get("all") or {}).get("status_counts") or {})
    try:
        cov = sum(int(v) for v in sc.values())
    except (TypeError, ValueError):
        cov = 0
    return n >= 0 and cov >= n - math.floor(Z_TOL * n)


def units_z(P, S, st):
    out = []
    for c in Z_CANDS:
        if c == "xtb":
            res = dict(part="C", c=32, mem=64, gpu=0)
        else:
            res = dict(part="G48" if c == "MACE-OMOL-0" else "GA", c=8, mem=64, gpu=1)
        out += one(c, ["zero_shot.py", "run", c], env="mlts", done=lambda P, c_=c: z_done(P, c_), disk_gb=0.5, **res)
    return out


SPEED_TAGS = ("MACE-OMOL-0|float64|", "MACE-MH-1|float64|", "AIMNet2||", "MACE-OFF24|float64|")


def done_zspeed(P):
    J = P.rjp(P.c.mlt / "z" / "speed.json") or {}
    return all(t in J and (J[t].get("ef_ms_median") is not None or J[t].get("error")) for t in SPEED_TAGS)


def done_zrep(P):
    p = P.c.res / "Z_report.json"
    J = P.rjp(p) or {}
    if "ranking" not in J:
        return False
    t = P.mt(p)
    deps = [P.mt(P.c.mlt / "z" / c / "z.json") for c in Z_CANDS] + [P.mt(P.c.mlt / "z" / "speed.json")]
    return all(d is None or t >= d for d in deps)


def decide_zrep(P, S, st):
    R = P.rj("Z_report.json") or {}
    top = [c for c in (R.get("top2_for_ft_pilot") or []) if c]
    rank = R.get("ranking") or []
    if not top:
        return dict(key="no eligible", halt="Phase Z: 순위에 든 후보가 없다(Si를 다루는 MLIP 없음 또는 모두 실패)",
                    numbers=dict(ranking=rank, candidates={k: (v or {}).get("available") for k, v in
                                                           (R.get("candidates") or {}).items()}))
    pil = [Z2FT.get(c, c) for c in top]
    P.state.setdefault("params", {})["pilots"] = pil
    extra = (" 대상 후보가 하나뿐이라 그 하나로 시범한다(추천안: 명령서 4절 '상위 2개'를 채울 수 없음)." if len(pil) == 1 else "")
    return dict(key="pilots " + ",".join(pil), line=f"Phase Z 순위 {', '.join(rank)} → FT 시범(fold 0) = "
                                                     f"{', '.join(pil)} (finetune.py 이름).{extra}")


# ---------------------------------------------------------------- FT
def units_ftdp(P, S, st):
    runs = [f"FTpilot_{c}_0" for c in pilots(P)]
    return one("data", ["finetune.py", "data"] + runs, env="mlts", part="C", c=4, mem=48) if runs else []


def done_ftdp(P):
    p = pilots(P)
    return bool(p) and all(data_ok(P, f"FTpilot_{c}_0") for c in p)


def units_ftp(P, S, st):
    out = []
    for c in pilots(P):
        out += train_unit(c, f"FTpilot_{c}_0", c)
    return out


def done_ftp(P):
    p = pilots(P)
    return bool(p) and all(train_done(P, f"FTpilot_{c}_0") for c in p)


def prog_train(runs_fn):
    return lambda P: sum(count_lines(P.c.mlt / "ft" / r / "epochs.jsonl") for r in runs_fn(P))


def truthy(v):
    return v is True or str(v).strip().lower() == "true"


def fail_class(r):
    """per-reaction FT-2 criterion on a geom.csv row (= ts_geom.fail_class): None = passes or not attempted; else
    'not_converged' (i), 'n_imag' (ii: converged, n_imag != 1, MLIP modes below -50 cm-1) or 'other_saddle' (iii:
    converged, n_imag == 1, not the same TS or |dd| > FT2_DD_CUT_A = 0.030 A)."""
    if str(r.get("status")) in NOT_ATTEMPTED:
        return None
    if not truthy(r.get("converged")):
        return "not_converged"
    ni = _f(r.get("n_imag"))
    if ni is None or int(round(ni)) != 1:
        return "n_imag"
    dd = _f(r.get("dd_max"))
    if not truthy(r.get("same_ts")) or dd is None or dd > FT2_DD_CUT_A:
        return "other_saddle"
    return None


def fail_counts(rows):
    out = dict(n_attempted=0, n_pass=0, **{c: 0 for c in FAIL_CLASSES})
    for r in rows:
        if str(r.get("status")) in NOT_ATTEMPTED:
            continue
        out["n_attempted"] += 1
        out[fail_class(r) or "n_pass"] += 1
    return out


def row_id(r):
    return _int(r.get("rxn_id"), None)


def recalc_dir(od):
    od = Path(od)
    return od.parent / f"{od.name}{RECALC_SUFFIX}"


GEOM_KEEP = ("rxn_id", "status", "converged", "dd_max", "rmsd_heavy", "rmsd_direct", "same_ts", "n_imag", "n_cycles",
             "wall_s", "dipole", "g1_class", "fold_T1", "start", "model", "recalc", "tight")


def geom_rows(od, P=None):
    """per-reaction rows of a ts_geom out_dir from its atomic records <od>/recs/*.json (authoritative: geom.csv /
    summary.json are rewritten by whichever task summarizes last, and a manual run's committed ts_geom may split its
    summary by another fold's E/Z list — never read for the numbers); <od>/geom.csv when there is no recs/ folder
    (older layouts, simulator). Cached per pass by the recs/ folder mtime."""
    od = Path(od)
    rd = od / "recs"
    try:
        m = rd.stat().st_mtime
    except OSError:
        return read_geom_csv(od / "geom.csv")
    cache = getattr(P, "_rows", None)
    if cache is not None and str(rd) in cache and cache[str(rd)][0] == m:
        return cache[str(rd)][1]
    rows = []
    try:
        names = sorted(n for n in os.listdir(rd) if n.endswith(".json"))
    except OSError:
        names = []
    for n in names:
        try:
            r = json.loads((rd / n).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and row_id(r) is not None:
            rows.append({c: r.get(c) for c in GEOM_KEEP if c in r})
    if cache is not None:
        cache[str(rd)] = (m, rows)
    return rows


def recalc_state(od, P=None):
    """(first-pass rxn ids failing the per-reaction criterion, re-measure rows, failed ids without a re-measure row)
    of a ts_geom out_dir (records of the first pass and of <out_dir>_recalc5, geom_rows)."""
    fails = [row_id(r) for r in geom_rows(od, P) if fail_class(r)]
    rows = geom_rows(recalc_dir(od), P)
    have = {row_id(r) for r in rows}
    return fails, rows, [x for x in fails if x not in have]


def akey_live(P, st, akey):
    """a job of a unit akey[i] of this stage is in squeue."""
    rx, live = re.compile(re.escape(akey) + r"\[\d+\]"), getattr(P, "live", None) or {}
    return any(rx.fullmatch(k) and any(j.get("id") in live for j in us.get("jobs", []))
               for k, us in (st.get("units") or {}).items())


def first_pass_final(P, st, od, nl, akey):
    """the first ts_geom pass of one (run, start) is final: no first-pass task of the stage in squeue (adopted manual
    tasks included, e.g. the gate fold's ft2_spec_f4_<start>), and every list reaction recorded or >= 99.5 % recorded
    (the reactions left raise pipeline errors; Stage.partial)."""
    if nl <= 0 or akey_live(P, st, akey):
        return False
    return geom_done(P, od, nl) or geom_cov(P, od) >= math.ceil(GEOM_PARTIAL * nl)


def recalc_units(P, st, run, cand, list_name, out_root, starts=STARTS):
    """user 11-05 (3): after the first pass of a start is final, its reactions that fail the per-reaction criterion are
    run again with a new MLIP Hessian every RECALC_K = 5 cycles (`ts_geom.py run ... --recalc 5 --only-failed
    <out_dir>` -> <out_dir>_recalc5/); a start whose failed reactions all have a re-measure row is skipped."""
    nl, out = n_list(P, list_name), []
    for s in starts:
        od = out_root / run / s
        if not first_pass_final(P, st, od, nl, f"{run}/{s}"):
            continue
        fails, _rows, miss = recalc_state(od, P)
        if not miss:
            continue
        ak = f"{run}/{s}{RECALC_SUFFIX}"
        n = fixed_n(P, st, ak, lambda: n_tasks(len(fails), RECALC_WALL_X * geom_unit_h(P, cand), 14, 4,
                                               wall=GEOM_WALL_H))
        out += arr(ak, ["ts_geom.py", "run", run, str(P.c.list_path(list_name)), s, str(recalc_dir(od)), "@I", str(n),
                        "--recalc", str(RECALC_K), "--only-failed", str(od)], n, env="mlts", part="GA", c=16, mem=64,
                   gpu=1, disk_gb=0.3)
    return out


def units_ft2p(P, S, st):
    """FT2-p on the gate fold (GATE_FOLD = 4): per pilot c with FTpilot_<c>_4 trained, the first pass of both starts on
    f4_test and, once a start's first pass is final, its recalc-5 re-measure. A start whose manual array
    ft2_spec_f4_<start> is in squeue gets no unit (adopted by name; nothing is submitted beside it)."""
    out, root = [], P.c.mlt / "ft2"
    starts = [s for s in STARTS if not geom_spec_live(P, GATE_FOLD, s)]
    for c in pilots(P):
        run = gate_run(c)
        if not train_done(P, run):
            continue
        out += geom_units(P, st, run, c, GATE_LIST, root, starts=starts)
        out += recalc_units(P, st, run, c, GATE_LIST, root, starts=starts)
    return out


def ft2p_dirs(P):
    return [P.c.mlt / "ft2" / gate_run(c) / s for c in pilots(P) for s in STARTS]


def done_ft2p(P):
    """first pass complete for both starts of every pilot's gate-fold model FTpilot_<c>_4 on f4_test, and every failed
    reaction re-measured (recalc 5)."""
    p, nl = pilots(P), n_list(P, GATE_LIST)
    return bool(p) and all(geom_done(P, od, nl) and not recalc_state(od, P)[2] for od in ft2p_dirs(P))


def recalc_partial(P, runs_lists):
    """Stage.partial core of a ts_geom stage with the recalc-5 re-measure (FT2-p, FT2-r2; runs_lists = [(run, list
    name)] under $MLT/ft2): the first pass >= 99.5 % recorded in every folder (geom_partial) and the re-measure of the
    failed reactions complete except at most max(1, 0.5 %) per folder (a failed reaction without a re-measure record
    keeps its first-pass failure)."""
    rl = [(r, ln) for r, ln in runs_lists if r]
    ok, txt = geom_partial(geom_dirs(lambda P_: rl, lambda P_: P_.c.mlt / "ft2"))(P)
    if not ok:
        return ok, txt
    rows = []
    for r, _ln in rl:
        for s in STARTS:
            fails, _r, miss = recalc_state(P.c.mlt / "ft2" / r / s, P)
            rows.append((r, s, len(fails), len(miss)))
    bad = [x for x in rows if x[3] > max(1, x[2] - math.ceil(GEOM_PARTIAL * x[2]))]
    if bad:
        return False, f"recalc-{RECALC_K} re-measure incomplete in {len(bad)} folders"
    nm = sum(x[3] for x in rows)
    return True, txt + (f"; recalc-{RECALC_K}: {nm} failed reactions without a re-measure record keep their first-pass "
                        f"result (failure)" if nm else "")


def partial_ft2p(P):
    """Stage.partial of FT2-p: recalc_partial over FTpilot_<c>_4 x f4_test (gate fold)."""
    return recalc_partial(P, [(gate_run(c), GATE_LIST) for c in pilots(P)])


def prog_recalc(P, runs):
    """Stage.progress of FT2-p / FT2-r2: first-pass + recalc-5 records under $MLT/ft2."""
    root = P.c.mlt / "ft2"
    return geom_progress(P, runs, root) + sum(count_dir(root / r / f"{s}{RECALC_SUFFIX}" / "recs") for r in runs
                                              for s in STARTS)


def prog_ft2p(P):
    return prog_recalc(P, [gate_run(c) for c in pilots(P)])


def pilot_speeds(P, cands):
    """GPU E+F ms per call for the tie-break: Z_report.speed_variants[default tag] when it covers every pilot (one
    z_speed job on one GPU: a fair comparison), else the per-candidate z.json speeds (each job's own GPU type)."""
    Z = P.rj("Z_report.json") or {}
    sv = Z.get("speed_variants") or {}
    same = {c: (sv.get(SPEED_DEFAULT_TAG.get(FT2Z.get(c, c), "")) or {}).get("ef_ms_median") for c in cands}
    if cands and all(v is not None for v in same.values()):
        return same, "speed.json (z_speed: 한 job · 한 GPU)"
    own = {c: ((Z.get("candidates") or {}).get(FT2Z.get(c, c)) or {}).get("ef_ms") for c in cands}
    return own, "후보별 z.json (job마다 GPU 종류가 다를 수 있음 — speed.json에 시범 후보가 다 있지 않음)"


def ez_map(P, k=0):
    """{rxn_id: True (an E/Z sibling on the training side) | False} from $MLT/pipeline/lists/f<k>_test_ez.csv
    (ez_groups.py run; fold 0 by default); {} when the file or its ez_sibling_in_train column is missing."""
    import csv
    try:
        with open(P.c.list_path(EZ_LIST_TMPL.format(k=k)), newline="") as f:
            rd = csv.DictReader(f)
            if "ez_sibling_in_train" not in (rd.fieldnames or []):
                return {}
            return {row_id(r): truthy(r.get("ez_sibling_in_train")) for r in rd if row_id(r) is not None}
    except OSError:
        return {}


def ez_lists_done(P):
    """the five E/Z sibling lists f0..f4_test_ez.csv readable (FT2-r2 waits for them)."""
    return all(ez_map(P, k) for k in range(5))


def list_ids(P, name):
    return [x for x in (row_id(r) for r in read_geom_csv(P.c.list_path(name))) if x is not None]


def ft2p_rows(P, run, s):
    """(first-pass rows, merged rows, n failed in the first pass, n of those re-measured). merged = per reaction the
    recalc-5 row when the first pass failed and a re-measure row exists, else the first-pass row (user 11-05 (3): a
    reaction fails only if it also fails with recalc 5)."""
    od = P.c.mlt / "ft2" / run / s
    base = geom_rows(od, P)
    rc = {row_id(r): r for r in geom_rows(recalc_dir(od), P) if str(r.get("status")) not in NOT_ATTEMPTED}
    merged, n_fail, n_re = [], 0, 0
    for r in base:
        if fail_class(r):
            n_fail += 1
            x = rc.get(row_id(r))
            if x is not None:
                n_re += 1
                merged.append(x)
                continue
        merged.append(r)
    return base, merged, n_fail, n_re


def ft2_split(rows, missing, ez):
    """pooled FT-2 numbers (+ failure classes) of the E/Z 'with' / 'without sibling' groups; missing = list reactions
    without any record (attempted failures of their group)."""
    return ft2_split_parts([(rows, missing, ez)])


def ft2_split_parts(parts):
    """ft2_split pooled over folds: parts = [(rows, missing ids, E/Z map of that fold)] (FT2-r2: one part per fold k,
    each with its own f<k>_test_ez.csv map)."""
    out = {}
    for lab, val in (("with", True), ("without", False)):
        rr = [r for rows, _m, ez in parts for r in rows if ez.get(row_id(r)) is val]
        nm = sum(1 for _r, miss, ez in parts for x in miss if ez.get(x) is val)
        out[lab] = dict(pooled(rr, nm), fail_classes=fail_counts(rr))
    out["n_unlisted"] = sum(sum(1 for r in rows if row_id(r) not in ez) + sum(1 for x in miss if x not in ez)
                            for rows, miss, ez in parts)
    return out


def fail_summary(base, merged, n_fail, n_re, n_miss):
    """failure classes of one start before / after the recalc-5 re-measure (FT2-p / FT2-r2 decision numbers)."""
    aft = fail_counts(merged)
    n_left = sum(aft[k] for k in FAIL_CLASSES)
    return dict(first_pass=fail_counts(base), after_recalc=aft, n_failed_first_pass=n_fail, n_remeasured=n_re,
                n_recovered=n_fail - n_left, n_failed_after=n_left, n_missing_records=n_miss)


def fail_txt(cls):
    """{start: fail_summary} -> the decision-line text of the classes before / after the re-measure."""
    cl = lambda c: " · ".join(f"{k} {c[k]}" for k in FAIL_CLASSES)        # noqa: E731
    return "; ".join(f"{s}: 1차 실패 {cls[s]['n_failed_first_pass']}({cl(cls[s]['first_pass'])}) → recalc "
                     f"{RECALC_K} 재측정 {cls[s]['n_remeasured']} → 남은 실패 {cls[s]['n_failed_after']}"
                     f"({cl(cls[s]['after_recalc'])})" for s in STARTS)


def ft2_txt(x):
    """one pooled FT-2 table as decision-line text."""
    return (f"같은 TS {fmt(x.get('same_ts_frac'))} · |Δd| 중앙값 {fmt(x.get('dd_median_A'), 4)} · 90 % "
            f"{fmt(x.get('dd_p90_A'), 4)} Å · RMSD {fmt(x.get('rmsd_median_A'), 4)} Å ({x.get('n_attempted')}반응)")


def ez_rule(sp):
    """(split?, |diff same-TS fraction|, |diff median |dd||) of the label-start groups: split when the difference is
    > EZ_SAME_DIFF (0.05) OR > EZ_DD_DIFF_A (0.002 A); never when a group is empty."""
    w, wo = sp["with"], sp["without"]
    if not w.get("n_attempted") or not wo.get("n_attempted"):
        return False, None, None
    ds = (abs(w["same_ts_frac"] - wo["same_ts_frac"]) if None not in (w.get("same_ts_frac"), wo.get("same_ts_frac"))
          else None)
    dd = (abs(w["dd_median_A"] - wo["dd_median_A"]) if None not in (w.get("dd_median_A"), wo.get("dd_median_A"))
          else None)
    return bool((ds is not None and ds > EZ_SAME_DIFF) or (dd is not None and dd > EZ_DD_DIFF_A)), ds, dd


FT2P_RULE = ("per-reaction criterion = converged, n_imag == 1 (MLIP modes below -50 cm-1), same TS and |dd| <= "
             f"{FT2_DD_CUT_A} A; failures (i) not_converged (ii) n_imag != 1 (iii) other_saddle; every first-pass "
             f"failure (ts_geom default analytic:30 = D1 Recalc_Hess 30) is run again from the same start with "
             f"analytic:{RECALC_K} (ts_geom --recalc {RECALC_K} --only-failed, <start>{RECALC_SUFFIX}/) and the "
             f"re-measured record replaces it, so a reaction fails only if it also fails with recalc {RECALC_K}; E/Z: if "
             f"the label-start 'with sibling' and 'without sibling' groups differ by > {EZ_SAME_DIFF} same-TS fraction "
             f"or > {EZ_DD_DIFF_A} A median |dd|, the FT-2 (a) gate is judged on 'without sibling' only, else on all "
             f"test reactions of the gate fold; an empty group never splits (user 11-05, fixed before the FT-2 results); "
             f"gate fold {GATE_FOLD}: FTpilot_<c>_{GATE_FOLD} on {GATE_LIST} with {EZ_LIST_NAME} (user 11-05 15:5x, "
             f"changed from fold 0 before any FT-2 number was seen); numbers from the per-reaction records (recs/*.json)")


def ez_grp_txt(x):
    """one E/Z group as decision-line text; an empty group says so."""
    return ft2_txt(x) if (x or {}).get("n_attempted") else "0반응(무리 비어 있음)"


def fold_model_state(P, cand, k):
    """'학습 끝' / '학습 중' / '없음' of the first-round model FTpilot_<cand>_<k> (fold 0: the FT-p job; folds 1-4: the
    manual ft_spec_<cand>_<k>)."""
    if train_done(P, f"FTpilot_{cand}_{k}"):
        return "학습 끝"
    if k == 0:
        S = P.by.get("FT-p") if hasattr(P, "by") else None
        return "학습 중(FT-p)" if S is not None and P.stage_live(S) else "없음"
    return "학습 중" if spec_live(P, cand, k) else "없음"


def decide_ft2p(P, S, st):
    """FT2-p (round 1): base MLIP + the FT-2 (a) gate on the GATE FOLD = 4 (user 11-05 (5) gate; HALT on failure).
    Gate fold: the user's 10:4x instruction judged fold 0; at 11-05 15:5x the user changed it to fold 4 ("바꿔줘")
    before any FT-2 number was seen: FTpilot_<c>_4 (ft_spec_AIMNet2_4) finished training first and its FT-2
    measurement already runs as the lead's manual arrays ft2_spec_f4_label / ft2_spec_f4_g2 (adopted unit by unit,
    n = 2), while the fold-0 model FTpilot_<c>_0 (FT-p) still trains. So FT2-p comes after Z-rep + lists and needs
    FTpilot_<c>_4 trained, not FT-p. The base MLIP is ranked on the gate-fold numbers; the fold-0 meaning of
    FTpilot_<base>_0 is kept: params base_run = FTpilot_<base>_0 = FT_0 = r1_run(0) (G-4 waits for FT-p and FT-r1),
    params gate_run = FTpilot_<base>_4. FT2-r1 (report only) measures fold 0 and folds 1-3 (fold 4 is measured here);
    FT2-f0 (report only) writes the fold-0 numbers split by f0_test_ez once fold 0's first pass is final. The numbers
    are recomputed from the per-reaction records (geom_rows: recs/*.json), never from a summary.json. Fold 1-4
    order: the user changed it at 11-05 13:0x ("지금미리 시작하고 FT-A 실패하면 그때가서 수정") — the fold 1-4 AIMNet2
    models no longer wait for this gate; they train speculatively now as the manual jobs ft_spec_AIMNet2_1..4 (runs
    FTpilot_AIMNet2_<k>, DECISIONS 'fold 1–4 미리 학습') and r1_run reuses them as FT_<k>. The stages FT-d1 / FT-r1 /
    FT2-r1 / FT2-f0 / G4 still come after FT2-p, so a HALT here stops them (FT-p keeps training fold 0); the HALT names
    the first-round models and their state and leaves the next step to the user (earlier user sequence: G-4 (fold 0)
    -> re-measure -> MACE-OMOL-0 lr <= 1e-4 fold 0 only -> STOP; the user said to fix it then). Rules fixed before any
    FT-2 number was seen (user 11-05 (2) / (3)):
      re-measure — a reaction passes when converged, n_imag == 1 (MLIP-Hessian modes below -50 cm-1, as before), same TS
        and |dd| <= FT2_DD_CUT_A = 0.030 A; first-pass failures are classified (i) not_converged, (ii) n_imag != 1,
        (iii) other_saddle (converged, n_imag == 1, other TS or |dd| > 0.030 A), per start. BEFORE a failure counts
        as a PES failure it is run again from the same start with a new MLIP Hessian every RECALC_K = 5 cycles (the
        first pass uses analytic:30 = the D1 config_run.yaml Recalc_Hess 30; never less frequent), as part of this
        stage (recalc_units, <start>_recalc5/); the gate uses per reaction the re-measured record when it exists, so
        a reaction fails only if it also fails with recalc 5 (a failure without a re-measure record stays a failure).
      E/Z siblings — $MLT/pipeline/lists/f4_test_ez.csv (ez_groups.py, the gate fold's own list): a fold-4 test
        reaction is 'with sibling' when a training-side D1 core reaction (fold != 4, incl. the -1 anchor-only cores and
        the fold-4 validation cores) has the same stereo-free but a different stereo-kept reactant-pair identity. FT-2
        (a) (label start) and (b) (G2 start) are reported for 'with' and 'without'. THRESHOLD: if the label-start
        groups differ by more than EZ_SAME_DIFF = 0.05 in the same-TS fraction OR more than EZ_DD_DIFF_A = 0.002 A in
        the median |dd|, the FT-2 (a) gate is judged on the 'without sibling' group only (the decision line says so and
        a note is appended to results/DEVIATIONS.md); otherwise on all fold-4 test reactions. An empty group (f4_test_ez
        lists 0 'with sibling' of 2,195) never splits: no comparison is made and the decision line says which group is
        empty; the gate uses all fold-4 test reactions.
    The base MLIP is ranked on the merged label-start numbers of all gate-fold test reactions (rank_key_cmp). params
    base / base_run / gate_run / gate_fold are written BEFORE the pass / fail branch (also before the E/Z-list HALT), so
    the user's 11-05 17:0x (1) `RESOLVE: accept` of a failed gate lets FT-d1 -> FT-r1 -> G4 -> FT-d2 -> FT-r2 -> FT2-r2
    continue with the base set (FT-r1 / G4 need it: BASE_NEED)."""
    from functools import cmp_to_key
    k = GATE_FOLD
    speeds, src = pilot_speeds(P, pilots(P))
    ids_g, ez = list_ids(P, GATE_LIST), ez_map(P, k)
    data, rows = {}, []
    for c in pilots(P):
        run = gate_run(c)
        d = {}
        for s in STARTS:
            base, merged, n_fail, n_re = ft2p_rows(P, run, s)
            have = {row_id(r) for r in base}
            miss = [x for x in ids_g if x not in have]
            d[s] = dict(base=base, merged=merged, miss=miss, n_fail=n_fail, n_re=n_re, all=pooled(merged, len(miss)),
                        first=pooled(base, len(miss)))
        data[c] = d
        g = d["label"]["all"]
        rows.append(dict(cand=c, run=run, same=g.get("same_ts_frac"), dd=g.get("dd_median_A"), ef=speeds.get(c),
                         n=g.get("n_attempted"), rmsd=g.get("rmsd_median_A")))
    order = sorted(rows, key=cmp_to_key(rank_key_cmp))
    b = order[0]
    bc, base_run = b["cand"], f"FTpilot_{b['cand']}_0"
    P.state.setdefault("params", {}).update(base=bc, base_run=base_run, gate_run=b["run"], gate_fold=k)
    desc = "; ".join(f"{r['cand']} 같은 TS {fmt(r['same'])} · |Δd| 중앙값 {fmt(r['dd'], 4)} Å · RMSD {fmt(r['rmsd'], 4)} Å "
                     f"({r['n']}반응), E+F {fmt(r['ef'], 1)} ms" for r in rows)
    D = data[bc]
    a, g2 = D["label"]["all"], D["g2"]["all"]
    cls = {s: fail_summary(D[s]["base"], D[s]["merged"], D[s]["n_fail"], D[s]["n_re"], len(D[s]["miss"]))
           for s in STARTS}
    cls_txt = fail_txt(cls)
    base_nums = dict(rows=rows, label=a, g2=g2, label_first_pass=D["label"]["first"], g2_first_pass=D["g2"]["first"],
                     fail_classes=cls, thresholds=FT2_A, dd_cut_A=FT2_DD_CUT_A, recalc=f"analytic:{RECALC_K}",
                     speed_source=src, rule=FT2P_RULE, gate_fold=k, gate_list=GATE_LIST, ez_list=EZ_LIST_NAME,
                     gate_run=b["run"], base_run=base_run)
    if not ez:
        return dict(key=f"base {bc} E/Z list missing", numbers=base_nums,
                    halt=f"FT2-p 판정에 필요한 관문 fold {k}의 E/Z 형제 목록 $MLT/pipeline/lists/{EZ_LIST_NAME}.csv가 없다"
                         f"(사용자 11-05 (2) 규칙)",
                    options=[f"`retry` — `ez_groups.py run --folds {k}`(MLTENV=mlts, CPU)으로 목록을 만든 뒤 다시 판정",
                             "`accept` — E/Z 나눔 없이 진행 (사용자 승인 필요)"])
    sp = {s: ft2_split(D[s]["merged"], D[s]["miss"], ez) for s in STARTS}
    split, ds, ddd = ez_rule(sp["label"])
    empty = [g_ for g_ in ("with", "without") if not (sp["label"][g_].get("n_attempted") or 0)]
    gate = sp["label"]["without"] if split else a
    chk = {x: (gate.get(x) is not None and (gate[x] <= v if not x.startswith("same") else gate[x] >= v))
           for x, v in FT2_A.items()}
    m, gt = ft2_txt, ez_grp_txt
    n_with = sum(1 for v in ez.values() if v)
    ez_txt = (f"E/Z 형제(fold {k} 목록 {EZ_LIST_NAME}) 있음 {gt(sp['label']['with'])} / 없음 {gt(sp['label']['without'])}; "
              f"G2 TS 출발(b) 있음 {gt(sp['g2']['with'])} / 없음 {gt(sp['g2']['without'])}")
    if empty:
        nm = " · ".join("'형제 있음'" if g_ == "with" else "'형제 없음'" for g_ in empty)
        gate_txt = (f"E/Z {nm} 무리가 비어 있어(fold {k} 목록 {len(ez)}반응 중 형제 있음 {n_with}) 두 무리를 비교하지 않고 "
                    f"관문은 fold {k} 시험 전체 {a.get('n_attempted')}반응으로 판정한다")
    elif split:
        gate_txt = (f"두 무리 차이(같은 TS {fmt(ds)}, |Δd| 중앙값 {fmt(ddd, 4)} Å)가 미리 정한 기준(같은 TS > {EZ_SAME_DIFF} "
                    f"또는 |Δd| > {EZ_DD_DIFF_A} Å)을 넘어 fold {k} FT-2 (a) 관문은 **E/Z 형제 없는 "
                    f"{gate.get('n_attempted')}반응만**으로 판정한다(DEVIATIONS.md에 적음)")
    else:
        gate_txt = (f"두 무리 차이(같은 TS {fmt(ds)}, |Δd| 중앙값 {fmt(ddd, 4)} Å)가 기준(같은 TS > {EZ_SAME_DIFF} 또는 |Δd| > "
                    f"{EZ_DD_DIFF_A} Å) 안이라 관문은 fold {k} 시험 전체 {a.get('n_attempted')}반응으로 판정한다")
    rule_txt = (f"규칙(결과 전 고정, 사용자 11-05): 반응별 기준 = 수렴 · 허수 1개(−50 cm⁻¹ 아래) · 같은 TS · |Δd| ≤ "
                f"{FT2_DD_CUT_A} Å; 1차(analytic:30 = D1 Recalc_Hess 30)에서 실패한 반응은 PES 실패로 보기 전에 같은 출발에서 "
                f"새 MLIP Hessian {RECALC_K} cycle마다(analytic:{RECALC_K}, G2 · G3L식)로 다시 재고, 반응마다 재측정 결과를 "
                f"쓴다(recalc {RECALC_K}에서도 실패해야 실패)")
    nums = dict(base_nums, ez=dict(label=sp["label"], g2=sp["g2"], split=split, diff_same=ds, diff_dd_A=ddd,
                                   empty_groups=empty, thresholds=dict(same_ts_frac=EZ_SAME_DIFF, dd_median_A=EZ_DD_DIFF_A),
                                   n_listed=len(ez), n_with_listed=n_with),
                gate_set="without_sibling" if split else "all", gate=gate, pass_=chk)
    ok = all(chk.values())
    if split:
        P.deviation(S.name, f"fold {k} FT-2 (a) 관문(FT2-p, {b['run']}): {gate_txt}. 명령서 5절 · 사용자 11-05 (5)의 관문은 "
                            f"fold 시험 전체이지만, 사용자 11-05 (2) 규칙(E/Z 형제 무리 차이가 기준을 넘으면 형제 없는 무리만)을 "
                            f"적용했다. 전체 수치(참고): {m(a)}; {ez_txt}. 판정 {'통과' if ok else '실패'}.",
                    key=f"ez:{st.get('epoch')}:{b['run']}")
    head = (f"FT-2 (a) 라벨 TS 출발(관문 fold {k} 시험 {GATE_LIST} — 사용자 11-05 15:5x '바꿔줘'로 fold 0에서 바꿈(FT-2 수치를 "
            f"보기 전, fold {k} 모델이 먼저 학습 끝); recalc {RECALC_K} 재측정 반영): {desc}. 실패 분류 — {cls_txt}. "
            f"{ez_txt}. {gate_txt}. {rule_txt}")
    others = ", ".join(str(x) for x in range(5) if x != k)
    if not ok:
        states = ", ".join(f"fold {x} {fold_model_state(P, bc, x)}" for x in range(5) if x != k)
        return dict(key=f"base {bc} FT-2(a) fold{k} fail", numbers=nums,
                    halt=f"{head} → FT-2 (a) fold {k} 실패({', '.join(x for x, v in chk.items() if not v)}; 판정 무리 "
                         f"{nums['gate_set']}). 1차 모델 상태: {states} (fold 0 = {base_run}, FT-p). fold 1–4 {bc} 모델은 "
                         f"사용자 11-05 13:0x 지시('지금미리 시작하고 FT-A 실패하면 그때가서 수정')로 이 관문을 기다리지 "
                         f"않고 미리 학습한다(손 job ft_spec_{bc}_1–4 → FTpilot_{bc}_1–4, r1_run이 FT_1–4로 재사용). HALT "
                         f"동안 FT-d1 · FT-r1 · FT2-r1 · FT2-f0 · G4는 시작하지 않는다(FT-p의 fold 0 학습은 계속). 다음 "
                         f"단계는 사용자가 정한다(사용자: 'FT-A 실패하면 그때가서 수정'; 앞선 사용자 11-05 순서: G-4 "
                         f"능동학습(fold 0) 뒤 재측정 → 그래도 실패면 MACE-OMOL-0 lr ≤ 1e-4로 fold 0만 재시도 → 둘 다 "
                         f"실패면 STOP; 관문 fold는 사용자 15:5x 결정으로 {k})",
                    options=["`retry` — G-4 능동학습 · 재학습을 lead가 준비한 뒤 다시 잰다 (사용자 순서 1단계)",
                             "`param ...` — MACE-OMOL-0 lr ≤ 1e-4 재시도 (2단계; 사용자 원문은 fold 0)",
                             "`accept` — 실패를 기록하고 진행 (사용자 승인 필요)"])
    f0 = "학습 끝" if train_done(P, base_run) else "FT-p에서 학습 중"
    return dict(key=f"base {bc}", numbers=nums,
                line=f"{head} → 통과 → 기반 MLIP = {bc} (규칙: 같은 TS 비율 → 차이 {TIE_SAME} 안이면 |Δd| → {TIE_DD} Å 안이면 "
                     f"빠른 쪽, '같으면 빠른 쪽'; 속도 출처 {src}). FT_0 = {base_run} 재사용(fold 0 분할, {f0}; G-4는 FT-r1과 "
                     f"FT-p를 기다림). 1차 FT_1–FT_4 = 미리 학습한 FTpilot_{bc}_1–4(사용자 11-05 13:0x, r1_run 재사용; fold "
                     f"{k}는 이 관문 모델); 끝나지도 돌고 있지도 않은 fold만 FT_k를 {bc}로 새로 학습한다. FT2-r1(보고용)은 "
                     f"fold {others}만 재고, fold 0 FT-2를 E/Z(f0_test_ez)로 나눈 수치는 FT2-f0가 fold 0 1차가 끝나면 "
                     f"results/{F0_REPORT}에 적는다.")


def units_ftd1(P, S, st):
    return one("data", ["finetune.py", "data", "FT_1", "FT_2", "FT_3", "FT_4"], env="mlts", part="C", c=4, mem=48)


def units_ftr1(P, S, st):
    b = base_cand(P)
    out = []
    for k in range(1, 5):
        if r1_run(P, k) != f"FT_{k}" or spec_live(P, b, k):   # reuse / wait for the speculative run (FT-spec)
            continue
        out += train_unit(f"FT_{k}", f"FT_{k}", b)
    return out


def ft2r1_runs(P):
    """[(k, run)] measured by FT2-r1 (report only, first pass): the first-round model r1_run(k) of every fold (fold 0 =
    FT_0 = FTpilot_<base>_0), except the gate fold when its model is the FT2-p gate model FTpilot_<base>_4 (FT2-p
    measured it, first pass + recalc 5)."""
    b = base_cand(P)
    return [(k, r1_run(P, k)) for k in range(5) if not (k == GATE_FOLD and b and r1_run(P, k) == gate_run(b))]


def units_ft2r1(P, S, st):
    b = base_cand(P)
    out = []
    for k, run in ft2r1_runs(P):
        if run and train_done(P, run):
            out += geom_units(P, st, run, b, f"f{k}_test", P.c.mlt / "ft2")
    return out


def done_ft2r1(P):
    return all(run and geom_done(P, P.c.mlt / "ft2" / run / s, n_list(P, f"f{k}_test"))
               for k, run in ft2r1_runs(P) for s in STARTS)


def gate_ft2f0(P):
    """FT2-f0 readiness (no jobs): FT_0 = r1_run(0) = FTpilot_<base>_0 has a final FT2-r1 first pass on f0_test for
    both starts (first_pass_final: every reaction recorded or >= 99.5 %, and no first-pass task in squeue) and
    f0_test_ez.csv is readable -> 'pass' (decide_ft2f0 writes the report); else 'wait'."""
    try:
        run = r1_run(P, 0)
        if not run:
            return "wait", "no params base_run yet (FT2-p decides it)", {}
        nl, st = n_list(P, "f0_test"), P.st("FT2-r1")
        todo = [s for s in STARTS if not first_pass_final(P, st, P.c.mlt / "ft2" / run / s, nl, f"{run}/{s}")]
        if todo:
            return "wait", (f"fold 0 first pass of {run} on f0_test not final ({', '.join(todo)}; FT2-r1 measures it once "
                            f"FT-p has trained it)"), {}
        if not ez_map(P, 0):
            return "wait", f"needs $MLT/pipeline/lists/{EZ_LIST_TMPL.format(k=0)}.csv (ez_groups.py run)", {}
        return "pass", "", {}
    except Exception as e:                                                  # noqa: BLE001
        return "wait", f"gate(): {type(e).__name__}: {e}"[:200], {}


def decide_ft2f0(P, S, st):
    """FT2-f0 (report only, never a gate; user 11-05 12:0x (2), kept after the gate moved to fold 4 at 15:5x): the
    fold-0 FT-2 numbers of FT_0 = FTpilot_<base>_0 on f0_test, (a) label start and (b) G2 start, for all reactions and
    split by the fold-0 E/Z sibling list f0_test_ez.csv ('with' / 'without sibling'), with the failure classes and the
    E/Z rule numbers (would the 0.05 / 0.002 A rule split? recorded only), from the per-reaction records of FT2-r1's
    first pass (FT2-r1 does not re-measure; recalc-5 records would be merged if they existed) ->
    results/FT2_f0_report.json (never in dry mode) + one DECISIONS line."""
    run, ez = r1_run(P, 0), ez_map(P, 0)
    ids = list_ids(P, "f0_test")
    per, cls = {}, {}
    for s in STARTS:
        base, merged, nf, nr = ft2p_rows(P, run, s)
        have = {row_id(r) for r in base}
        miss = [x for x in ids if x not in have]
        sp = ft2_split(merged, miss, ez)
        per[s] = dict(all=pooled(merged, len(miss)), first_pass=pooled(base, len(miss)), with_sibling=sp["with"],
                      without_sibling=sp["without"], n_unlisted=sp["n_unlisted"])
        cls[s] = fail_summary(base, merged, nf, nr, len(miss))
    lab, g2 = per["label"], per["g2"]
    split, ds, ddd = ez_rule({"with": lab["with_sibling"], "without": lab["without_sibling"]})
    a = lab["all"]
    chk = {x: (a.get(x) is not None and (a[x] <= v if not x.startswith("same") else a[x] >= v)) for x, v in FT2_A.items()}
    n_with = sum(1 for v in ez.values() if v)
    rule = dict(would_split=split, diff_same=ds, diff_dd_A=ddd,
                thresholds=dict(same_ts_frac=EZ_SAME_DIFF, dd_median_A=EZ_DD_DIFF_A))
    rep = dict(stage="FT2-f0", written=P.now, writer="pipeline.py (FT2-f0, report only)", run=run, fold=0,
               list="f0_test", ez_list=EZ_LIST_TMPL.format(k=0), n_listed=len(ez), n_with_listed=n_with,
               label=lab, g2=g2, fail_classes=cls, ez_rule=rule, ft2_a_thresholds=FT2_A, ft2_a_all_reference=chk,
               gate_fold=GATE_FOLD,
               note=(f"report only: the round-1 FT-2 (a) gate is fold {GATE_FOLD} (FT2-p, user 11-05 15:5x); these are "
                     f"the fold-0 numbers of FT_0 split by the fold-0 E/Z sibling list (user 11-05 12:0x (2)); FT2-r1 is "
                     f"first pass only (analytic:30, no recalc-5 re-measure); missing records count as attempted "
                     f"failures"))
    if P.dry:
        P.planned.append(f"(dry) would write results/{F0_REPORT}")
    else:
        M.write_json(P.c.res / F0_REPORT, rep)
    gt = ez_grp_txt
    line = (f"fold 0 FT-2 보고(FT2-f0, 보고용 · 관문 아님 — 1차 관문은 fold {GATE_FOLD}(FT2-p, 사용자 11-05 15:5x); E/Z 나눔은 "
            f"사용자 12:0x (2)): {run} · f0_test 1차(FT2-r1, recalc 재측정 없음). (a) 라벨 TS 출발 전체 {ft2_txt(a)}; E/Z "
            f"형제(f0_test_ez, {len(ez)}반응 중 {n_with}) 있음 {gt(lab['with_sibling'])} / 없음 {gt(lab['without_sibling'])}. "
            f"(b) G2 TS 출발 전체 {ft2_txt(g2['all'])}; 있음 {gt(g2['with_sibling'])} / 없음 {gt(g2['without_sibling'])}. "
            f"라벨 출발 두 무리 차이(같은 TS {fmt(ds)}, |Δd| 중앙값 {fmt(ddd, 4)} Å)는 12:0x (2) 기준(같은 TS > "
            f"{EZ_SAME_DIFF} 또는 |Δd| > {EZ_DD_DIFF_A} Å)을 {'넘는다' if split else '넘지 않는다'}(기록만, 판정에 안 씀). "
            f"실패 분류 — {fail_txt(cls)}. results/{F0_REPORT}.")
    return dict(key="fold0 report", line=line,
                numbers=dict(run=run, report=F0_REPORT, ez_rule=rule, n_listed=len(ez), n_with_listed=n_with,
                             label={x: lab[x] for x in ("all", "with_sibling", "without_sibling")},
                             g2={x: g2[x] for x in ("all", "with_sibling", "without_sibling")}))


# ---------------------------------------------------------------- FT-abl (round-2 force weight, user 11-05 17:0x (2))
FTABL_RULE = (f"pre-registered (lead 11-05 17:0x, before any ablation number): candidates = the baseline {ABL_BASE_RUN} "
              f"(fw {ABL_FW['baseline']}, patience {ABL_PATIENCE['baseline']}) + every finished ablation "
              f"FTabl_{ABL_CAND}_{GATE_FOLD}_<tag> ({', '.join(f'{t}: ' + ' '.join(ABL_ARGS[t]) for t in ABL_TAGS)}); "
              f"metric = the {ABL_START}-start first-pass same_ts_frac over {GATE_LIST} from recs/*.json (geom_rows + "
              f"pooled; a list reaction without a record counts as an attempted failure, as FT2-p); an ablation must "
              f"beat the baseline by more than {ABL_MIN_GAIN} absolute, else the baseline is kept; among those that do, "
              f"the ones within {ABL_MIN_GAIN} of the best -> the lower median |dd|; within {ABL_DD_TIE_A} A of that too "
              f"-> the smaller fw (closer to the default); an ablation is finished when its FT-2 first pass covers the "
              f"list (all, or >= {GEOM_PARTIAL:.1%} with no live ft2_abl_<tag> task) and dead when its training ended "
              f"without results (sacct) or nothing of it is live and the FT-2 is not finished; both dead -> baseline; "
              f"the chosen arguments go to `finetune.py train` of FT-r2 and FT-fin when the base candidate is "
              f"{ABL_CAND}")


def abl_run(tag):
    """FTabl_<ABL_CAND>_<GATE_FOLD>_<tag> (finetune.parse_run: a fold-k ablation run)."""
    return f"FTabl_{ABL_CAND}_{GATE_FOLD}_{tag}"


def abl_od(P, run):
    return P.c.mlt / "ft2" / run / ABL_START


def abl_live(P, tag):
    """squeue ids of the ablation's training (ft_abl_<tag>) and FT-2 (ft2_abl_<tag>) jobs; a pending task whose
    dependency can never be satisfied (afterok on a failed training) is not live."""
    names = {f"ft_abl_{tag}": "train", f"ft2_abl_{tag}": "ft2"}
    out = dict(train=[], ft2=[])
    for j in getattr(P, "sq", None) or []:
        k = names.get(j.get("name"))
        if k and not (j.get("state") == "PENDING" and str(j.get("reason") or "").startswith("DependencyNeverSatisfied")):
            out[k].append(j["id"])
    return out


def abl_hist(P):
    """sacct rows (since HISTORY_SINCE) of the ablation jobs, one query per pass (cached on P); [] when sacct fails
    (then squeue alone decides: neither done nor live = failed)."""
    if getattr(P, "_abl_hist", None) is None:
        names = [f"{p}_{t}" for t in ABL_TAGS for p in ("ft_abl", "ft2_abl")]
        try:
            rows = P.c.slurm.sacct(names=names, since=HISTORY_SINCE)
        except Exception:                                                   # noqa: BLE001
            rows = None
        P._abl_hist = list(rows or [])
    return P._abl_hist


def abl_cov(P, od):
    """reactions recorded in one ts_geom out_dir: the larger of summary.json's status counts and the record rows."""
    return max(geom_cov(P, od), len({row_id(r) for r in geom_rows(od, P)}))


def abl_status(P, tag):
    """('done' | 'live' | 'failed', why) of one ablation. done: its FT-2 label first pass covers f4_test (every reaction,
    or >= 99.5 % with no live ft2_abl_<tag> task); failed: its training job ended (sacct) without a finished training
    (train_done), or nothing of it is live and the FT-2 is not done ('neither done nor live'); else live."""
    run, nl = abl_run(tag), n_list(P, GATE_LIST)
    cov, live = abl_cov(P, abl_od(P, run)), abl_live(P, tag)
    if nl > 0 and cov >= nl:
        return "done", f"FT-2 {cov}/{nl}"
    if nl > 0 and cov >= math.ceil(GEOM_PARTIAL * nl) and not live["ft2"]:
        return "done", f"FT-2 {cov}/{nl} (>= {GEOM_PARTIAL:.1%}, no live ft2_abl_{tag})"
    ended = {}
    for r in abl_hist(P):
        s = str(r.get("state") or "").split()
        if s and s[0] not in LIVE_STATES:
            ended.setdefault(r.get("name"), []).append(f"{r.get('id')} {s[0]}")
    et = ended.get(f"ft_abl_{tag}") or []
    if et and not live["train"] and not train_done(P, run):
        return "failed", f"training ended without results ({', '.join(et)}; results/ft/{run}_train.json not done)"
    if live["train"] or live["ft2"]:
        return "live", (f"training {','.join(live['train']) or '-'} / FT-2 {','.join(live['ft2']) or '-'} in squeue, "
                        f"FT-2 {cov}/{nl}")
    e2 = ended.get(f"ft2_abl_{tag}") or []
    return "failed", (f"neither done nor live: FT-2 {cov}/{nl} (< {GEOM_PARTIAL:.1%})"
                      + (f", ft2_abl_{tag} ended {', '.join(e2)}" if e2 else ""))


def abl_metrics(P, run, label):
    """label-start first-pass numbers of one candidate over f4_test (geom_rows: recs/*.json; pooled, list reactions
    without a record = attempted failures) + its FT-1 valid force MAE (results/ft/<run>_ft1.json) and training."""
    want = set(list_ids(P, GATE_LIST))
    rows = [r for r in geom_rows(abl_od(P, run), P) if not want or row_id(r) in want]
    have = {row_id(r) for r in rows}
    m = pooled(rows, sum(1 for x in want if x not in have))
    T = P.rjp(P.c.res / "ft" / f"{run}_train.json") or {}
    F = P.rjp(P.c.res / "ft" / f"{run}_ft1.json") or {}
    sets = F.get("sets") or {}
    v = sets.get("valid") or sets.get("valid_subset") or {}
    fm = v.get("force_mae_kcal_A") or {}
    hy, ep = T.get("hyper") or {}, T.get("epochs") or {}
    best = ep.get("best") if isinstance(ep.get("best"), dict) else {}
    out = dict(label=label, run=run, fw=hy.get("fw") if hy.get("fw") is not None else ABL_FW.get(label),
               patience=hy.get("patience") if hy.get("patience") is not None else ABL_PATIENCE.get(label))
    out.update({k: m.get(k) for k in ("same_ts_frac", "dd_median_A", "dd_p90_A", "rmsd_median_A", "nimag1_frac",
                                      "n_attempted", "n_missing_records", "n_optimised")})
    out.update(ft1_valid_force_mae_ts_near=fm.get("ts_near"), ft1_valid_force_mae_all=fm.get("all"),
               train_best_valid_force_mae=best.get("mae_f_kcal_A"), best_epoch=ep.get("best_epoch"),
               n_epochs=ep.get("n"), gpu_h=T.get("gpu_h"))
    return out


def abl_choose(base, abls):
    """the pre-registered rule (FTABL_RULE) -> (choice 'baseline' | tag, reason). base / abls: abl_metrics."""
    bs = base.get("same_ts_frac")
    if bs is None:
        return "baseline", "기준선 수치가 없어 기준선 유지"
    if not abls:
        return "baseline", "끝난 ablation이 없어(모두 실패) 기준선 유지"
    gain = {t: round(m["same_ts_frac"] - bs, 9) for t, m in abls.items() if m.get("same_ts_frac") is not None}
    gtxt = ", ".join(f"{t} {g:+.3f}" for t, g in gain.items())
    elig = [t for t, g in gain.items() if g > ABL_MIN_GAIN]
    if not elig:
        return "baseline", f"기준선보다 같은 TS가 {ABL_MIN_GAIN} 넘게 높은 ablation이 없어 기준선 유지(차이 {gtxt})"
    top = max(abls[t]["same_ts_frac"] for t in elig)
    near = [t for t in elig if round(top - abls[t]["same_ts_frac"], 9) <= ABL_MIN_GAIN]
    dd = {t: abls[t]["dd_median_A"] if abls[t].get("dd_median_A") is not None else 9.9 for t in near}
    dmin = min(dd.values())
    tied = [t for t in near if round(dd[t] - dmin, 9) <= ABL_DD_TIE_A]
    ch = min(tied, key=lambda t: (ABL_FW.get(t, 9e9), t))
    why = f"기준선보다 같은 TS가 {ABL_MIN_GAIN} 넘게 높은 ablation: {', '.join(elig)}(차이 {gtxt})"
    if len(near) == 1:
        why += f" → 같은 TS가 가장 높은 {ch}"
    elif len(tied) == 1:
        why += f" → {', '.join(near)}는 같은 TS 차이 {ABL_MIN_GAIN} 안 → |Δd| 중앙값이 낮은 {ch}"
    else:
        why += (f" → {', '.join(near)}는 같은 TS 차이 {ABL_MIN_GAIN} 안, |Δd| 중앙값도 {ABL_DD_TIE_A} Å 안 → fw가 작은 "
                f"{ch}(기본값에 가까움)")
    return ch, why


def abl_txt(m):
    """one candidate's numbers as decision-line text."""
    return (f"같은 TS {fmt(m.get('same_ts_frac'))} · |Δd| 중앙값 {fmt(m.get('dd_median_A'), 4)} · 90 % "
            f"{fmt(m.get('dd_p90_A'), 4)} Å · RMSD 중앙값 {fmt(m.get('rmsd_median_A'), 4)} Å · 허수 1개 "
            f"{fmt(m.get('nimag1_frac'))} · FT-1 검증 힘 MAE(TS 근처) {fmt(m.get('ft1_valid_force_mae_ts_near'))} · 전체 "
            f"{fmt(m.get('ft1_valid_force_mae_all'))} kcal/mol/Å ({m.get('n_attempted')}반응)")


def gate_ftabl(P):
    """FT-abl readiness (no jobs): the baseline's label first pass on f4_test final (FT2-p: first_pass_final) and
    every ablation finished or dead (abl_status) -> 'pass' (decide_ftabl chooses); else 'wait'. Independent of the
    FT2-p decision and of its HALT (after: lists only)."""
    try:
        nl = n_list(P, GATE_LIST)
        if not first_pass_final(P, P.st("FT2-p"), abl_od(P, ABL_BASE_RUN), nl, f"{ABL_BASE_RUN}/{ABL_START}"):
            return "wait", f"the baseline {ABL_BASE_RUN} {ABL_START} first pass on {GATE_LIST} not final (FT2-p)", {}
        sts = {t: abl_status(P, t) for t in ABL_TAGS}
        nums = {t: dict(state=s, why=w) for t, (s, w) in sts.items()}
        live = [f"{t}: {w}" for t, (s, w) in sts.items() if s == "live"]
        if live:
            return "wait", "ablations not finished — " + "; ".join(live), nums
        return "pass", "", nums
    except Exception as e:                                                  # noqa: BLE001
        return "wait", f"gate(): {type(e).__name__}: {e}"[:200], {}


def decide_ftabl(P, S, st):
    """FT-abl (user 11-05 17:0x (2) "비교 결과로 자동 선택"; no jobs): the force weight (and patience) of the round-2
    retraining FT-r2 and of FT-fin, picked from the lead's fold-4 ablations. PRE-REGISTERED RULE (lead 11-05 17:0x,
    fixed before any ablation number existed; FTABL_RULE):
      candidates — the baseline FTpilot_AIMNet2_4 (fw 0.2, patience 25; the FT2-p gate model) + every FINISHED
        ablation FTabl_AIMNet2_4_<tag>, tags fw1 (--fw 1.0 --patience 50) and fw5 (--fw 5.0 --patience 50). Finished =
        its FT-2 label first pass covers f4_test (all, or >= 99.5 % with no live ft2_abl_<tag>); an ablation whose
        training job ended without results (sacct) or that is neither finished nor live counts as failed and is not a
        candidate (both failed -> baseline, the decision line says so).
      metric — label-start FIRST-PASS same_ts_frac over f4_test from the per-reaction records recs/*.json (geom_rows +
        pooled, like for like: the baseline's recalc-5 re-measure is not used; a list reaction without a record counts
        as an attempted failure, as in FT2-p).
      choice — an ablation must beat the baseline by MORE than ABL_MIN_GAIN = 0.02 absolute, otherwise the baseline is
        kept; among the ablations that do, those within 0.02 of the best -> the lower median |dd|; if that is also
        within ABL_DD_TIE_A = 0.002 A -> the smaller fw (closer to the default).
    The decision line reports every candidate's same_ts_frac, |dd| median / p90, RMSD median, n_imag == 1 fraction and
    the FT-1 valid force MAE (results/ft/<run>_ft1.json). Writes params ft_train_args (e.g. ["--fw", "1.0",
    "--patience", "50"]; [] for the baseline), ft_abl_choice and ft_abl_cand = AIMNet2: FT-r2 and FT-fin append
    ft_train_args to `finetune.py train` (not ft1) when the base candidate is AIMNet2 (ft_args). finetune's run_key
    holds lr / ew / fw / bs, so a changed fw makes a new run dir (no FT_<k>_al run exists yet)."""
    sts = {t: abl_status(P, t) for t in ABL_TAGS}
    base = abl_metrics(P, ABL_BASE_RUN, "baseline")
    abls = {t: abl_metrics(P, abl_run(t), t) for t, (s, _w) in sts.items() if s == "done"}
    ch, why = abl_choose(base, abls)
    args = list(ABL_ARGS[ch]) if ch in ABL_ARGS else []
    P.state.setdefault("params", {}).update(ft_train_args=args, ft_abl_choice=ch, ft_abl_cand=ABL_CAND)
    failed = {t: w for t, (s, w) in sts.items() if s != "done"}
    nums = dict(choice=ch, ft_train_args=args, cand=ABL_CAND, baseline=base, ablations=abls,
                status={t: dict(state=s, why=w) for t, (s, w) in sts.items()}, rule=FTABL_RULE, min_gain=ABL_MIN_GAIN,
                dd_tie_A=ABL_DD_TIE_A, list=GATE_LIST, start=ABL_START)
    parts = [f"기준선 {ABL_BASE_RUN}(fw {base.get('fw')}, patience {base.get('patience')}) {abl_txt(base)}"]
    parts += [f"{t} {abl_run(t)}(fw {abls[t].get('fw')}, patience {abls[t].get('patience')}) {abl_txt(abls[t])}"
              for t in ABL_TAGS if t in abls]
    parts += [f"{t} {abl_run(t)} 실패 — {failed[t]}" for t in ABL_TAGS if t in failed]
    both = "두 ablation 모두 실패 → " if not abls else ""
    use = f"`finetune.py train ... {' '.join(args)}`" if args else "추가 인자 없음(기본 fw 0.2 · patience 25)"
    line = (f"FT-abl 힘 가중치 선택(사용자 11-05 17:0x (2) '비교 결과로 자동 선택'; fold {GATE_FOLD} {GATE_LIST} "
            f"{ABL_START} 출발 1차, 반응 기록 recs/*.json; 규칙은 ablation 수치 전 고정): " + "; ".join(parts)
            + f" → {both}**선택 = {'기준선' if ch == 'baseline' else ch}**: {why}. FT-r2(FT_0–4_al)와 FT-fin(FT_all_s0–2_al · "
              f"FT_LODO_0–4_al) 학습에 {use}(기반 후보가 {ABL_CAND}일 때만; ft1은 인자 없음). 규칙: 후보 = 기준선 + 끝난 "
              f"ablation; 지표 = 같은 TS 비율(기록 없는 반응은 실패); 기준선보다 {ABL_MIN_GAIN} 넘게 높아야 바꾼다; 그런 "
              f"ablation끼리 {ABL_MIN_GAIN} 안이면 |Δd| 중앙값이 낮은 쪽, {ABL_DD_TIE_A} Å 안이면 fw가 작은 쪽.")
    return dict(key=f"choice {ch}", numbers=nums, line=line)


def ft_args(P):
    """the FT-abl choice (params ft_train_args) for `finetune.py train` of FT-r2 / FT-fin when the base candidate is
    the ablated one (params ft_abl_cand = ABL_CAND), else [] (the ablations measured AIMNet2 only)."""
    p = P.state.get("params", {})
    a = p.get("ft_train_args") or []
    return [str(x) for x in a] if a and base_cand(P) == (p.get("ft_abl_cand") or ABL_CAND) else []


def base_set(P):
    """params base and base_run set (decide_ft2p sets them before its pass / fail branch, so a RESOLVE accept of a
    failed gate keeps them; FT-r1 / G4 wait otherwise, e.g. after an accepted decision-rule error)."""
    p = P.state.get("params", {})
    return bool(p.get("base")) and bool(p.get("base_run"))


BASE_NEED = ("params base / base_run (the FT2-p decision; set on its pass and its fail branch)", base_set)


def models_r1(P):
    return ",".join(r1_run(P, k) or "" for k in range(5))


def g4_est(P, step):
    e = (P.rj("G4_plan.json") or {}).get("est") or {}
    e = e.get(step) if isinstance(e.get(step), dict) else e
    return e or {}


G4CMD = ["g4_al.py"]                 # G-4 lives in g4_al.py (gen_dft.py belongs to the D0 workflow)


def units_g4(P, S, st):
    u = one("plan", G4CMD + ["g4_plan", "--models", models_r1(P)], env="reactot", part="C", c=4, mem=16,
            done=rep_complete("G4_plan.json"))
    if not (P.rj("G4_plan.json") or {}).get("complete"):
        return u
    eo, er = g4_est(P, "opt"), g4_est(P, "run")
    no = fixed_n(P, st, "opt", lambda: n_tasks(eo.get("units_total", 1500), eo.get("unit_wall_h", 0.02),
                                               eo.get("workers", 14), 4))
    nr = fixed_n(P, st, "run", lambda: n_tasks(er.get("units_total", 1500), er.get("unit_wall_h", 0.1),
                                               er.get("workers", 64), 4))
    opt = arr("opt", G4CMD + ["g4_opt", "@I", str(no), "--models", models_r1(P)], no, env="mlts", part="GA", c=16,
              mem=64, gpu=1, deps=["plan"])
    run = arr("run", G4CMD + ["run", "G4", "@I", str(nr)], nr, env="reactot", part="C2", c=64, mem=240, deps=keys(opt))
    return u + opt + run + one("collect", G4CMD + ["collect", "--g4"], env="reactot", part="C", c=4, mem=32,
                               deps=keys(run), sink=True)


def done_g4(P):
    return (P.rj("G4_report.json") or {}).get("complete") is True and (P.c.mlt / "dft" / "gen_G4.extxyz").is_file()


def core_h_of(J):
    c = (J or {}).get("core_h_measured")
    if isinstance(c, dict):
        return float(c.get("total") or 0.0)
    return float((J or {}).get("core_h") or 0.0)


def guard_g4(P):
    """$MLT/dft/STOP or G4_report stop_file -> HALT; and, before any G-4 DFT runs, the plan's predicted cost:
    measured D1 + D0 + predicted G-4 > the G cap (gen_dft.BUDGET_CORE_H, user decisions; spec 0-5 5,000) -> HALT."""
    g = guard_dft_stop("G4_report.json")(P)
    if g:
        return g
    J = P.rj("G4_plan.json") or {}
    pred = J.get("predicted_core_h")
    pred = pred.get("total") if isinstance(pred, dict) else pred
    bud = g_budget(P)
    if J.get("complete") and pred is not None and not (P.rj("G4_report.json") or {}).get("complete"):
        meas = core_h_of(P.rj("G_report.json")) + core_h_of(P.rj("G_report_d0.json"))
        if meas + float(pred) > bud:
            return (f"G-4 계획 예상: G 실측 {meas:,.0f} + G-4 예상 {float(pred):,.0f} = {meas + float(pred):,.0f} core-h > "
                    f"{bud:,.0f} (gen_dft.BUDGET_CORE_H; G-4 DFT 실행 전; 명령서 0-5 STOP)")
    return None


def decide_g4(P, S, st):
    parts = {n: core_h_of(P.rj(n)) for n in ("G_report.json", "G_report_d0.json", "G4_report.json")}
    tot = sum(parts.values())
    bud = g_budget(P)
    if tot > bud:
        return dict(key="over budget", numbers=dict(parts=parts, total=tot, budget=bud),
                    halt=f"G 전체 실측 {tot:,.0f} core-h > {bud:,.0f} (gen_dft.BUDGET_CORE_H; G-4 뒤; 명령서 0-5 STOP)")
    return dict(key="ok", line=f"G-4 능동학습 끝: G 전체 실측 {tot:,.0f} core-h ≤ {bud:,.0f} "
                               f"({', '.join(f'{k} {v:,.0f}' for k, v in parts.items())}).")


ALL_RUNS = [f"FT_{k}" for k in range(5)] + [f"FT_all_s{i}" for i in range(3)] + [f"FT_LODO_{d}" for d in range(5)]
FIN_RUNS = [f"FT_all_s{i}" for i in range(3)] + [f"FT_LODO_{d}" for d in range(5)]


def units_ftd2(P, S, st):
    return one("data", ["finetune.py", "data", "--all", "--with-al"], env="mlts", part="C", c=4, mem=64)


def done_ftd2(P):
    return all(data_ok(P, r + "_al", newer_than=("G4_report.json",)) for r in ALL_RUNS)


def units_ftr2(P, S, st):
    """FT_<k>_al (k = 0..4) with the base candidate, --with-al + the FT-abl arguments (ft_args) on train."""
    b, a = base_cand(P), ft_args(P)
    out = []
    for k in range(5):
        out += train_unit(f"FT_{k}_al", f"FT_{k}", b, al=True, args=a)
    return out


def abl_use_txt(P):
    """the FT-abl choice as start-line text (FT-r2 / FT-fin on_start)."""
    a, ch = ft_args(P), P.state.get("params", {}).get("ft_abl_choice")
    return (f"FT-abl 선택 {ch} → `finetune.py train ... --with-al {' '.join(a)}`(ft1은 인자 없음)" if a else
            f"FT-abl 선택 {ch or '없음'} → 추가 인자 없음(기본 fw · patience; 기반 후보 {base_cand(P)})")


def start_ftr2(P, S, st):
    return f"FT_0–4_al 학습 인자: {abl_use_txt(P)}. 사용자 11-05 17:0x (2) '비교 결과로 자동 선택'."


FT2R2_RUNS = [(f"FT_{k}_al", f"f{k}_test") for k in range(5)]
EZ_FOLDS_ARG = ",".join(str(k) for k in range(5))


def units_ft2r2(P, S, st):
    """FT2-r2: the E/Z lists f<k>_test_ez.csv (k = 0..4, user 11-05 (2)) by one `ez_groups.py run --folds 0,1,2,3,4`
    CPU job while one is missing (existing lists are kept); per trained FT_<k>_al the ts_geom first pass on fold k (both
    starts) and, once a start's first pass is final, the recalc-5 re-measure of its failed reactions (user 11-05 (3),
    recalc_units as in FT2-p)."""
    b = base_cand(P)
    out = [] if ez_lists_done(P) else one("ez", ["ez_groups.py", "run", "--folds", EZ_FOLDS_ARG], env="mlts", part="C",
                                          c=4, mem=16, done=ez_lists_done)
    for run, ln in FT2R2_RUNS:
        if train_done(P, run):
            out += geom_units(P, st, run, b, ln, P.c.mlt / "ft2")
            out += recalc_units(P, st, run, b, ln, P.c.mlt / "ft2")
    return out


def done_ft2r2(P):
    """the five E/Z lists present, the first pass complete for both starts of every FT_<k>_al, and every failed
    reaction re-measured (recalc 5)."""
    return ez_lists_done(P) and all(geom_done(P, od, n_list(P, ln)) and not recalc_state(od, P)[2]
                                    for run, ln in FT2R2_RUNS for od in (P.c.mlt / "ft2" / run / s for s in STARTS))


def partial_ft2r2(P):
    """Stage.partial of FT2-r2: the E/Z lists present, then recalc_partial over FT_<k>_al x f<k>_test."""
    if not ez_lists_done(P):
        return False, f"E/Z sibling lists {EZ_LIST_TMPL}.csv missing (ez_groups.py run --folds {EZ_FOLDS_ARG})"
    return recalc_partial(P, FT2R2_RUNS)


def prog_ft2r2(P):
    return prog_recalc(P, [r for r, _ln in FT2R2_RUNS])


def read_geom_csv(path):
    import csv
    rows = []
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        pass
    return rows


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def pooled(rows, n_missing=0):
    """ts_geom.table over pooled records: fractions over attempted, |dd| / RMSD over optimised. n_missing = list
    reactions without any record (pipeline errors, Stage.partial): counted as attempted failures (not same TS,
    n_imag != 1), never in the |dd| / RMSD percentiles."""
    att = [r for r in rows if r.get("status") not in ("no_start", "no_label", "not_covered")]
    dd = [v for v in (_f(r.get("dd_max")) for r in att) if v is not None]
    rm = [_f(r.get("rmsd_heavy")) for r in att if _f(r.get("dd_max")) is not None]
    rm = [v for v in rm if v is not None]
    n = len(att) + int(n_missing)
    return dict(n_attempted=n, n_missing_records=int(n_missing), n_optimised=len(dd), dd_median_A=pct(dd, 50),
                dd_p90_A=pct(dd, 90), rmsd_median_A=pct(rm, 50),
                same_ts_frac=(sum(str(r.get("same_ts")) == "True" for r in att) / n) if n else None,
                nimag1_frac=(sum(_f(r.get("n_imag")) == 1.0 for r in att) / n) if n else None)


FT_GEOM_COLS = ("run", "fold", "start", "rxn_id", "dipole", "g1_class", "status", "converged", "dd_max", "rmsd_heavy",
                "rmsd_direct", "same_ts", "n_imag", "n_cycles", "wall_s", "fail_class", "remeasured",
                "first_pass_fail_class")


def ft2_rows(P, run, start):
    return read_geom_csv(P.c.mlt / "ft2" / run / start / "geom.csv")


def write_ft_outputs(P, gate=None):
    """spec 5 outputs: results/FT_report.json (FT-1 per run from results/ft/*_ft1.json, train cost, FT-2 summaries of
    every round and the pooled round-2 gate) and results/FT_geom.csv (one row per FT_<k>_al test reaction and start:
    the merged record the gate uses — the recalc-5 re-measure when the first pass failed and one exists — with |Δd|,
    RMSD, same TS, imaginary-mode count, fail_class, remeasured, first_pass_fail_class). Never in dry mode."""
    if P.dry:
        P.planned.append("(dry) would write results/FT_report.json + results/FT_geom.csv")
        return
    import csv
    runs = {}
    for f in sorted((P.c.res / "ft").glob("*_train.json")):
        run = f.name[:-len("_train.json")]
        T = P.rjp(f) or {}
        F1 = P.rjp(P.c.res / "ft" / f"{run}_ft1.json") or {}
        runs[run] = dict(cand=T.get("cand"), status=T.get("status"), gpu_h=T.get("gpu_h"), epochs=T.get("epochs"),
                         data_fingerprint=T.get("data_fingerprint"),
                         ft1={s: {k: v.get(k) for k in ("force_mae_kcal_A", "rel_e", "n") if k in v}
                              for s, v in (F1.get("sets") or {}).items()})
    ft2 = {}
    for root in ("ft2", "x"):
        base = P.c.mlt / root
        for sj in sorted(base.glob("*/*/summary.json")) if base.is_dir() else []:
            run, start = sj.parent.parent.name, sj.parent.name
            S = P.rjp(sj) or {}
            ft2.setdefault(root, {}).setdefault(run, {})[start] = dict(all=S.get("all"), by_dipole=S.get("by_dipole"),
                                                                      by_g1_class=S.get("by_g1_class"))
    rep = dict(spec_section="5 Phase FT", written=P.now, writer="pipeline.py (FT2-r2 decision)",
               thresholds_ft2_a=FT2_A, nimag1_min=NIMAG1_MIN, runs=runs, ft2=ft2, gate_round2=gate,
               note="FT-1 is recorded only; the FT-2 (a) gate is the pooled round-2 label-start numbers (FT_<k>_al, "
                    "fold k test reactions) on the merged records (recalc-5 re-measure of the first-pass failures) and, "
                    "by the user's 11-05 E/Z rule, on gate_round2.gate_set; per-reaction rows in results/FT_geom.csv")
    M.write_json(P.c.res / "FT_report.json", rep)
    out = []
    for k in range(5):
        for s in STARTS:
            base, merged, _nf, _nr = ft2p_rows(P, f"FT_{k}_al", s)
            b0 = {row_id(r): r for r in base}
            for r0 in merged:
                f0 = b0.get(row_id(r0))
                r = dict(r0, run=f"FT_{k}_al", fold=k, start=s, fail_class=fail_class(r0) or "",
                         remeasured=f0 is not None and r0 is not f0,
                         first_pass_fail_class=(fail_class(f0) or "") if f0 is not None else "")
                out.append({c: r.get(c, "") for c in FT_GEOM_COLS})

    def w(t):
        with open(t, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(FT_GEOM_COLS))
            wr.writeheader()
            wr.writerows(out)
    M._atomic(P.c.res / "FT_geom.csv", w)


FT2R2_RULE = (f"FT2-r2 = the FT2-p rules (user 11-05 (2) / (3)) on FT_<k>_al x the fold-k tests pooled over k = 0..4: "
              f"per-reaction criterion as FT2-p (|dd| cut {FT2_DD_CUT_A} A); every first-pass failure re-measured with "
              f"analytic:{RECALC_K} (<start>{RECALC_SUFFIX}/), the gate on the merged records; E/Z lists f<k>_test_ez.csv "
              f"(training side of fold k = fold_T1 != k, incl. fold -1 anchors and the fold-k validation cores); if the "
              f"pooled label-start 'with' and 'without sibling' groups differ by > {EZ_SAME_DIFF} same-TS fraction or > "
              f"{EZ_DD_DIFF_A} A median |dd|, the gate is judged on 'without sibling' only (DEVIATIONS), else on all")


def decide_ft2r2(P, S, st):
    """FT2-r2 (round 2): the spec-5 FT-2 (a) gate pooled over FT_<k>_al on the fold-k tests (k = 0..4), with the user's
    11-05 rules (2) and (3) applied exactly as in FT2-p (decide_ft2p), fixed before any round-2 number:
      re-measure — the first-pass failures of every (run, start) are run again with recalc 5 (units_ft2r2 ->
        recalc_units, <start>_recalc5/); the gate uses the merged per-reaction records (ft2p_rows: the re-measured
        record when it exists, so a reaction fails only if it also fails with recalc 5); the failure classes (i)
        not_converged (ii) n_imag (iii) other_saddle are reported per start, pooled over the folds and per fold, before
        and after the re-measure.
      E/Z siblings — $MLT/pipeline/lists/f<k>_test_ez.csv (ez_groups.py; training side of fold k = fold_T1 != k incl.
        the -1 anchor-only cores and the fold-k validation cores); (a) label-start and (b) G2-start numbers for 'with'
        / 'without sibling', pooled over the five folds (each fold with its own list). THRESHOLD as FT2-p: if the
        label-start groups differ by more than EZ_SAME_DIFF = 0.05 same-TS fraction OR EZ_DD_DIFF_A = 0.002 A median
        |dd|, the gate is judged on 'without sibling' only and a DEVIATIONS.md line is appended (Pass.deviation);
        otherwise on all pooled reactions.
    HALT on a failed gate (spec 5 / 14 STOP). results/FT_report.json (gate_round2 = these numbers) + FT_geom.csv (the
    merged records) are written at the decision. FT2-r1 stays first pass only (report for G-4, not a gate)."""
    ez = {k: ez_map(P, k) for k in range(5)}
    D = {}
    for s in STARTS:
        base_all, merged_all, parts, by_fold, n_fail, n_re, n_miss = [], [], [], {}, 0, 0, 0
        for k, (run, ln) in enumerate(FT2R2_RUNS):
            base, merged, nf, nr = ft2p_rows(P, run, s)
            have = {row_id(r) for r in base}
            miss = [x for x in list_ids(P, ln) if x not in have]
            base_all += base
            merged_all += merged
            parts.append((merged, miss, ez[k]))
            n_fail, n_re, n_miss = n_fail + nf, n_re + nr, n_miss + len(miss)
            by_fold[str(k)] = fail_summary(base, merged, nf, nr, len(miss))
        D[s] = dict(all=pooled(merged_all, n_miss), first=pooled(base_all, n_miss), parts=parts,
                    cls=dict(fail_summary(base_all, merged_all, n_fail, n_re, n_miss), by_fold=by_fold))
    a, g2 = D["label"]["all"], D["g2"]["all"]
    cls = {s: D[s]["cls"] for s in STARTS}
    base_nums = dict(pooled=dict(label=a, g2=g2), label=a, g2=g2, label_first_pass=D["label"]["first"],
                     g2_first_pass=D["g2"]["first"], fail_classes=cls, thresholds=FT2_A, dd_cut_A=FT2_DD_CUT_A,
                     recalc=f"analytic:{RECALC_K}", rule=FT2R2_RULE)
    missing = [k for k in range(5) if not ez[k]]
    if missing:
        nums = dict(base_nums, pass_all=None)
        write_ft_outputs(P, gate=nums)
        return dict(key="FT-2 E/Z lists missing", numbers=nums,
                    halt=f"FT2-r2 판정에 필요한 E/Z 형제 목록 "
                         f"{', '.join(EZ_LIST_TMPL.format(k=k) + '.csv' for k in missing)}이 $MLT/pipeline/lists에 없다"
                         f"(사용자 11-05 (2) 규칙)",
                    options=[f"`retry` — `ez_groups.py run --folds {EZ_FOLDS_ARG}`(MLTENV=mlts, CPU)으로 목록을 만든 뒤 "
                             f"다시 판정", "`accept` — E/Z 나눔 없이 진행 (사용자 승인 필요)"])
    sp = {s: ft2_split_parts(D[s]["parts"]) for s in STARTS}
    split, ds, ddd = ez_rule(sp["label"])
    gate = sp["label"]["without"] if split else a
    chk = {k: (gate.get(k) is not None and (gate[k] <= v if not k.startswith("same") else gate[k] >= v))
           for k, v in FT2_A.items()}
    ok = all(chk.values())
    nums = dict(base_nums, ez=dict(label=sp["label"], g2=sp["g2"], split=split, diff_same=ds, diff_dd_A=ddd,
                                   thresholds=dict(same_ts_frac=EZ_SAME_DIFF, dd_median_A=EZ_DD_DIFF_A),
                                   n_listed=sum(len(v) for v in ez.values())),
                gate_set="without_sibling" if split else "all", gate=gate, pass_=chk, pass_all=ok)
    write_ft_outputs(P, gate=nums)
    m = ft2_txt
    ez_txt = (f"E/Z 형제 있음 {m(sp['label']['with'])} / 없음 {m(sp['label']['without'])}; G2 TS 출발(b) 있음 "
              f"{m(sp['g2']['with'])} / 없음 {m(sp['g2']['without'])}")
    gate_txt = (f"두 무리 차이(같은 TS {fmt(ds)}, |Δd| 중앙값 {fmt(ddd, 4)} Å)가 미리 정한 기준(같은 TS > {EZ_SAME_DIFF} 또는 "
                f"|Δd| > {EZ_DD_DIFF_A} Å)을 넘어 2차 FT-2 (a) 관문은 **E/Z 형제 없는 {gate.get('n_attempted')}반응만**(5 fold "
                f"합침)으로 판정한다(DEVIATIONS.md에 적음)" if split else
                f"두 무리 차이(같은 TS {fmt(ds)}, |Δd| 중앙값 {fmt(ddd, 4)} Å)가 기준(같은 TS > {EZ_SAME_DIFF} 또는 |Δd| > "
                f"{EZ_DD_DIFF_A} Å) 안이라 관문은 5 fold 시험 전체 {a.get('n_attempted')}반응으로 판정한다")
    txt = (f"FT-2 (2차, 5 fold 합침, recalc {RECALC_K} 재측정 반영) 라벨 TS 출발 {m(a)}, 허수 1개 "
           f"{fmt(a.get('nimag1_frac'))}; G2 TS 출발 {m(g2)}. 실패 분류(5 fold 합침) — {fail_txt(cls)}. {ez_txt}. "
           f"{gate_txt}. 규칙: FT2-p와 같음(사용자 11-05 (2) · (3), 2차 결과 전 고정)")
    if split:
        P.deviation(S.name, f"2차 FT-2 (a) 관문(FT2-r2, FT_0–4_al 5 fold 합침): {gate_txt}. 명령서 5절의 관문은 fold 시험 "
                            f"전체이지만, 사용자 11-05 (2) 규칙(E/Z 형제 무리 차이가 기준을 넘으면 형제 없는 무리만)을 FT2-p와 "
                            f"같이 적용했다. 전체 수치(참고): {m(a)}; {ez_txt}. 판정 {'통과' if ok else '실패'}.",
                    key=f"ez:{st.get('epoch')}:FT2-r2")
    if not ok:
        return dict(key="FT-2 fail", numbers=nums,
                    halt=f"{txt} → 2차 미세조정 뒤에도 FT-2 (a) 실패({', '.join(k for k, v in chk.items() if not v)}; 판정 "
                         f"무리 {nums['gate_set']}) — 명령서 5절 · 14절 STOP(설계가 틀렸다는 뜻, 보고)",
                    options=["`accept` — 기준 미달을 기록하고 A 단계로 진행", "`retry` — 원인을 고친 뒤 FT2-r2를 다시 잰다",
                             "`skip` — FT2-r2 판정을 건너뛴다"])
    note = ("" if (gate.get("nimag1_frac") or 0) >= NIMAG1_MIN else
            f" 허수 진동수 1개 비율 {fmt(gate.get('nimag1_frac'))} < 0.95(기록만).")
    return dict(key="FT-2 pass", numbers=nums, line=f"{txt} → FT-2 (a) 통과.{note}")


def units_ftfin(P, S, st):
    """FT_all_s0-2 + FT_LODO_0-4, --with-al + the FT-abl arguments (ft_args, the same setting as FT-r2) on train."""
    b, a = base_cand(P), ft_args(P)
    out = []
    for r in FIN_RUNS:
        out += train_unit(r + "_al", r, b, al=True, args=a)
    return out


def start_ftfin(P, S, st):
    return "FT_all_s0–2와 FT_LODO_0–4를 모두 --with-al(G-4 능동학습 자료 포함)로 학습한다. 추천안: LODO는 최종 모델과 같은 자료"\
           "(능동학습 포함)를 쓰고 그 dipole 구조만 뺀다 — X(LODO) 결과가 배포 모델의 일반화를 그대로 대신하게 하기 위함. "\
           f"학습 인자는 FT-r2와 같다: {abl_use_txt(P)}."


def units_xrun(P, S, st):
    b = base_cand(P)
    out = []
    for d in range(5):
        run = f"FT_LODO_{d}_al"
        if train_done(P, run):
            out += geom_units(P, st, run, b, f"dip{d}", P.c.mlt / "x")
    return out


def done_xrun(P):
    return all(geom_done(P, P.c.mlt / "x" / f"FT_LODO_{d}_al" / s, n_list(P, f"dip{d}")) for d in range(5) for s in STARTS)


# ---------------------------------------------------------------- C-1 with FT_0_al, A
def done_c1ft(P):
    p = P.c.res / "C1_tests.json"
    m = ((P.rjp(p) or {}).get("models") or {}).get("FT_0_al") or {}
    tm = P.mt(P.c.mlt / "models" / "FT_0_al.json")
    return m.get("pass_C1") is not None and m.get("pass_C4") is not None and tm is not None and P.mt(p) >= tm


def decide_c1ft(P, S, st):
    m = ((P.rj("C1_tests.json") or {}).get("models") or {}).get("FT_0_al") or {}
    if m.get("pass_C1") and m.get("pass_C4"):
        return dict(key="pass", line="C-1 · C-4 (FT_0_al): 다리 · 최적화기 · Hessian · 단위 검사 모두 통과.")
    return dict(key="fail", numbers=dict(passes=m.get("passes"), pass_C1=m.get("pass_C1"), pass_C4=m.get("pass_C4")),
                halt="C-1/C-4 실패 (FT_0_al) — 버그일 수 있어 고치기 전에 보고한다(명령서 머리말)")


def decide_a2(P, S, st):
    J = P.rj("A2_smoke.json") or {}
    if J.get("pass"):
        return dict(key="pass", line="A-2 smoke: autodE-MLTS 입력에 범함수 · 기저 · CPCM · SMD 단어 없음, ExtOpt 있음(C-3 문구) → 통과.")
    return dict(key="fail", numbers={k: J.get(k) for k in list(J)[:12]}, halt="A-2 smoke 실패(C-3 문구 검사 또는 실행)")


def a3_per_rxn_disk(P):
    v = (P.rj("A3_report.json") or {}).get("disk_gb_per_rxn")
    return float(v) if v is not None else DISK_GB_PER_RXN_DEFAULT


A_GPU_C, A_GPU_MEM = A_WORKERS * A_CORES + 4, 64          # 6 autodE workers x 2 cores + server / parent


def units_a3(P, S, st):
    na = fixed_n(P, st, "A", lambda: 2)
    npp = fixed_n(P, st, "App", lambda: 1)
    A = arr("A", ["mlts_autode.py", "run", "a3", "@I", str(na), "--list", str(P.c.list_path("a3"))], na, env="mlts",
            part="GA", c=A_GPU_C, mem=A_GPU_MEM, gpu=1, disk_gb=600 / na * DISK_GB_PER_RXN_DEFAULT)
    App = arr("App", ["mlts_autode.py", "refine_dft", "a3", "@I", str(npp)], npp, env="reactot", part="C2", c=64,
              mem=240, deps=keys(A), disk_gb=300 / npp * 0.01)
    return A + App + one("report", ["mlts_autode.py", "a3_report"], env="reactot", part="C", c=4, mem=16,
                         deps=keys(App), sink=True)


def prog_a3(P):
    """A-3 progress: the published A-2 folders of the three arms ($MLT/a3/<A|Ap|App>/ade/jobs, mlts_autode A3_ROOT)."""
    return sum(count_dir(P.c.mlt / "a3" / a / "ade" / "jobs") for a in ("A", "Ap", "App"))


def prog_a4(P):
    """A-4 progress: the main-engine A run writes $MLT/ade/jobs, or $MLT/ade_pre/jobs when A'' is the engine (its DFT
    refine then fills $MLT/ade/jobs)."""
    n = count_dir(P.c.mlt / "ade" / "jobs")
    if (P.state.get("params") or {}).get("engine") == "App":
        n += count_dir(P.c.mlt / "ade_pre" / "jobs")
    return n


ARM_NAME = {"A": "A (autodE-MLTS)", "Ap": "A′ (G2 TS → MLTS OptTS)", "App": "A″ (A → 라벨 수준 OptTS)"}


def arm_in_budget(arm, v):
    """spec 0-5: A / A′ <= 0.6 core-h + 3 GPU-min per reaction (median); A″ (A + label-level OptTS) <= the deployment
    target 1 core-h + 3 GPU-min. A missing cost is not in budget. A′'s core_h_median must include generating the G2 TS
    (contract), since a new reaction has none."""
    ch, gm = A3_BUDGET.get(arm, (0.6, 3.0))
    c, g = v.get("core_h_median"), v.get("gpu_min_median")
    try:
        return c is not None and g is not None and float(c) <= ch and float(g) <= gm
    except (TypeError, ValueError):
        return False


def write_arms(P):
    """$MLT/pipeline/arms.json: the A-3 engine and the optional A″ arm, read by train_k.py / eval_v2.py (contract)."""
    p = P.state.get("params") or {}
    J = dict(engine=p.get("engine"), extra_arm_App=bool(p.get("extra_arm_App")),
             app_root=str(P.c.mlt / "app") if p.get("extra_arm_App") else None, app_usable=p.get("app_usable"),
             written=P.now, writer="pipeline.py")
    if P.dry:
        P.planned.append(f"(dry) would write $MLT/pipeline/arms.json {J}")
        return
    M.write_json(P.c.pdir / "arms.json", J)


def decide_a3(P, S, st):
    R = P.rj("A3_report.json") or {}
    arms = R.get("arms") or {}
    nums = dict(arms={k: {kk: (v or {}).get(kk) for kk in ("same_ts_frac", "core_h_median", "gpu_min_median", "n")}
                      for k, v in arms.items()}, budget=A3_BUDGET)
    rows = [dict(arm=k, same=v.get("same_ts_frac"), cost=v.get("core_h_median"), gpu=v.get("gpu_min_median"),
                 ok=arm_in_budget(k, v)) for k, v in arms.items() if v and v.get("same_ts_frac") is not None]
    if not rows:
        return dict(key="report incomplete", numbers=nums, halt="A3_report.json에 arms{A, Ap, App}.same_ts_frac가 없다")

    def cmp(a, b):
        if abs((a["same"] or 0) - (b["same"] or 0)) > TIE_SAME:
            return -1 if (a["same"] or 0) > (b["same"] or 0) else 1
        ca, cb = (a["cost"] if a["cost"] is not None else 1e9), (b["cost"] if b["cost"] is not None else 1e9)
        return -1 if ca < cb else (1 if ca > cb else 0)
    from functools import cmp_to_key
    order = sorted(rows, key=cmp_to_key(cmp))
    desc = ", ".join(f"{r['arm']} {fmt(r['same'])} / {fmt(r['cost'], 2)} core-h / {fmt(r['gpu'], 1)} GPU-분"
                     f"{'' if r['ok'] else ' (예산 초과)'}" for r in order)
    within = [r for r in order if r["ok"]]
    if not within:
        return dict(key="all over budget", numbers=nums,
                    halt=f"A-3 비용: 예산 안의 엔진이 없다({desc}; A · A′ ≤ 0.6 core-h + 3 GPU-분, A″ ≤ 1 core-h + 3 GPU-분, "
                         f"명령서 0-5 · 14절 STOP)")
    best = within[0]
    if (best["same"] or 0) < A3_RULE["stop_below"]:
        return dict(key="same-TS < 0.70", numbers=nums,
                    halt=f"A-3: 예산 안의 가장 좋은 엔진 {best['arm']}의 같은 TS 비율 {fmt(best['same'])} < 0.70 ({desc}; "
                         f"명령서 6절 STOP: 형태 문제를 다른 방법으로 풀어야 함)")
    P.state.setdefault("params", {})["engine"] = best["arm"]
    app = arms.get("App") or {}
    extra = best["arm"] != "App" and app.get("same_ts_frac") is not None and arm_in_budget("App", app) and \
        float(app["core_h_median"]) <= A3_RULE["app_core_h"]
    P.state["params"]["extra_arm_App"] = bool(extra)
    write_arms(P)
    skipped = [r["arm"] for r in order if not r["ok"] and order.index(r) < order.index(best)]
    sk = (f" 같은 TS가 더 높은 {', '.join(skipped)}는 예산 초과라 고르지 않았다(명령서 0-5)." if skipped else "")
    lim = (f" 같은 TS 비율 {best['same']:.3f}는 0.70–0.85 사이: 진행하고 형태 불일치를 주된 한계로 기록한다(명령서 6절)."
           if best["same"] < A3_RULE["limit_below"] else "")
    ex = (" A″ 비용 ≤ 1 core-h → K · E에서 별도 arm으로 평가한다(A4-App: 9,266반응 refine_dft → $MLT/app, H-App: 그 입력, "
          "$MLT/pipeline/arms.json)." if extra else "")
    return dict(key=f"engine {best['arm']}", numbers=nums,
                line=f"A-3 엔진 선택(예산 안에서 같은 TS 비율 → 0.025 안이면 싼 쪽): {desc} → A-4 엔진 = "
                     f"{ARM_NAME.get(best['arm'], best['arm'])}.{sk}{lim}{ex}")


def h_disk_per_rxn(P):
    cache = P.state.setdefault("params", {})
    if "h_disk_gb_per_rxn" in cache:
        return cache["h_disk_gb_per_rxn"]
    tot, n = 0, count_dir(P.c.mlt / "h1" / "final")
    for sub in ("final", "hf"):
        for root, _, files in os.walk(P.c.mlt / "h1" / sub):
            for f in files:
                try:
                    tot += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    v = (tot / 1e9 / n) if n and tot else H_DISK_GB_PER_RXN_DEFAULT
    cache["h_disk_gb_per_rxn"] = round(v, 5)
    return cache["h_disk_gb_per_rxn"]


def gate_a4(P):
    q = P.state.get("quota") or {}
    if q.get("stale") or q.get("gb") is None:
        return "wait", "QUOTA_FREE_GB가 없거나 24 h 넘음 (lead가 mmlsquota 값을 적어야 판정)", {}
    per, h = a3_per_rxn_disk(P), h_disk_per_rxn(P)
    extra = bool((P.state.get("params") or {}).get("extra_arm_App"))
    app = app_disk_per_rxn(P) if extra else 0.0
    need = (per + h + (app + h if extra else 0.0)) * N_D1
    free = float(q.get("est_free") if q.get("est_free") is not None else q["gb"])
    nums = dict(free_gb_est=round(free, 1), a3_gb_per_rxn=per, h_gb_per_rxn=h, app_gb_per_rxn=app, extra_arm_App=extra,
                need_gb=round(need, 1), margin_gb=30)
    if free - need >= QUOTA_MIN_GB:
        return "pass", "", nums
    return "fail", (f"디스크: 예상 여유 {free:,.0f} GB − 필요 {need:,.0f} GB < 30 GB (A-4 9,266반응 + H"
                    f"{' + A″ arm' if extra else ''})"), nums


def app_disk_per_rxn(P):
    v = (((P.rj("A3_report.json") or {}).get("arms") or {}).get("App") or {}).get("disk_gb_per_rxn")
    return float(v) if v is not None else APP_DISK_GB_PER_RXN_DEFAULT


def fold_sizes(P):
    return {k: n_list(P, f"f{k}_test") for k in range(5)}


def units_a4(P, S, st):
    eng = P.state.get("params", {}).get("engine") or "A"
    est = (P.rj("A3_report.json") or {}).get("est") or {}
    per = a3_per_rxn_disk(P)
    u, sink_deps = [], []
    folds = list(range(5)) + ([-1] if (lists_info(P).get("n_fold_minus1") or 0) > 0 else [])
    for k in folds:
        nf = n_list(P, f"f{k}_test") if k >= 0 else int(lists_info(P).get("n_fold_minus1") or 0)
        n = fixed_n(P, st, f"f{k}", lambda: n_tasks(nf, est.get("unit_wall_h", 0.3), est.get("workers", A_WORKERS), 6))
        model = f"FT_{k}_al" if k >= 0 else "FT_0_al"
        r = arr(f"f{k}", ["mlts_autode.py", "run", f"d1:{k}", "@I", str(n), "--model", model, "--engine", eng], n,
                env="mlts", part="GA", c=A_GPU_C, mem=A_GPU_MEM, gpu=1, disk_gb=nf / n * per)
        u += r
        sink_deps += keys(r)
        if eng == "App":
            n2 = fixed_n(P, st, f"r{k}", lambda: max(1, math.ceil(n / 2)))
            rr = arr(f"r{k}", ["mlts_autode.py", "refine_dft", f"d1:{k}", "@I", str(n2)], n2, env="reactot", part="C2",
                     c=64, mem=240, deps=keys(r), disk_gb=nf / n2 * 0.01)
            u += rr
            sink_deps += keys(rr)
    return u + one("status", ["mlts_autode.py", "status"], env="reactot", part="C", c=2, mem=8, deps=sink_deps, sink=True)


def start_a4(P, S, st):
    if (lists_info(P).get("n_fold_minus1") or 0) > 0:
        return (f"A-4: fold −1(anchor-only core, 모든 학습 쪽) {lists_info(P)['n_fold_minus1']}반응은 자기 FT_k가 없어 "
                "FT_0_al을 쓴다(추천안: C-1을 다시 한 모델, 어느 FT_k_al에서도 학습 쪽).")
    return None


def a4_folds_done(P):
    f = (P.rj("A4_report.json") or {}).get("folds") or {}
    return sorted(str(k) for k, v in f.items() if (v or {}).get("complete"))


def h_chain(P, st, tag, early=False, root=None):
    """the inputs_mlts.py chain for source mlts; root = an A-2 layout below another folder (--root R: every mlts path
    and report moves below R, e.g. the A″ arm $MLT/app)."""
    def ak(x):
        return f"{x}{tag}"
    ro = ["--root", str(root)] if root else []

    def im(*a):
        return ["inputs_mlts.py", *a] + ro
    dpr = h_disk_per_rxn(P)
    nfin = fixed_n(P, st, ak("fin"), lambda: 2 if early else 4)
    fin = arr(ak("fin"), im("finalize", "mlts", "@I", str(nfin)), nfin, env="reactot", part="C2", c=64,
              mem=240, disk_gb=N_D1 / nfin * dpr / 2)
    meta = one(ak("meta"), im("meta", "mlts"), env="reactot", part="C", c=4, mem=16, deps=keys(fin))
    hfin = one(ak("hfin"), im("hf_inputs", "mlts"), env="reactot", part="C", c=4, mem=16, deps=keys(meta))
    nhf = fixed_n(P, st, ak("hfr"), lambda: 2 if early else 4)
    hfr = arr(ak("hfr"), im("hf_run", "mlts", "@I", str(nhf)), nhf, env="reactot", part="C2", c=64, mem=240,
              deps=keys(hfin), disk_gb=N_D1 / nhf * dpr / 2)
    if early:
        return fin + meta + hfin + hfr
    nx = fixed_n(P, st, "x182", lambda: 4)
    x = arr("x182", im("x182", "mlts", "@I", str(nx)), nx, env="reactot", part="C", c=32, mem=64, deps=keys(meta))
    xm = one("x182m", im("x182_merge", "mlts"), env="reactot", part="C", c=4, mem=32, deps=keys(x))
    na = fixed_n(P, st, "atoms", lambda: 4)
    at = arr("atoms", im("atoms", "mlts", "@I", str(na)), na, env="reactot", part="C", c=32, mem=64, deps=keys(xm))
    tok = one("tokens", im("tokens", "mlts"), env="reactot", part="C", c=8, mem=32, deps=keys(at))
    hfb = one("hfb", im("hf_build", "mlts"), env="reactot", part="C", c=8, mem=32, deps=keys(hfr))
    d3 = one("d3", im("d3", "mlts"), env="reactot", part="C", c=8, mem=16, deps=keys(meta))
    b = one("build", im("build", "mlts"), env="reactot", part="C", c=8, mem=64,
            deps=keys(tok) + keys(hfb) + keys(d3) + keys(xm))
    bm = one("buildmin", im("build", "mlts", "--refs", "min"), env="reactot", part="C", c=8, mem=64, deps=keys(b))
    h2 = one("h2", im("h2", "mlts"), env="reactot", part="C", c=2, mem=8, deps=keys(b) + keys(bm), sink=True)
    return fin + meta + x + xm + at + tok + hfin + hfr + hfb + d3 + b + bm + h2


# ---- the optional A″ arm (spec 6 A-3: A″ <= 1 core-h per reaction -> a separate arm in K / E)
def app_root(P):
    return P.c.mlt / "app"


def extra_app(P):
    return bool((P.state.get("params") or {}).get("extra_arm_App"))


def units_a4app(P, S, st):
    """A″ for every D1 target: refine_dft reads A's folders ($MLT/ade/jobs) and writes the A-2 layout below
    $MLT/app (contract: `mlts_autode.py refine_dft d1:<k> @I n --root R`, then `status --root R` ->
    R/reports/A4_report.json)."""
    u = []
    per = app_disk_per_rxn(P)
    folds = list(range(5)) + ([-1] if (lists_info(P).get("n_fold_minus1") or 0) > 0 else [])
    for k in folds:
        nf = n_list(P, f"f{k}_test") if k >= 0 else int(lists_info(P).get("n_fold_minus1") or 0)
        n = fixed_n(P, st, f"r{k}", lambda: 2)
        u += arr(f"r{k}", ["mlts_autode.py", "refine_dft", f"d1:{k}", "@I", str(n), "--root", str(app_root(P))], n,
                 env="reactot", part="C2", c=64, mem=240, disk_gb=nf / n * per)
    return u + one("status", ["mlts_autode.py", "status", "--root", str(app_root(P))], env="reactot", part="C", c=2,
                   mem=8, deps=keys(u), sink=True)


def done_a4app(P):
    return (P.rjp(app_root(P) / "reports" / "A4_report.json") or {}).get("complete") is True


def done_happ(P):
    p = app_root(P) / "reports" / "H2_report.json"
    J = P.rjp(p) or {}
    ta = P.mt(app_root(P) / "reports" / "A4_report.json")
    return bool(J.get("judged")) and (ta is None or P.mt(p) >= ta)


def decide_happ(P, S, st):
    """the optional arm's H-2: a rate > 2 % drops the arm from K / E (recommended option: the spec STOP is about the
    main inputs; the arm is extra) — recorded, no HALT."""
    J = P.rjp(app_root(P) / "reports" / "H2_report.json") or {}
    ok = not J.get("stop")
    P.state.setdefault("params", {})["app_usable"] = ok
    write_arms(P)
    if ok:
        return dict(key="A'' arm inputs ok", line=f"A″ arm 입력(H-App): 실패율 {fmt(J.get('rate'), 4)} ≤ 0.02 → K · E에서 별도 "
                                                   f"arm으로 쓴다($MLT/pipeline/arms.json app_usable true).")
    return dict(key="A'' arm dropped", line=f"A″ arm 입력(H-App): 실패율 {fmt(J.get('rate'), 4)} > 0.02 → 추천안: 이 선택 arm을 "
                                              f"K · E에서 뺀다(주 입력 H는 따로 판정; arms.json app_usable false).")


def units_hearly(P, S, st):
    if P.st("A4").get("status") == "done":
        return []
    fd = a4_folds_done(P)
    if not fd:
        return []
    tag = "@" + "".join(fd)
    if tag in (st.get("params") or {}).get("done_tags", []):
        return []                                               # this fold set is finalised; wait for more folds
    st.setdefault("params", {})["cur_tag"] = tag
    return h_chain(P, st, tag, early=True)


def done_h(P):
    p = P.c.res / "H2_report.json"
    J = P.rjp(p) or {}
    ta = P.mt(P.c.res / "A4_report.json")
    return bool(J.get("judged")) and not J.get("stop") and (ta is None or P.mt(p) >= ta)


def decide_h(P, S, st):
    J = P.rj("H2_report.json") or {}
    if J.get("stop"):
        return dict(key="H-2 fail", numbers={k: J.get(k) for k in ("n_with_ts", "n_fail", "rate", "by_stage")},
                    halt=f"H-2 입력 실패율 {fmt(J.get('rate'), 4)} > 0.02 (명령서 7절 STOP)")
    return dict(key="H-2 pass", line=f"H-2: 입력 실패율 {fmt(J.get('rate'), 4)} ≤ 0.02 ({J.get('n_fail')}/{J.get('n_with_ts')}).")


def decide_cagg(P, S, st):
    """tests.py all: a failed C-1 .. C-6 check = HALT (C_tests stop; C-2 / C-6 / H-1 are spec 14 STOPs, C-1 / C-3 /
    C-4 failures are bugs the spec says to report before fixing); C-7 is report only."""
    J = P.rj("C_tests.json") or {}
    secs = {k: (J.get(k) or {}) for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}
    bad = [k for k in ("C1", "C2", "C3", "C4", "C5", "C6") if secs[k].get("pass") is False]
    if J.get("stop") or bad:
        return dict(key="C fail " + ",".join(bad), numbers={k: secs[k].get("pass") for k in secs},
                    halt=f"Phase C 검사 실패 {bad or J.get('stop')} (명령서 8 · 14절: C-2 누수 / C-6 회귀 / C-5 = H-1은 STOP, "
                         "C-1 · C-3 · C-4 실패는 버그라 고치기 전에 보고)")
    c7 = secs["C7"]
    return dict(key="ok", line=f"Phase C 검사 모음: C-1 … C-6 통과, C-7 보고(통과 {c7.get('pass')}: "
                               f"{str(c7.get('summary') or '')[:120]}).")


def done_r(P):
    """PREREG_V2.md and prereg_v2.json both committed and unchanged, and the json's md_sha256 = sha256(md) (the same
    rule as `prereg_v2.py check`, computed here with hashlib: K and E read criteria.refs.primary from the json)."""
    f, j = P.c.res / "PREREG_V2.md", P.c.res / "prereg_v2.json"
    if not (f.is_file() and j.is_file()):
        return False
    J = P.rjp(j) or {}
    try:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
    except OSError:
        return False
    return J.get("md_sha256") == sha and all(P.c.git.tracked(p) and P.c.git.clean(p) for p in (f, j))


def prereg_refs(P):
    J = P.rj("prereg_v2.json") or {}
    return ((J.get("criteria") or {}).get("refs") or {}).get("primary") or "min"


def c1_report_ok(P, name, refs):
    p, tp = P.c.res / name, P.mt(P.c.res / "PREREG_V2.md")
    J = P.rjp(p) or {}
    t = P.mt(p)
    return t is not None and tp is not None and t >= tp and J.get("refs") == refs


def done_kc1(P):
    """K-C1 (no retraining): C1 predictions on the registered refs mode (prereg primary: C1_report.json) and on the
    other mode (secondary, descriptive: C1_<mode>_report.json), both written after PREREG_V2.md."""
    prim = prereg_refs(P)
    sec = "own" if prim == "min" else "min"
    return c1_report_ok(P, "C1_report.json", prim) and c1_report_ok(P, f"C1_{sec}_report.json", sec)


def units_kc1(P, S, st):
    prim = prereg_refs(P)
    sec = "own" if prim == "min" else "min"
    u = []
    for tag, refs in (("", prim), (f"_{sec}", sec)):
        x = ["--refs", refs]
        u += one(f"atoms{tag}", ["train_k.py", "c1pred", "atoms"] + x, env="reactot", part="GA", c=8, mem=64, gpu=1)
        u += one(f"xgb{tag}", ["train_k.py", "c1pred", "xgb"] + x, env="reactot", part="C", c=32, mem=64)
        u += one(f"collect{tag}", ["train_k.py", "c1pred", "collect"] + x, env="reactot", part="C", c=4, mem=32,
                 deps=[f"atoms{tag}", f"xgb{tag}"], sink=True)
    return u


def units_kc23(P, S, st):
    u = one("plan", ["train_k.py", "plan"], env="reactot", part="C", c=2, mem=8, done=rep_complete("K_plan.json"))
    K = P.rj("K_plan.json") or {}
    if not K.get("complete"):
        return u
    allk = []
    for x in K.get("units") or []:
        n = _int(x.get("n"), 3)
        c = _int(x.get("cpus"), 64)
        part = "GA" if _int(x.get("gpu"), 0) else ("C2" if c >= 64 else "C")
        a = arr(f"{x['arm']}/{x['unit']}", ["train_k.py", x["arm"], x["unit"], "@I", str(n)], n, env="reactot", part=part,
                c=c, mem=_int(x.get("mem_gb"), 240 if c >= 64 else 64), gpu=_int(x.get("gpu"), 0), deps=["plan"])
        u += a
        allk += keys(a)
    return u + one("select", ["train_k.py", "select"], env="reactot", part="C", c=4, mem=16, deps=allk, sink=True)


def done_kc23(P):
    K = P.rj("K_report.json") or {}
    return K.get("complete") is True and "primary" in K


def decide_kc23(P, S, st):
    K = P.rj("K_report.json") or {}
    lk = K.get("leakage_C2") or {}
    if lk and lk.get("pass_") is False:
        return dict(key="C-2 fail (K)", numbers=dict(leakage_C2=lk), halt="K: 채널 단위 학습 짝 ∩ fold k ≠ ∅ (C-2 누수, 명령서 14절 STOP)")
    return dict(key=f"primary {K.get('primary')}", line=f"K: 1차 모델 = {K.get('primary')} (안쪽 검증 oi + cpcm para MAE; "
                                                       f"results/K_report.json). C-2(K 단위)는 C2-K 단계가 따로 검사한다.")


def done_c2k(P):
    p = P.c.res / "C2K_report.json"
    J = P.rjp(p) or {}
    tk = P.mt(P.c.res / "K_report.json")
    return J.get("complete") is True and "pass" in J and (tk is None or P.mt(p) >= tk)


def decide_c2k(P, S, st):
    J = P.rj("C2K_report.json") or {}
    if J.get("pass") is True:
        return dict(key="pass", line=f"C-2 (K 채널 단위): 학습 짝 ∩ fold k = ∅ — {J.get('n_units')}개 단위 통과.")
    return dict(key="C-2 fail", numbers={k: J.get(k) for k in ("n_units", "n_overlap", "overlaps")},
                halt="C-2 누수: K 채널 단위(C2/C3)의 f{k} 학습 짝 ∩ fold k ≠ ∅ (명령서 8 · 14절 STOP)")


def decide_e(P, S, st):
    E = P.rj("E_report.json") or {}
    if E.get("prereg_change_needed"):
        return dict(key="prereg change", numbers=dict(reason=E.get("prereg_change_reason")),
                    halt="E: 결과를 본 뒤 등록 내용을 바꿔야 할 것 같다(명령서 14절 STOP)")
    return dict(key="ok", line=f"E 평가 끝: 등급 {E.get('grade', '—')} (results/E_report.json).")


def start_p(P, S, st):
    return "P-3 DFT 증명서(최종 TS에서 라벨 수준 EnGrad 1회, 시험 200반응, 약 20 core-h)는 선택 항목이지만 추천안대로 실행한다."


def decide_p(P, S, st):
    J = P.rj("P_report.json") or {}
    if J.get("selfcheck_pass") is False:
        return dict(key="self-check fail", numbers={k: J.get(k) for k in ("selfcheck", "selfcheck_pass")},
                    halt="P-2 자체 검사 실패(D1 20반응 E 예측 불일치 또는 새 SMILES 5개 실패)")
    return dict(key="ok", line="P 추론기 자체 검사 통과 (results/P_report.json).")


def decide_lists(P, S, st):
    info = lists_info(P)
    a3 = (info.get("files") or {}).get("a3") or {}
    fd = a3.get("by_fold_dipole") or {}
    per_fold = {}
    for kd, v in fd.items():
        k, _, d = kd.partition("/")
        per_fold.setdefault(k, []).append(f"{d} {v}")
    short = sorted(k for k, v in per_fold.items() if len(v) < 5)
    why = (f"fold {', '.join(short)}에는 시험 반응이 없는 dipole이 있어 dipole마다 12개를 채울 수 없다. 추천안: 그 fold에 있는 "
           f"dipole에 60개를 고르게 나눈다" if short else "dipole마다 12개")
    return dict(key="lists", numbers=dict(n=a3.get("n"), by_fold=a3.get("by_fold")),
                line=(f"목록(pipeline.py lists): fold 시험 {', '.join(str(n_list(P, f'f{k}_test')) for k in range(5))}반응, "
                      f"fold −1 {info.get('n_fold_minus1')}반응(A-4에서 FT_0_al). A-3 시범 목록 = fold마다 60반응(명령서 6절): "
                      f"{why}(시드 {M.SEED}+3) — " + "; ".join(f"f{k}: {', '.join(v)}" for k, v in sorted(per_fold.items()))
                      + "."))


def decide_ftfin(P, S, st):
    write_ft_outputs(P, gate=((P.st("FT2-r2").get("decision") or {}).get("numbers")))
    return dict(key="final models", line="FT_all_s0–2 · FT_LODO_0–4 (--with-al) 학습 끝 → results/FT_report.json에 FT-1 추가.")


def decide_s(P, S, st):
    P.state["finished"] = P.now
    if not P.dry:
        M.write_text(P.c.res / "STOP_S.md", f"# 명령서 S: 멈춤\n\n{P.now} pipeline.py: SUMMARY_V2.md가 생겼다. 명령서 순서의 끝"
                                            "(S 요약 후 멈춤). launcher는 더 이상 자신을 다시 내지 않는다.\n")
    return dict(key="finished", line="명령서 S: 멈춤 — SUMMARY_V2.md 작성 완료, 자동 실행을 끝낸다(launcher 재예약 중지).")


# ================================================================ the registry (topological order)
def registry():
    R, Mm, D = "reactot", "mlts", "d1"
    S = []
    S.append(Stage("lists", "pipe", 5, lambda P, S_, st: one("lists", ["pipeline.py", "lists"], env=R, part="C", c=2, mem=8),
                   code=[("pipeline.py", "lists")], done=lists_done, decide=decide_lists,
                   summary="pipeline.py lists -> $MLT/pipeline/lists/{f<k>_test,dip<d>,a3}.csv"))
    S.append(Stage("G-D1", "G", 10, units_gd1, code=[("gen_dft.py", "run"), ("gen_dft.py", "run_g2"),
                                                       ("gen_dft.py", "collect")],
                   done=done_gd1, decide=decide_gd1, guard=guard_dft_stop("G_report.json"), progress=prog_gd1,
                   adopt=[(r"g_run1", "G1[{i}]"), (r"g_run1rev", "G1rev"), (r"g_run3", "G3[{i}]"), (r"g_run2", "G2[{i}]"),
                          (r"g_collect", "collect")], results=["G_report.json"],
                   summary="adopt g_run1/1rev/3/2 + g_collect; done = G_report complete, or G1 / G3 complete with the "
                           "G-2 paths the user stopped at 317 / 400 (decision Nurion); replan: run G1|G3 for groups "
                           "with n_missing > 0 (never G-2 again), collect"))
    d0code = ("gen_dft.py", "plan_d0", r"gen_d0")
    S.append(Stage("G-D0p", "G", 11, units_gd0p, code=[d0code, ("gen_d0.py", "")], done=done_gd0p, guard=guard_d0,
                   adopt=[(r"g_plan_?d0", "plan_d0")],
                   results=["G_plan_d0.json"], summary="gen_dft plan_d0 (auto) -> results/G_plan_d0.json"))
    S.append(Stage("G-D0e", "G", 12, units_gd0e, after=["G-D0p"], code=[("gen_dft.py", "g0_d0"), ("gen_d0.py", "")],
                   done=done_gd0e, decide=decide_gd0e, on_exit3="decide", guard=guard_d0, results=["G0_estimate_d0.json"],
                   adopt=[(r"g_g0_?d0", "g0_d0")],
                   summary="gen_dft g0_d0 -> G0_estimate_d0.json; D1 + D0 G1/G3 + 200 D0G2 + G-4 reserve <= "
                           "gen_dft.BUDGET_CORE_H -> the user's D0 selection, else HALT"))
    S.append(Stage("G-D0f", "G", 13, units_gd0f, after=["G-D0e"], code=[("gen_d0.py", "", r"--n-holdout")],
                   done=done_gd0f, guard=guard_d0, results=["G_plan_d0.json", "d0_holdout.csv"],
                   adopt=[(r"d0p_plan", "plan_d0")],
                   summary="gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100 (user decisions D0-n3 / D0-plus / "
                           "D0-gauss: 200 D0G2 training paths, 100 evaluation-only reactions not refined)"))
    S.append(Stage("G-D0", "G", 14, units_gd0, after=["G-D0f"],
                   code=[("gen_dft.py", "run", r"D0G1"), ("gen_dft.py", "run_g2", r"--d0"),
                         ("gen_dft.py", "collect", r"--d0"), ("gen_d0.py", "")],
                   done=done_gd0, guard=guard_d0, progress=prog_gd0, results=["G_report_d0.json"],
                   adopt=[(r"g_run_?d0g?1", "D0G1[{i}]"), (r"g_run_?d0g?3", "D0G3[{i}]"), (r"g_run_?d0g?2", "D0G2[{i}]"),
                          (r"g_collect_?d0", "collect"), (r"g_\w*d0\w*", None)],
                   summary="run D0G1|D0G3 @I n (n = suggested_tasks or the adopted manual array), run_g2 @I n --d0 (the "
                           "200 training units of g2_plan_d0.json), collect --d0 -> G_report_d0.json; helpers "
                           "g_run_d0g*rev / shuf<K> recognised (block resubmission while live)"))
    S.append(Stage("N-scan", "I", 15, units_nscan, code=[("inv_nurion.py", "scan"), ("inv_nurion.py", "verify"),
                                                         ("inv_nurion.py", "collect")],
                   done=done_nscan, progress=prog_nscan, rearm=True, halt_scope="local", results=["N_report.json"],
                   adopt=[(r"nur_(?P<x>.+)", lambda m, i: f"scan/{m.group('x')}")],
                   summary="per uploaded, unscanned Nurion tar: inv_nurion scan <tar> ::: verify <tar> (cpu1/cpu2 8c "
                           "24G); collect -> results/N_report.json; re-arms until N_report complete (23 result tars); "
                           "the user deletes the copies by hand; FT data wait: N_report complete or '- **N partial "
                           "ok' in DECISIONS"))
    S.append(Stage("Z-run", "Z", 20, units_z, code=[("zero_shot.py", "run")],          # Z reads no G output
                   done=lambda P: all(z_done(P, c) for c in Z_CANDS),
                   progress=lambda P: sum(count_dir(P.c.mlt / "z" / c / "geom_label" / "recs") for c in Z_CANDS),
                   adopt=[(r"z_omol", "MACE-OMOL-0"), (r"z_mh1", "MACE-MH-1"), (r"z_aim", "AIMNet2"),
                          (r"z_off24", "MACE-OFF24"), (r"z_xtb", "xtb")],
                   summary="zero_shot run <c> x 5 (adopt z_*)"))
    S.append(Stage("Z-speed", "Z", 21, lambda P, S_, st: one("speed", ["zero_shot.py", "speed"], env=Mm, part="G48", c=8,
                                                              mem=32, gpu=1),
                   code=[("zero_shot.py", "speed")], done=done_zspeed, adopt=[(r"z_speed", "speed")],
                   summary="zero_shot speed (one G48 job)"))
    S.append(Stage("Z-rep", "Z", 22, lambda P, S_, st: one("report", ["zero_shot.py", "report"], env=Mm, part="C", c=2, mem=8),
                   after=["Z-run", "Z-speed"], code=[("zero_shot.py", "report")], done=done_zrep, decide=decide_zrep,
                   adopt=[(r"z_report", "report")], results=["Z_report.json", "Z_DECISION.md"],
                   summary="zero_shot report -> top 2 pilots (1 -> pilot it; 0 -> HALT)"))
    S.append(Stage("FT-dp", "FT", 30, units_ftdp, after=["G-D1", "G-D0", "Z-rep"], code=[("finetune.py", "data")],
                   need=[N_NEED], done=done_ftdp, adopt=[(r"ft_data", "data")], results=["ft/FTpilot_*_data.json"],
                   summary="finetune data FTpilot_<a>_0 FTpilot_<b>_0 (after D1 + D0 G collect and the Nurion data)"))
    S.append(Stage("FT-p", "FT", 31, units_ftp, after=["FT-dp"], code=[("finetune.py", "train"), ("finetune.py", "ft1")],
                   done=done_ftp, progress=prog_train(lambda P: [f"FTpilot_{c}_0" for c in pilots(P)]),
                   adopt=[(r"ft_p_(?P<x>.+)", lambda m, i: FT_ALIAS.get(m.group("x"), m.group("x")))],
                   results=["ft/FTpilot_*_train.json", "ft/FTpilot_*_ft1.json"],
                   summary="finetune train FTpilot_<c>_0 <c> ::: ft1 (MACE G24 96G, AIMNet2 GA 64G)"))
    ft2root, xroot = (lambda P: P.c.mlt / "ft2"), (lambda P: P.c.mlt / "x")
    S.append(Stage("FT2-p", "FT", 32, units_ft2p, after=["Z-rep", "lists"], code=[("ts_geom.py", "run", r"--only-failed")],
                   need=[(f"the gate-fold models FTpilot_<c>_{GATE_FOLD} trained (finetune.py train ::: ft1; FT2-p does not "
                          f"wait for FT-p / fold 0)", gate_trained),
                         (f"the E/Z sibling list $MLT/pipeline/lists/{EZ_LIST_NAME}.csv (ez_groups.py run --folds "
                          f"{GATE_FOLD})", lambda P: bool(ez_map(P, GATE_FOLD)))],
                   adopt=[(FT2_SPEC_RX, lambda m, i: f"{gate_run(FT2_SPEC_CAND)}/{m.group('s')}[{i}]")],
                   done=done_ft2p, decide=decide_ft2p, partial=partial_ft2p, progress=prog_ft2p,
                   results=["DEVIATIONS.md"],
                   summary=f"gate fold {GATE_FOLD} (user 11-05 15:5x '바꿔줘', was fold 0; FTpilot_<c>_{GATE_FOLD} trained "
                           f"first): ts_geom run FTpilot_<c>_{GATE_FOLD} {GATE_LIST} label|g2 (the lead's arrays "
                           f"ft2_spec_f{GATE_FOLD}_<start> adopted, n = 2; a start whose array is live gets no unit) -> "
                           "failed reactions again with --recalc 5 --only-failed (<start>_recalc5) -> base MLIP ranked "
                           f"on fold {GATE_FOLD} (base_run = FT_0 := FTpilot_<base>_0, fold 0) + the FT-2 (a) gate on the "
                           f"merged records (recs/*.json), E/Z split by {EZ_LIST_NAME} (> 0.05 same-TS or > 0.002 A "
                           "median |dd| -> 'without' only; an empty group -> all); after Z-rep + lists, not FT-p"))
    S.append(Stage("FT-d1", "FT", 33, units_ftd1, after=["FT2-p"], code=[("finetune.py", "data")], need=[N_NEED],
                   done=lambda P: all(data_ok(P, f"FT_{k}") for k in range(1, 5)), results=["ft/FT_[1-4]_data.json"],
                   summary="finetune data FT_1 FT_2 FT_3 FT_4"))
    S.append(Stage("FT-r1", "FT", 34, units_ftr1, after=["FT-d1"], code=[("finetune.py", "train"), ("finetune.py", "ft1")],
                   need=[BASE_NEED], done=lambda P: all(train_done(P, r1_run(P, k)) for k in range(1, 5)),
                   progress=prog_train(lambda P: [r1_run(P, k) for k in range(1, 5)]),
                   results=["ft/FT_[1-4]_train.json", "ft/FT_[1-4]_ft1.json"],
                   summary="train FT_<k> <base> ::: ft1 FT_<k>, k = 1..4"))
    S.append(Stage("FT2-r1", "FT", 35, units_ft2r1, after=["FT2-p", "lists"], code=[("ts_geom.py", "run")],
                   done=done_ft2r1, filler=True,
                   partial=geom_partial(geom_dirs(lambda P: [(run, f"f{k}_test") for k, run in ft2r1_runs(P)], ft2root)),
                   progress=lambda P: geom_progress(P, [run or "_" for _k, run in ft2r1_runs(P)], P.c.mlt / "ft2"),
                   summary="filler: ts_geom on FT_k (round 1), fold k, both starts (report only): fold 0 (FT_0 = "
                           f"FTpilot_<base>_0) and folds 1-3; fold {GATE_FOLD} skipped (the FT2-p gate model)"))
    S.append(Stage("FT2-f0", "FT", 36, None, after=["FT2-p"], gate=gate_ft2f0, decide=decide_ft2f0, halt_scope="local",
                   results=[F0_REPORT],
                   summary="report only, no jobs: once FT2-r1's fold-0 first pass of FT_0 = FTpilot_<base>_0 on f0_test is "
                           "final (both starts), the fold-0 FT-2 (a) / (b) numbers split by f0_test_ez.csv (user 12:0x "
                           f"(2)) -> results/{F0_REPORT} + one DECISIONS line (the gate is fold {GATE_FOLD}, FT2-p)"))
    S.append(Stage("FT-abl", "FT", 37, None, after=["lists"], gate=gate_ftabl, decide=decide_ftabl, halt_scope="local",
                   summary=f"decision only, no jobs (user 11-05 17:0x (2)): once the baseline {ABL_BASE_RUN} label first "
                           f"pass is final and every ablation FTabl_{ABL_CAND}_{GATE_FOLD}_<{'|'.join(ABL_TAGS)}> (the "
                           f"lead's ft_abl_<tag> ::: ft2_abl_<tag>, read by name from squeue / sacct, not adopted) is "
                           f"finished (FT-2 label first pass on {GATE_LIST}: all or >= 99.5 %, no live task) or dead "
                           f"(training ended without results, or neither finished nor live) -> pre-registered choice "
                           f"(first-pass same-TS gain > {ABL_MIN_GAIN} over the baseline, then median |dd| within "
                           f"{ABL_DD_TIE_A} A, then the smaller fw) -> params ft_train_args / ft_abl_choice / ft_abl_cand "
                           f"for FT-r2 + FT-fin `finetune.py train`; after lists only (independent of FT2-p and its "
                           f"HALT); G4 / FT-d2 do not wait for it"))
    S.append(Stage("G4", "G", 40, units_g4, after=["FT-r1", "FT-p"], need=[BASE_NEED],
                   code=[("g4_al.py", "g4_plan"), ("g4_al.py", "g4_opt"), ("g4_al.py", "run", r"G4"),
                         ("g4_al.py", "collect", r"--g4")],
                   done=done_g4, decide=decide_g4, guard=guard_g4,
                   results=["G4_plan.json", "G4_report.json"],
                   summary="after FT-r1 and FT-p (all five first-round models, FT_0 = FTpilot_<base>_0; not FT-abl): "
                           "g4_al g4_plan (1,000 D1 + 500 D0 training-side reactions) -> g4_opt @I n (GPU) -> run G4 "
                           "@I n -> collect --g4"))
    S.append(Stage("FT-d2", "FT", 41, units_ftd2, after=["G4"], code=[("finetune.py", "data")], need=[N_NEED],
                   done=done_ftd2, results=["ft/*_al_data.json"], summary="finetune data --all --with-al"))
    S.append(Stage("FT-r2", "FT", 42, units_ftr2, after=["FT-d2", "FT-abl"],
                   code=[("finetune.py", "train"), ("finetune.py", "ft1")], on_start=start_ftr2,
                   done=lambda P: all(train_done(P, f"FT_{k}_al") for k in range(5)),
                   progress=prog_train(lambda P: [f"FT_{k}_al" for k in range(5)]),
                   results=["ft/FT_[0-4]_al_train.json", "ft/FT_[0-4]_al_ft1.json"],
                   summary="after FT-d2 and FT-abl: train FT_<k> <base> --with-al <FT-abl ft_train_args when the base "
                           "is AIMNet2> ::: ft1 FT_<k> --with-al (k = 0..4)"))
    S.append(Stage("FT2-r2", "FT", 43, units_ft2r2, after=["FT-d2", "lists"],
                   code=[("ts_geom.py", "run", r"--only-failed"), ("ez_groups.py", "run", r"--folds")],
                   done=done_ft2r2, decide=decide_ft2r2, partial=partial_ft2r2, progress=prog_ft2r2,
                   results=["FT_report.json", "FT_geom.csv", "DEVIATIONS.md"],
                   summary="ez_groups run --folds 0,1,2,3,4 (CPU, while a list f<k>_test_ez.csv is missing) + ts_geom "
                           "FT_<k>_al fold k both starts -> failed reactions again with --recalc 5 --only-failed "
                           "(<start>_recalc5) -> pooled FT-2 (a) gate on the merged records, E/Z 'with' / 'without "
                           "sibling' split pooled over the folds (> 0.05 same-TS or > 0.002 A median |dd| -> 'without' "
                           "only + DEVIATIONS) -> HALT on fail; results/FT_report.json + FT_geom.csv"))
    S.append(Stage("FT-fin", "FT", 44, units_ftfin, after=["FT-d2", "FT-abl"],
                   code=[("finetune.py", "train"), ("finetune.py", "ft1")],
                   done=lambda P: all(train_done(P, r + "_al") for r in FIN_RUNS), filler=True, on_start=start_ftfin,
                   decide=decide_ftfin, progress=prog_train(lambda P: [r + "_al" for r in FIN_RUNS]),
                   results=["ft/FT_all_s*_al_train.json", "ft/FT_all_s*_al_ft1.json", "ft/FT_LODO_*_al_train.json",
                            "ft/FT_LODO_*_al_ft1.json", "FT_report.json"],
                   summary="filler, after FT-d2 and FT-abl: FT_all_s0-2 + FT_LODO_0-4, --with-al + the FT-abl "
                           "ft_train_args (the same setting as FT-r2)"))
    S.append(Stage("X-run", "X", 45, units_xrun, after=["FT-d2", "lists"], code=[("ts_geom.py", "run")], done=done_xrun,
                   filler=True, progress=lambda P: geom_progress(P, [f"FT_LODO_{d}_al" for d in range(5)], P.c.mlt / "x"),
                   partial=geom_partial(geom_dirs(lambda P: [(f"FT_LODO_{d}_al", f"dip{d}") for d in range(5)], xroot)),
                   summary="filler: ts_geom FT_LODO_<d>_al on dipole d, both starts"))
    S.append(Stage("X-rep", "X", 46, lambda P, S_, st: one("x_report", ["ts_geom.py", "x_report"], env=Mm, part="C", c=4,
                                                             mem=16),
                   after=["X-run"], code=[("ts_geom.py", "x_report")], done=rep_complete("X_report.json"),
                   halt_scope="local", results=["X_report.json"],
                   summary="ts_geom x_report -> results/X_report.json (report only)"))
    S.append(Stage("D0-eval", "X", 47, units_d0eval, after=["FT-fin"], code=[("ts_geom.py", "d0_eval")],
                   done=rep_complete("D0_eval_report.json"), filler=True, results=["D0_eval_report.json"],
                   progress=lambda P: sum(count_dir(P.c.mlt / "d0eval" / m / s / "recs") for m in D0E_MODELS
                                          for s in STARTS),
                   summary="D0 supplementary geometry evaluation (user decisions D0-plus / D0-gauss, report only): "
                           "ts_geom d0_eval run @I n (FT_all_s0-2_al x {Coley TS, G2c xTB TS} x the 100 D0 evaluation "
                           "reactions) -> d0_eval report -> results/D0_eval_report.json"))
    S.append(Stage("C1-ft", "C", 50, lambda P, S_, st: one("c1", ["tests_c1.py", "all", "FT_0_al"], env=Mm, part="G48",
                                                             c=16, mem=64, gpu=1),
                   after=["FT-d2"], need=[("FT_0_al trained", lambda P: train_done(P, "FT_0_al"))],
                   code=[("tests_c1.py", "all")], done=done_c1ft, decide=decide_c1ft, results=["C1_tests.json"],
                   summary="tests_c1 all FT_0_al (G48) -> pass_C1 and pass_C4 else HALT"))
    S.append(Stage("A2-smoke", "A", 51, lambda P, S_, st: one("smoke", ["mlts_autode.py", "smoke", "--model", "FT_0_al"],
                                                               env=Mm, part="GA", c=8, mem=32, gpu=1),
                   after=["C1-ft"], code=[("mlts_autode.py", "smoke")],
                   done=lambda P: (P.rj("A2_smoke.json") or {}).get("complete") is True, on_exit3="decide",
                   decide=decide_a2, results=["A2_smoke.json"],
                   summary="mlts_autode smoke --model FT_0_al -> results/A2_smoke.json {complete, pass}"))
    S.append(Stage("A3", "A", 52, units_a3, after=["FT2-r2", "A2-smoke", "lists"],
                   code=[("mlts_autode.py", "run"), ("mlts_autode.py", "refine_dft"), ("mlts_autode.py", "a3_report")],
                   done=rep_complete("A3_report.json"), decide=decide_a3, results=["A3_report.json", "A3_status.csv"],
                   progress=prog_a3,
                   summary="mlts_autode run a3 @I n (A + A', GPU 16c) -> refine_dft a3 @I n (A'') -> a3_report; "
                           "engine rule on the allocated cost per reaction"))
    S.append(Stage("A4_disk", "A", 53, None, after=["A3"], gate=gate_a4,
                   summary="disk gate: quota fresh and free - (A-3 disk/rxn + H [+ A'' arm]) x 9,266 >= 30 GB"))
    S.append(Stage("A4", "A", 54, units_a4, after=["A4_disk"],
                   code=[("mlts_autode.py", "run"), ("mlts_autode.py", "refine_dft"), ("mlts_autode.py", "status")],
                   done=rep_complete("A4_report.json"), on_start=start_a4, results=["A4_report.json"],
                   progress=prog_a4,
                   summary="per fold k: mlts_autode run d1:<k> @I n --model FT_<k>_al --engine <e>; status -> A4_report"))
    S.append(Stage("A4-App", "A", 55, units_a4app, after=["A4"], active=extra_app,
                   code=[("mlts_autode.py", "refine_dft", r"--root"), ("mlts_autode.py", "status")],
                   done=done_a4app, progress=lambda P: count_dir(app_root(P) / "ade" / "jobs"),
                   summary="only if A-3 set extra_arm_App: refine_dft d1:<k> @I n --root $MLT/app (A'' for all 9,266); "
                           "status --root $MLT/app -> $MLT/app/reports/A4_report.json"))
    S.append(Stage("H-early", "H", 59, units_hearly, after=["A4_disk"],
                   code=[("inputs_mlts.py", "finalize"), ("inputs_mlts.py", "meta"), ("inputs_mlts.py", "hf_inputs"),
                         ("inputs_mlts.py", "hf_run")],
                   done=lambda P: P.st("A4").get("status") == "done", filler=True, rearm=True,
                   summary="filler: finalize -> meta -> hf_inputs -> hf_run on the A-4 folds finished so far"))
    S.append(Stage("H", "H", 60, lambda P, S_, st: h_chain(P, st, ""), after=["A4", "H-early"],
                   code=[("inputs_mlts.py", x) for x in ("finalize", "meta", "x182", "x182_merge", "atoms", "tokens",
                                                         "hf_inputs", "hf_run", "hf_build", "d3", "build", "h2")],
                   done=done_h, decide=decide_h, progress=lambda P: count_dir(P.c.mlt / "final"),
                   results=["H_*_mlts.json", "H2_report.json", "fu2_mlts/*"],
                   summary="inputs_mlts mlts: finalize -> meta -> x182/merge -> atoms/tokens, hf_inputs -> hf_run -> "
                           "hf_build, d3 -> build (own, min) -> h2"))
    S.append(Stage("H-App", "H", 61, lambda P, S_, st: h_chain(P, st, "", root=app_root(P)), after=["A4-App"],
                   active=extra_app,
                   code=[("inputs_mlts.py", x) for x in ("finalize", "meta", "x182", "x182_merge", "atoms", "tokens",
                                                         "hf_inputs", "hf_run", "hf_build", "d3", "build", "h2")],
                   done=done_happ, decide=decide_happ, progress=lambda P: count_dir(app_root(P) / "final"),
                   summary="only with the A'' arm: the H chain with --root $MLT/app -> $MLT/app/reports/H2_report.json; "
                           "H-2 > 2 % drops the arm (recorded)"))
    S.append(Stage("C-agg", "C", 70, lambda P, S_, st: one("all", ["tests.py", "all"], env=Mm, part="GA", c=16, mem=64, gpu=1),
                   after=["FT-r2", "A4", "H"], code=[("tests.py", "all")], done=rep_complete("C_tests.json"),
                   on_exit3="decide", decide=decide_cagg, results=["C_tests.json"],
                   summary="tests all -> C_tests.json; a failed C-1..C-6 = HALT, C-7 report only"))
    S.append(Stage("R", "R", 71, lambda P, S_, st: one("write", ["prereg_v2.py", "write"], env=R, part="C", c=2, mem=8),
                   after=["FT2-r2", "A3", "H", "H-App", "C-agg"], code=[("prereg_v2.py", "write")], done=done_r,
                   commit_gate=True, results=["PREREG_V2.md", "prereg_v2.json"],
                   summary="prereg_v2 write -> PREREG_V2.md + prereg_v2.json, committed by the launcher before K "
                           "(done = both tracked, unchanged, json md_sha256 = sha256(md))"))
    S.append(Stage("K-C1", "K", 72, units_kc1, after=["R"], code=[("train_k.py", "c1pred")],
                   done=done_kc1, results=["C1_report.json", "C1_*_report.json"],
                   summary="train_k c1pred atoms|xgb|collect --refs <prereg primary> (-> C1_predictions) and --refs "
                           "<other> (descriptive): predict_c1 on the registered refs table"))
    S.append(Stage("K-C2C3", "K", 73, units_kc23, after=["R"],
                   code=[("train_k.py", "plan"), ("train_k.py", "c2"), ("train_k.py", "c3"), ("train_k.py", "c1v"),
                         ("train_k.py", "select")],
                   done=done_kc23, decide=decide_kc23, results=["K_plan.json", "K_report.json", "K_val.csv"],
                   summary="train_k plan -> c2|c3|c1v <unit> @I 1 -> select (primary by inner validation oi + cpcm)"))
    S.append(Stage("C2-K", "C", 74, lambda P, S_, st: one("c2k", ["tests.py", "c2_k"], env=R, part="C", c=4, mem=32),
                   after=["K-C2C3"], code=[("tests.py", "c2_k")], done=done_c2k, decide=decide_c2k, on_exit3="decide",
                   results=["C2K_report.json"],
                   summary="tests c2_k -> C2K_report.json: K channel units' f{k} training pairs ∩ fold k = ∅, else HALT"))
    S.append(Stage("E", "E", 75, lambda P, S_, st: one("run", ["eval_v2.py", "run"], env=R, part="C", c=16, mem=64),
                   after=["K-C1", "K-C2C3", "C2-K"], code=[("eval_v2.py", "run")], done=rep_complete("E_report.json"),
                   decide=decide_e, results=["E_*"], summary="eval_v2 run -> E_report.json + E_metrics / E_metrics_pos / "
                                                             "E_bends / E_bend_rates / E_compare / figures"))
    S.append(Stage("W", "W", 75, lambda P, S_, st: (one("channel", ["final_w.py", "channel"], env=Mm, part="GA", c=8, mem=64,
                                                         gpu=1) +
                                                     arr("committee", ["final_w.py", "committee", "@I", "3"], 3, env=Mm,
                                                         part="GA", c=16, mem=64, gpu=1) +
                                                     one("report", ["final_w.py", "report"], env=R, part="C", c=4, mem=16,
                                                         deps=["channel", "committee[0]", "committee[1]", "committee[2]"],
                                                         sink=True)),
                   after=["FT-fin", "K-C2C3", "E"],
                   code=[("final_w.py", "channel"), ("final_w.py", "committee"), ("final_w.py", "report")],
                   done=rep_complete("W_report.json"), results=["W_*"],
                   summary="final_w channel (C1 / C2 / C3 final model); committee @I 3 (FT_all_s<i>_al); report -> "
                           "W_report.json"))
    S.append(Stage("P", "P", 76, lambda P, S_, st: (one("selfcheck", ["predict_v2.py", "selfcheck"], env=Mm, part="GA", c=16,
                                                         mem=64, gpu=1) +
                                                     arr("cert", ["predict_v2.py", "cert", "@I", "1"], 1, env=R, part="C2",
                                                         c=64, mem=240) +
                                                     one("report", ["predict_v2.py", "report"], env=R, part="C", c=4,
                                                         mem=16, deps=["selfcheck", "cert[0]"], sink=True)),
                   after=["W"], code=[("predict_v2.py", "selfcheck"), ("predict_v2.py", "cert"), ("predict_v2.py", "report")],
                   done=rep_complete("P_report.json"), decide=decide_p, on_start=start_p,
                   results=["P_report.json"], summary="predict_v2 selfcheck + cert @I 1 (P-3) -> report"))
    S.append(Stage("S", "S", 77, lambda P, S_, st: one("write", ["summary_v2.py", "write"], env=R, part="C", c=2, mem=8),
                   after=["X-rep", "D0-eval", "P"], code=[("summary_v2.py", "write")],
                   done=lambda P: (P.c.res / "SUMMARY_V2.md").is_file(), decide=decide_s,
                   results=["SUMMARY_V2.md", "STOP_S.md"], summary="summary_v2 write -> SUMMARY_V2.md, STOP_S.md, stop"))
    del D
    return S


def ancestors(reg):
    by = {s.name: s for s in reg}
    out = {}
    for s in reg:
        seen, todo = set(), list(s.after)
        while todo:
            a = todo.pop()
            if a in seen or a not in by:
                continue
            seen.add(a)
            todo += by[a].after
        out[s.name] = seen
    return out


# ================================================================ one pass
def new_state():
    return dict(version=CODE_VERSION, created=None, paused=False, halted=[], quota={}, slots={}, stages={},
                intents=[], params={}, push_pending=False, commit_pending=[], finished=None, resolved={},
                stop_removed=[], hist_scan=None)


def new_stage_state():
    return dict(status="", why="", epoch=0, att=dict(cont=0, infra=0), noprog=0, prog_epoch=None, units={}, params={},
                decision=None, t_done=None, commit=None, sticky=True, started=False, halt_kind=None, sbatch_err=0,
                sbatch_err_pass=None)


def new_unit_state():
    return dict(jobs=[], ok=None, epoch=0, att=dict(cont=0, infra=0, oom=0, code=0), noprog=0, prog_sub=None,
                mem_mult=1.0)


class Pass:
    def __init__(self, ctx, dry=False):
        self.c, self.dry = ctx, dry
        if dry:
            self.c.slurm = ReadOnlySlurm(self.c.slurm)
            if isinstance(self.c.git, Git):
                self.c.git.readonly = True
        self.t = ctx.clock()
        self.now = iso(self.t)
        self.reg = registry()
        self.by = {s.name: s for s in self.reg}
        self.anc = ancestors(self.reg)
        self.events, self.declines, self.planned, self.attn = [], [], [], []
        self.to_commit = []
        self._json, self._mt, self._prog, self._code, self._git = {}, {}, {}, {}, {}
        self._rows = {}                     # geom_rows cache (ts_geom recs/ folders, per pass)
        self.stop_submit = None
        self.n_sub = 0
        self.locked = False
        self.successor_ok = True
        self.cands = []
        self.unresolved, self.foreign, self.jobidx, self.masters = set(), {}, {}, set()
        self.paused = self.pause_exit = False

    # ---------------------------------------------------------------- io helpers
    def rjp(self, p):
        p = Path(p)
        try:
            m = p.stat().st_mtime
        except OSError:
            return None
        k = str(p)
        if k in self._json and self._json[k][0] == m:
            return self._json[k][1]
        try:
            v = json.loads(p.read_text())
        except (OSError, ValueError):
            v = None
        self._json[k] = (m, v)
        return v

    def rj(self, rel):
        return self.rjp(self.c.res / rel)

    def mt(self, p):
        try:
            return Path(p).stat().st_mtime
        except OSError:
            return None

    def st(self, name):
        s = self.state["stages"].setdefault(name, new_stage_state())
        for k, v in new_stage_state().items():
            s.setdefault(k, v)
        return s

    def log(self, ev):
        self.events.append(ev)

    def dec(self, stage, text, key=None):
        """append one DECISIONS.md line, idempotently: the line ends with an id (an invisible markdown comment) made
        from the stage and the event key, and a line whose id is already in the file is not written again (a pass
        that crashed after writing it and is re-run makes the same decision with the same key)."""
        did = hashlib.sha1(f"{stage}|{key if key is not None else text}".encode()).hexdigest()[:10]
        line = (f"- **auto {stage} ({time.strftime('%m-%d %H:%M', time.localtime(self.t))}, pipeline)**: {text} "
                f"<!-- mltp:{did} -->\n")
        self.declines.append(line)
        if self.dry:
            return
        p = M.assert_write_ok(self.c.dec_path)
        pre = ""
        try:
            raw = Path(p).read_bytes()
            if f"<!-- mltp:{did} -->".encode() in raw:
                return
            pre = "" if (not raw or raw.endswith(b"\n")) else "\n"
        except OSError:
            pass
        with open(p, "a") as f:
            f.write(pre + line)

    def deviation(self, stage, text, key):
        """append one line to results/DEVIATIONS.md (a rule applied at a decision that departs from the spec, e.g. the
        FT2-p E/Z 'without sibling' gate), idempotent by an id like dec(); never in dry mode."""
        did = hashlib.sha1(f"dev|{stage}|{key}".encode()).hexdigest()[:10]
        line = (f"- **auto {stage} ({time.strftime('%m-%d %H:%M', time.localtime(self.t))}, pipeline)**: {text} "
                f"<!-- mltp:{did} -->\n")
        if self.dry:
            self.planned.append(f"(dry) would append to results/DEVIATIONS.md: {text[:160]}")
            return
        p = M.assert_write_ok(self.c.res / "DEVIATIONS.md")
        try:
            raw = Path(p).read_bytes()
        except OSError:
            raw = b""
        if f"<!-- mltp:{did} -->".encode() in raw:
            return
        pre = "# DEPLOY-V2 (MLTS) 편차 기록\n\n" if not raw else ("" if raw.endswith(b"\n\n") else
                                                              ("\n" if raw.endswith(b"\n") else "\n\n"))
        with open(p, "a") as f:
            f.write(pre + line)
        self.log(f"{stage}: DEVIATIONS note")

    def write_state(self):
        if self.dry:
            return
        M.write_json(self.c.state_path, self.state)

    def flush_log(self):
        if self.dry or not self.events:
            return
        p = M.assert_write_ok(self.c.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        head = "" if p.is_file() else "# MLTS pipeline log (pipeline.py: one line per pass that changed something)\n\n"
        line = f"- {self.now[:16]} p{self.state['pass'].get('n')} j{self.c.job} | " + " | ".join(self.events) + "\n"
        with open(p, "a") as f:
            f.write(head + line)
        self.events = []

    # ---------------------------------------------------------------- lock
    def lock(self):
        """$MLT/pipeline/lock '<job> <iso time> <host> <pid>'. Held by a RUNNING job and younger than 12 min -> no
        pass; otherwise stale and taken over (a pass is killed after 600 s, so an older lock is never live — e.g. a
        manual `once` in the sandbox killed by SBX_T leaves the RUNNING sandbox as its holder)."""
        p = M.assert_write_ok(self.c.pdir / "lock")
        p.parent.mkdir(parents=True, exist_ok=True)
        me = f"{self.c.job} {self.now} {socket.gethostname()} {os.getpid()}\n"
        self.lock_token = me
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, me.encode())
            os.close(fd)
            self.locked = True
            return True
        except FileExistsError:
            pass
        try:
            tok = p.read_text().split()
        except OSError:
            tok = []
        holder = tok[0] if tok else ""
        t_lock = parse_iso(tok[1]) if len(tok) > 1 else None
        age = (self.t - t_lock) if t_lock else None
        if age is None or age < LOCK_STALE_S:
            try:
                sq = self.c.slurm.squeue()
            except SnapshotError as e:
                print(f"[pipeline] lock held by {holder}; squeue failed ({e}): no pass", flush=True)
                return False
            alive = any(l["id"] == holder and l["state"] == "RUNNING" for l in sq)
            mine = holder == self.c.job and len(tok) > 3 and tok[3] == str(os.getpid())
            if alive and not mine:
                print(f"[pipeline] lock held by live job {holder} ({'?' if age is None else round(age)} s): no pass",
                      flush=True)
                return False
        M.write_text(p, me)
        self.locked = True
        self.takeover = f"{holder} ({'age unknown' if age is None else f'{age:.0f} s'})"
        return True

    def unlock(self):
        if not self.locked:
            return
        p = self.c.pdir / "lock"
        try:
            if p.read_text() == getattr(self, "lock_token", None):
                os.remove(p)
        except OSError:
            pass

    # ---------------------------------------------------------------- the pass
    def run(self):
        if not self.dry and not self.lock():
            return 0
        try:
            if not self.load_state():
                return 0
            pn = int((self.state.get("pass") or {}).get("n") or 0) + 1
            self.state["pass"] = dict(n=pn, job=self.c.job, t0=self.now, t1=None, code=CODE_VERSION,
                                       host=socket.gethostname(), dry=self.dry)
            if getattr(self, "takeover", None):
                self.log(f"lock takeover from {self.takeover}")
            self.snapshot()
            self.read_pause()
            self.resolve_stops()
            self.index_jobs()
            self.reconcile_intents()
            self.adopt()
            self.fetch_ends()
            self.quota()
            self.successor()
            for S in self.reg:
                self.eval_stage(S)
            self.state["halted"] = [n for n, s in self.state["stages"].items() if s.get("status") == "halted"]
            self.commit()
            self.submit_all()
            self.finish()
            return 0
        finally:
            if not self.dry:
                self.unlock()

    def load_state(self):
        """results/PIPELINE_STATE.json; unreadable -> the previous pass's copy $MLT/pipeline/state.prev.json (jobs
        submitted since are re-adopted by their squeue comment); neither -> no pass (never a fresh start over an
        existing state: that would re-adopt and resubmit from scratch)."""
        st = None
        if self.c.state_path.is_file():
            st = self.rjp(self.c.state_path)
            if not isinstance(st, dict):
                prev = self.rjp(self.c.pdir / "state.prev.json")
                if isinstance(prev, dict):
                    st = prev
                    self.attn.append("results/PIPELINE_STATE.json unreadable: continued from $MLT/pipeline/state.prev.json")
                    self.log("state file unreadable -> state.prev.json")
                else:
                    msg = (f"{iso(self.c.clock())} results/PIPELINE_STATE.json unreadable and no $MLT/pipeline/"
                           "state.prev.json: pass skipped, nothing submitted — lead: restore it (git show HEAD:"
                           "analysis/mlts/results/PIPELINE_STATE.json)\n")
                    print(msg.strip(), flush=True)
                    if not self.dry:
                        M.write_text(self.c.ctl / "PIPELINE_ATTENTION", msg)
                    return False
        if not st:
            st = new_state()
            st["created"] = self.now
        for k, v in new_state().items():
            st.setdefault(k, v)
        self.state = st
        return True

    # ---------------------------------------------------------------- snapshot
    def snapshot(self):
        """one squeue (-r: one line per array task). total = every task of the user (launchers, sandbox, manual
        work); EP = eligible pending non-launcher tasks (they start before anything submitted later: FIFO)."""
        self.sq = self.c.slurm.squeue()
        self.live = {l["id"]: l for l in self.sq}
        other = [l for l in self.sq if l["name"] != LAUNCH]
        self.total = len(self.sq)
        self.T = len(other)
        self.Rn = sum(l["state"] == "RUNNING" for l in self.sq)
        me = self.live.get(self.c.job)
        self.self_run = 1 if (self.c.launcher and me and me["state"] == "RUNNING") else 0
        self.EP = sum(1 for l in other if eligible(l))
        self.CPU = sum(l["cpus"] for l in self.sq)
        self.GPU = sum(l["gpus"] for l in self.sq)
        fillers = {f"mltp_{s.name}" for s in self.reg if s.filler}
        self.T_filler = sum(l["name"] in fillers for l in other)
        self.succ_pending = any(l["name"] == LAUNCH and l["state"] == "PENDING" and l["id"] != self.c.job for l in self.sq)
        self.state["slots"] = dict(total=self.total, T=self.T, running=self.Rn, eligible_pending=self.EP, CPU=self.CPU,
                                   GPU=self.GPU, T_filler=self.T_filler, launcher=self.total - self.T, t=self.now)

    def read_pause(self):
        p = self.c.ctl / "PIPELINE_PAUSE"
        self.paused = p.is_file()
        try:
            self.pause_exit = self.paused and p.read_text().strip() == "exit"
        except OSError:
            self.pause_exit = False
        if self.paused != bool(self.state.get("paused")):
            self.log("paused" if self.paused else "resumed")
        self.state["paused"] = self.paused

    # ---------------------------------------------------------------- RESOLVE
    def reset_stage(self, name, why):
        """a new round with fresh attempt counters. Jobs of the stage that are still in squeue are carried into the
        new round (their end is judged there), so a retry never submits a second copy of a unit that still runs."""
        st = self.st(name)
        st.update(status="", why=why, decision=None, t_done=None, noprog=0, prog_epoch=None, att=dict(cont=0, infra=0),
                  halt_kind=None, sbatch_err=0)
        st["epoch"] = int(st.get("epoch") or 0) + 1
        live = getattr(self, "live", {})
        for us in st["units"].values():
            us.update(att=dict(cont=0, infra=0, oom=0, code=0), noprog=0)
            for j in us.get("jobs", []):
                if j["id"] in live:
                    j["epoch"], j["carried"] = st["epoch"], True
        if name in self.state["halted"]:
            self.state["halted"].remove(name)
        self.log(f"{name} reset ({why})")

    def rename_stop(self, f, name):
        """STOP_<name>.md -> STOP_<name>.resolved[.k].md (a rename, never a deletion); the old tracked path is
        committed as removed with the next commit."""
        k = 1
        tgt = f.with_name(f"STOP_{name}.resolved.md")
        while tgt.exists():
            k += 1
            tgt = f.with_name(f"STOP_{name}.resolved.{k}.md")
        os.replace(f, tgt)
        if str(f) not in self.state["stop_removed"]:
            self.state["stop_removed"].append(str(f))
        return tgt

    def auto_resolve(self, S, st, why):
        """a cancelled / operational HALT whose outputs became complete: the STOP file is marked resolved."""
        f = self.c.res / f"STOP_{S.name}.md"
        self.dec(S.name, f"자동 해제: {why} → `STOP_{S.name}.md`를 .resolved.md로 바꿨다.", key=f"autores:{st.get('epoch')}:{why}")
        self.log(f"{S.name} auto-resolved ({why})")
        if S.name in self.state["halted"]:
            self.state["halted"].remove(S.name)
        if not self.dry and f.is_file():
            self.rename_stop(f, S.name)
        self.to_commit.append((S.name, "auto-resolved"))

    def resolve_stops(self):
        """first line `RESOLVE: <option>` of results/STOP_<stage>.md. Crash-safe order: apply to the state, record the
        file hash in state.resolved, DECISIONS line (idempotent id), write the state, THEN rename the file. A file
        whose hash is already recorded is only renamed (never applied twice)."""
        for f in sorted(self.c.res.glob("STOP_*.md")):
            if ".resolved" in f.name or f.name == "STOP_S.md":
                continue
            try:
                txt = f.read_text()
            except OSError:
                continue
            first = txt.split("\n", 1)[0].strip()
            m = re.match(r"RESOLVE:\s*(\w+)\s*(.*)$", first)
            if not m:
                continue
            name = f.name[len("STOP_"):-len(".md")]
            if name not in self.by:
                self.attn.append(f"{f.name}: unknown stage")
                continue
            h = hashlib.sha1((f.name + "\n" + txt).encode()).hexdigest()[:12]
            opt, arg = m.group(1).lower(), m.group(2).strip()
            if h in self.state["resolved"]:
                if not self.dry:
                    tgt = self.rename_stop(f, name)
                    self.state["resolved"][h]["renamed"] = tgt.name
                    self.log(f"{name}: RESOLVE already applied (hash {h}); {f.name} -> {tgt.name}")
                self.to_commit.append((name, f"resolve {opt}"))
                continue
            st = self.st(name)
            if opt == "param":
                targets = set()
                for kv in arg.split():
                    k, _, v = kv.partition("=")
                    tgt, _, k2 = k.rpartition(".")
                    tgt = tgt or name
                    if tgt not in self.by:
                        continue
                    try:
                        val = int(v)
                    except ValueError:
                        try:
                            val = float(v)
                        except ValueError:
                            val = v
                    self.st(tgt).setdefault("params", {})[k2] = val
                    targets.add(tgt)
                for t in targets - {name}:
                    self.reset_stage(t, f"RESOLVE param from {f.name}")
                if name in targets or not targets:
                    self.reset_stage(name, f"RESOLVE param {arg}")       # retry this stage with the new parameter
                else:                                                    # continue past this STOP with another stage's parameter
                    st.update(status="done", why=f"accepted with {arg} (RESOLVE)", accepted=self.now, t_done=self.now)
                    if name in self.state["halted"]:
                        self.state["halted"].remove(name)
            elif opt == "retry":
                self.reset_stage(name, "RESOLVE retry")
            elif opt == "accept":
                st.update(status="done", why="accepted by lead (RESOLVE)", accepted=self.now, t_done=self.now)
                if name in self.state["halted"]:
                    self.state["halted"].remove(name)
            elif opt == "skip":
                st.update(status="skipped", why="skipped by lead (RESOLVE)", t_done=self.now)
                if name in self.state["halted"]:
                    self.state["halted"].remove(name)
            else:
                self.attn.append(f"{f.name}: unknown RESOLVE option {opt!r}")
                continue
            self.state["resolved"][h] = dict(file=f.name, stage=name, opt=opt, arg=arg, t=self.now, renamed=None)
            self.dec(name, f"lead `RESOLVE: {opt}{(' ' + arg) if arg else ''}` 적용 ({f.name} → .resolved.md).",
                     key=f"resolve:{h}")
            self.log(f"{name} RESOLVE {opt} {arg}".strip())
            if not self.dry:
                self.write_state()                                   # persisted before the file disappears
                tgt = self.rename_stop(f, name)
                self.state["resolved"][h]["renamed"] = tgt.name
            self.to_commit.append((name, f"resolve {opt}"))

    # ---------------------------------------------------------------- jobs index / intents / adoption / ends
    def all_jobs(self):
        for name, st in self.state["stages"].items():
            for key, us in st.get("units", {}).items():
                for j in us.get("jobs", []):
                    yield name, key, us, j

    def index_jobs(self):
        self.jobidx = {j["id"]: (n, k) for n, k, _, j in self.all_jobs()}
        self.masters = {j["id"].split("_")[0] for _, _, _, j in self.all_jobs()}

    def attach(self, stage, key, jid, intent, t_sub, adopted=False, name=None, end=None):
        st = self.st(stage)
        us = st["units"].setdefault(key, new_unit_state())
        if any(j["id"] == jid for j in us["jobs"]):
            return
        us["jobs"].append(dict(id=jid, intent=intent, t_sub=t_sub, epoch=int(st.get("epoch") or 0), adopted=adopted,
                               name=name, end=end))
        self.jobidx[jid] = (stage, key)
        self.masters.add(jid.split("_")[0])
        st["started"] = True

    def reconcile_intents(self):
        """an intent left 'submitting' by an interrupted pass / a timed-out sbatch: matched by its comment in squeue
        (accounting stores no comments here, AccountingStoreFlags=(null)), else by name + submit time + the same
        array index set in sacct; sacct failing keeps the intent (its units wait) — never voided on a failed query."""
        for it in self.state["intents"]:
            if it.get("status") != "submitting":
                continue
            m = [l for l in self.sq if l["comment"].endswith(":" + it["id"])]
            if not m:
                rows = self.c.slurm.sacct(names=[f"mltp_{it['stage']}"], since=it["t"][:19])
                if rows is None:
                    self.attn.append(f"sacct failed: intent {it['id']} ({it['stage']}) kept, its units wait")
                    continue
                m = [r for r in rows if (r.get("comment") or "").endswith(":" + it["id"])]
                if not m and rows and not any(r.get("comment") for r in rows):     # comments not in accounting
                    want = {re.fullmatch(r".+\[(\d+)\]", k).group(1) for k in it.get("units", [])
                            if it.get("array") and re.fullmatch(r".+\[(\d+)\]", k)}
                    by_master = {}
                    for r in rows:
                        if r["id"].split("_")[0] in self.masters or (r.get("submit") or "") < it["t"][:19]:
                            continue
                        for x in expand_ids(r["id"]):
                            by_master.setdefault(x.split("_")[0], set()).add(x.split("_", 1)[1] if "_" in x else "")
                    ok = sorted(ms for ms, idx in by_master.items()
                                if (idx == want if it.get("array") else idx == {""}))
                    m = [dict(id=ok[0])] if ok else []
            if m:
                master = m[0]["id"].split("_")[0]
                for key in it["units"]:
                    idx = re.fullmatch(r".+\[(\d+)\]", key)
                    jid = f"{master}_{idx.group(1)}" if (it.get("array") and idx) else master
                    self.attach(it["stage"], key, jid, it["id"], it["t"])
                it.update(status="adopted", job=master)
                self.log(f"{it['stage']} intent {it['id']} -> job {master} (adopted after an interrupted pass)")
            elif self.t - (parse_iso(it["t"]) or self.t) > INTENT_VOID_S:
                it["status"] = "void"
                self.log(f"{it['stage']} intent {it['id']} void (no job found)")
        if len(self.state["intents"]) > 300:
            keep = [x for x in self.state["intents"] if x.get("status") == "submitting"]
            self.state["intents"] = (self.state["intents"][-300:] if not keep else
                                     [x for x in self.state["intents"][-300:] if x not in keep] + keep)
        self.unresolved = {(it["stage"], k) for it in self.state["intents"] if it.get("status") == "submitting"
                           for k in it.get("units", [])}

    def match(self, name, jid, comment):
        """(stage, unit key | None) for a job name / id / comment, or None."""
        cm = parse_comment(comment)
        if cm and cm[0] in self.by:
            ks = spec_keys(cm[1], jid)
            return cm[0], (ks[0] if len(ks) == 1 else None)
        if cm and cm[0] == "nur" and "N-scan" in self.by:     # inv_nurion pending's manual line mltp:nur:<tar>:manual
            return "N-scan", f"scan/{nur_short(cm[1])}"
        idx = str(jid).split("_", 1)[1] if "_" in str(jid) else "0"
        for S in self.reg:
            for rx, key in S.adopt:
                mm = re.fullmatch(rx, name or "")
                if mm:
                    if key is None:                          # recognised as this stage's, unit unknown: blocks it
                        return S.name, None
                    k = key(mm, idx) if callable(key) else key.format(i=idx)
                    return S.name, k
        if (name or "").startswith("mltp_") and name[5:] in self.by:
            return name[5:], None
        return None

    def adopt(self):
        self.foreign = {}
        for l in self.sq:
            if l["name"] == LAUNCH or l["id"] in self.jobidx:
                continue
            hit = self.match(l["name"], l["id"], l["comment"])
            if not hit:
                continue
            stage, key = hit
            if key is None:
                self.foreign.setdefault(stage, []).append(l["id"])
                continue
            self.attach(stage, key, l["id"], "adopt", self.now, adopted=True, name=l["name"])
            self.log(f"{stage} adopt {l['id']} ({l['name']}) as {key}")
        # every pass: sacct since the last scan (- 15 min), so a manual job that started and ended between two passes
        # is adopted too (newer than the unit's latest job only)
        last = parse_iso(self.state.get("hist_scan")) if self.state.get("hist_scan") else None
        since = iso(last - HISTORY_MARGIN_S) if last else HISTORY_SINCE
        rows = self.c.slurm.sacct(since=since)
        if rows is None:
            self.attn.append("sacct failed: manual-job history not scanned this pass")
            return
        best = {}
        for r in rows:
            for jid in expand_ids(r["id"]):
                if jid in self.jobidx or jid in self.live:
                    continue
                hit = self.match(r["name"], jid, r.get("comment"))
                if not hit or hit[1] is None:
                    continue
                k = (hit[0], hit[1])
                if k not in best or (r.get("submit") or "", jid) > (best[k][0].get("submit") or "", best[k][1]):
                    best[k] = (r, jid)
        for (stage, key), (r, jid) in sorted(best.items()):
            us = self.st(stage)["units"].get(key)
            latest = max([str(j.get("t_sub") or "") for j in (us or {}).get("jobs", [])] or [""])
            if us and us.get("jobs") and (r.get("submit") or "") <= latest:
                continue
            end = None
            if r["state"].split()[0] not in LIVE_STATES:
                end = dict(state=r["state"], exit=r["exit"], sig=r.get("sig"), elapsed=r["elapsed"], start=r["start"],
                           end=r["end"])
            self.attach(stage, key, jid, "adopt-history", r.get("submit") or self.now, adopted=True, name=r["name"], end=end)
            self.log(f"{stage} adopt {jid} ({r['name']}, {r['state']}) from sacct history as {key}")
        if not self.dry:
            self.state["hist_scan"] = self.now

    def fetch_ends(self):
        need = [j for _, _, _, j in self.all_jobs() if j.get("end") is None and j["id"] not in self.live]
        if not need:
            return
        rows = self.c.slurm.sacct(ids=sorted({j["id"] for j in need}))
        if rows is None:
            self.attn.append("sacct failed: job end states deferred")
            return
        by = {}
        for r in rows:
            for x in expand_ids(r["id"]):
                by[x] = r
        if not by:
            self.attn.append(f"sacct returned no rows for {len(need)} ended jobs (accounting delay?): end states deferred")
        for j in need:
            r = by.get(j["id"])
            age = self.t - (parse_iso(j.get("t_sub")) or self.t)
            if r and r["state"].split()[0] not in LIVE_STATES:
                j["end"] = dict(state=r["state"], exit=r["exit"], sig=r.get("sig"), elapsed=r["elapsed"],
                                start=r["start"], end=r["end"])
            elif r is None and ((by and age > LOST_AFTER_S) or age > LOST_ALONE_S):
                # LOST only when accounting answers for other jobs (or after 12 h): an empty answer is an outage
                j["end"] = dict(state="LOST", exit=None, sig=None, elapsed=None, start=None, end=None)

    # ---------------------------------------------------------------- quota / successor
    def quota(self):
        f = self.c.ctl / "QUOTA_FREE_GB"
        q = dict(gb=None, t=None, age_h=None, stale=True, est_free=None, used_since=0.0)
        try:
            tok = f.read_text().split()
            q["gb"], q["t"] = float(tok[0]), tok[1]
            tq = parse_iso(tok[1])
            q["age_h"] = round((self.t - tq) / 3600.0, 2) if tq else None
            q["stale"] = tq is None or q["age_h"] > QUOTA_STALE_H
        except (OSError, IndexError, ValueError):
            pass
        if q["gb"] is not None and q["t"]:
            # every job that could still write after the reading: submitted after it, live now, or ended after it
            # (a job running at the reading time keeps writing); conservative (its whole estimate is counted)
            used, tq = 0.0, q["t"][:19]
            for _, _, _, j in self.all_jobs():
                d = float(j.get("disk_gb") if j.get("disk_gb") is not None else DISK_GB_DEFAULT)
                if d <= 0:
                    continue
                e = (j.get("end") or {}).get("end")
                if (j.get("t_sub") or "") >= tq or j["id"] in self.live or (e and str(e)[:19] >= tq):
                    used += d
            q["used_since"] = round(used, 2)
            q["est_free"] = round(q["gb"] - used, 2)
        self.state["quota"] = q
        self.no_submit = None
        if q["gb"] is None or q["stale"]:
            self.no_submit = "QUOTA_FREE_GB missing or > 24 h old (DECISIONS 0-8: free disk checked before submitting)"
            self.attn.append("QUOTA_FREE_GB missing or > 24 h old: NO submissions — lead: write '<GB> <ISO time>' from login "
                             "mmlsquota to $MLT/QUOTA_FREE_GB (refresh at the hourly check)")
        elif q["est_free"] is not None and q["est_free"] < QUOTA_MIN_GB:
            self.no_submit = f"estimated free disk {q['est_free']:.0f} GB < 30 GB"
            self.attn.append(f"estimated free disk {q['est_free']:.0f} GB < 30 GB: no submissions (DECISIONS 0-8)")

    def successor(self):
        if not self.c.launcher or self.dry or self.pause_exit or self.state.get("finished"):
            return
        if self.succ_pending:
            return
        rc, out, err = self.c.slurm.sbatch(["--begin=now+10minutes", "jobs/pipeline.slurm"], cwd=str(self.c.code),
                                           env=child_env(self.c.env))
        if rc == 0:
            self.log(f"successor launcher {out} (bash resubmission had failed)")
            self.CPU += 1
            self.total += 1
            self.succ_pending = True
        else:
            self.successor_ok = False
            self.attn.append(f"launcher successor could not be queued ({err[:120]}): lead — `sbatch jobs/pipeline.slurm`")

    # ---------------------------------------------------------------- per stage
    def safe(self, fn, *a):
        try:
            return fn(self, *a)
        except Exception:                                                   # noqa: BLE001
            return False

    def progress(self, S):
        if S.progress is None:
            return None
        if S.name not in self._prog:
            try:
                self._prog[S.name] = S.progress(self)
            except Exception:                                               # noqa: BLE001
                self._prog[S.name] = None
        return self._prog[S.name]

    def file_ready(self, name):
        """(ok, why) for one analysis/mlts file: present, tracked, equal to HEAD (cached per pass)."""
        f = self.c.code / name
        if not f.is_file():
            return False, f"{name} missing"
        if name not in self._git:
            self._git[name] = (self.c.git.tracked(f), self.c.git.clean(f))
        tr, cl = self._git[name]
        if not tr:
            return False, f"{name} not committed"
        if not cl:
            return False, f"{name} modified since HEAD (being edited)"
        return True, ""

    def closure(self, scripts):
        k = tuple(sorted(set(scripts)))
        if k not in self._code:
            self._code[k] = local_closure(self.c.code, list(k))
        return self._code[k]

    def code_ready(self, S):
        """every registered (script, sub-command) present, and the script + every analysis/mlts module it imports +
        mlt_common.py tracked and equal to HEAD."""
        for script, sub, extra in S.code:
            ok, why = self.file_ready(script)
            if not ok:
                return False, why
            try:
                txt = (self.c.code / script).read_text()
            except OSError:
                txt = ""
            if sub and not token_re(sub).search(txt):
                return False, f"{script} has no '{sub}' command"
            if extra and not re.search(extra, txt):
                return False, f"{script} '{sub}' lacks {extra}"
        for name in self.closure([s for s, _, _ in S.code]):
            ok, why = self.file_ready(name)
            if not ok:
                return False, f"{why} (imported)"
        return True, ""

    def stage_live(self, S):
        ids = set()
        for us in self.st(S.name)["units"].values():
            ids |= {j["id"] for j in us["jobs"]}
        out = []
        for l in self.sq:
            if l["name"] == LAUNCH:
                continue
            if l["id"] in ids or l["name"] == f"mltp_{S.name}" or (parse_comment(l["comment"]) or ("",))[0] == S.name:
                out.append(l)
                continue
            if any(re.fullmatch(rx, l["name"]) for rx, _ in S.adopt):
                out.append(l)
        return out

    def eval_stage(self, S, again=False):
        st = self.st(S.name)
        st.pop("_planned", None)
        if st.get("status") in ("done", "skipped") and st.get("sticky", True):
            return
        live = self.stage_live(S)
        if st.get("status") in ("halted", "cancelled"):
            # an operational stop (a lead-cancelled unit, retries used up) clears itself once nothing of the stage
            # runs and its outputs are complete (prerequisites done); a decision HALT (spec STOP) waits for RESOLVE
            if (st.get("status") == "cancelled" or st.get("halt_kind") == "ops") and not live and \
                    S.done is not None and self.after_ready(S)[0] and self.safe(S.done):
                self.auto_resolve(S, st, "산출물이 모두 끝남")
                self.mark_done(S, st)
            return
        if S.guard:
            g = self.safe_guard(S)
            if g:
                self.halt(S, st, g)
                return
        after_ok, after_why = self.after_ready(S)
        if S.gate is not None:
            if not after_ok:
                self.set_status(S, st, "blocked", after_why)
                return
            r, why, nums = S.gate(self)
            if r == "pass":
                st["decision"] = dict(key="pass", numbers=nums)
                self.mark_done(S, st)
            elif r == "fail":
                self.halt(S, st, why, numbers=nums, options=[
                    "지울 대상을 사용자에게 묻는다(DECISIONS 0-8: 삭제는 사용자 승인) — 지운 뒤 `retry`",
                    "`accept` — 여유가 모자란 채로 A-4를 낸다(권하지 않음)"])
            else:
                self.set_status(S, st, "blocked", why)
            return
        if after_ok and S.active is not None and not live and not self.safe(S.active):
            st.update(status="skipped", why="optional branch not taken (its condition is false)", t_done=self.now)
            self.log(f"{S.name}: skipped (optional branch not taken)")
            return
        # outputs complete count only once the prerequisites are done (never mark a stage done ahead of its inputs)
        if after_ok and not live and S.done is not None and self.safe(S.done):
            self.mark_done(S, st)
            return
        code_ok, code_why = self.code_ready(S)
        need_ok, need_why = True, ""
        for w, fn in S.need:
            if not self.safe(fn):
                need_ok, need_why = False, f"needs {w}"
                break
        if not (code_ok and after_ok and need_ok):
            self.exit3_scan(S, st)
            if st.get("status") in ("halted", "done"):
                return
            why = code_why or after_why or need_why
            if live:
                self.set_status(S, st, "running" if any(l["state"] == "RUNNING" for l in live) else "queued",
                                f"{len(live)} live ({why})")
            else:
                self.set_status(S, st, "waiting_code" if not code_ok else "blocked",
                                f"waiting: code — {why}" if not code_ok else why)
            return
        try:
            units = S.units(self, st)
        except Exception as e:                                              # noqa: BLE001
            self.set_status(S, st, "blocked", f"units(): {type(e).__name__}: {e}"[:200])
            return
        ep = int(st.get("epoch") or 0)
        res = {}
        for u in units:
            us = st["units"].setdefault(u.key, new_unit_state())
            for k, v in new_unit_state().items():
                us.setdefault(k, v)
            if us.get("epoch") != ep:
                us.update(epoch=ep, att=dict(cont=0, infra=0, oom=0, code=0), noprog=0)
            res[u.key] = self.eval_unit(S, st, u, us, res, ep)
            if st.get("status") in ("halted", "cancelled"):
                return
        if any(v == "halt" for v in res.values()):
            return
        foreign = self.foreign.get(S.name) or []
        if not units:
            if live:
                self.set_status(S, st, "running", f"{len(live)} live")
            else:
                self.set_status(S, st, "idle", "nothing to run now")
            return
        canc = [k for k, v in res.items() if v == "cancelled"]
        if all(v in ("ok", "cancelled") for v in res.values()):
            if live or foreign:
                self.set_status(S, st, "running", f"{len(live)} live (not ours)")
                return
            if S.done is None or self.safe(S.done):
                self.mark_done(S, st)
                return
            if canc:                                       # the lead cancelled a unit and the outputs are incomplete
                self.cancel_stop(S, st, canc)
                return
            if S.rearm:                                     # e.g. H-early: this input set is finished, wait for more
                tag = (st.get("params") or {}).get("cur_tag")
                st.setdefault("params", {}).setdefault("done_tags", [])
                if tag and tag not in st["params"]["done_tags"]:
                    st["params"]["done_tags"].append(tag)
                    self.log(f"{S.name}: {tag} finished; waiting for more inputs")
                self.set_status(S, st, "idle", f"{tag} finished; waiting for more inputs")
                return
            if S.commit_gate:
                if (S.name, "commit gate") not in self.to_commit:
                    self.to_commit.append((S.name, "commit gate"))
                self.set_status(S, st, "committing", "outputs ready; the launcher commits them")
                return
            self.epoch_end(S, st)
            if st.get("status") != "halted" and not again:
                self.eval_stage(S, again=True)
            return
        cands = []
        if not foreign:
            for u in units:
                if res[u.key] != "todo":
                    continue
                deps, ok = [], True
                for d in u.deps:
                    r = res.get(d)
                    if r in ("ok", "cancelled"):                # afterany: a cancelled input does not block
                        continue
                    if r == "live:R":
                        deps += [j["id"] for j in st["units"][d]["jobs"] if j["id"] in self.live]
                        continue
                    ok = False
                    break
                if ok:
                    cands.append((u, sorted(set(deps))))
        nl = sum(1 for v in res.values() if v.startswith("live"))
        nok = sum(1 for v in res.values() if v == "ok")
        summ = f"units {nok}/{len(units)} ok, {nl} live" + (f", {len(cands)} ready" if cands else "") + \
            (f", cancelled {canc}" if canc else "")
        if foreign:
            summ += f"; foreign live {foreign} (no submissions)"
        if cands:
            self.cands += [(S, u, d) for u, d in cands]
        status = ("running" if any(v == "live:R" for v in res.values()) else "queued" if nl else
                  "ready" if cands else "waiting")
        self.set_status(S, st, status, summ)

    def safe_guard(self, S):
        try:
            return S.guard(self)
        except Exception:                                                   # noqa: BLE001
            return None

    def set_status(self, S, st, status, why):
        if st.get("status") != status:
            if status not in ("waiting", "queued", "running") or st.get("status") in ("", "blocked", "waiting_code", "ready"):
                self.log(f"{S.name}: {st.get('status') or 'new'} -> {status}")
        st["status"], st["why"] = status, why

    def after_ready(self, S):
        for a in S.after:
            s = self.st(a).get("status")
            if s not in ("done", "skipped"):
                return False, f"after {a} ({s or 'not started'})"
        return True, ""

    def exit3_scan(self, S, st):
        """jobs of a stage that cannot act now: only an exit 3 (STOP) is handled immediately."""
        for key, us in st["units"].items():
            for j in us["jobs"]:
                if j.get("end") and not j.get("handled") and classify(j["end"]) == "stop":
                    j["handled"] = "stop"
                    self.on_exit3(S, st, key, j)
                    return

    def ok_job(self, st, key, ep):
        js = [j for j in st["units"].get(key, {}).get("jobs", []) if j.get("epoch", 0) == ep and j.get("handled") == "ok"]
        return js[-1] if js else None

    def eval_unit(self, S, st, u, us, res, ep):
        # live in ANY round (a RESOLVE retry / new round never resubmits a unit whose earlier job still runs)
        lv = [j for j in us["jobs"] if j["id"] in self.live]
        if lv:
            return "live:R" if any(self.live[j["id"]]["state"] == "RUNNING" for j in lv) else "live:PD"
        jobs = [j for j in us["jobs"] if j.get("epoch", 0) == ep and not j.get("void")]
        if us.get("ok") == ep:
            return "ok"
        if (S.name, u.key) in self.unresolved:
            return "wait"                                   # an interrupted sbatch: never submit twice (void after 2 min)
        if u.done is not None and not u.sink and self.safe(u.done):
            us["ok"] = ep
            if jobs and not jobs[-1].get("handled"):
                jobs[-1]["handled"] = "ok"
            return "ok"
        if not jobs:
            return "todo"
        j = jobs[-1]
        h = j.get("handled")
        if h:
            return {"retry": "todo", "ok": "ok", "void": "todo", "cancelled": "cancelled"}.get(h, "halt")
        end = j.get("end")
        if end is None:
            return "wait"
        cls = classify(end)
        if cls not in ("stop", "cancelled") and u.deps:
            js = parse_iso(end.get("start"))
            stale = False
            for d in u.deps:
                if res.get(d) not in ("ok", "cancelled"):
                    stale = True
                    break
                dj = self.ok_job(st, d, ep)
                de = parse_iso(((dj or {}).get("end") or {}).get("end")) if dj else None
                if de and js and de > js + 1:
                    stale = True
                    break
            if stale:
                j["handled"], j["void"] = "void", True
                self.log(f"{S.name} {u.key}: {j['id']} ran before its inputs were final -> again")
                return "todo"
        if cls == "ok" and u.done is not None and not u.sink and not self.safe(u.done):
            cls = "cont"                                                    # exit 0 but its outputs are incomplete
        if u.sink and cls in ("ok", "cont"):
            j["handled"], us["ok"] = "ok", ep
            return "ok"
        if cls == "ok":
            j["handled"], us["ok"] = "ok", ep
            return "ok"
        if cls == "stop":
            j["handled"] = "stop"
            self.on_exit3(S, st, u.key, j)
            return "halt"
        if cls == "cancelled":
            # never resubmitted (the lead decided); counts as finished for afterany dependants; the stage is done if
            # its outputs are complete, else it stops and asks (cancel_stop)
            j["handled"] = "cancelled"
            self.log(f"{S.name} {u.key}: {j['id']} CANCELLED (lead) -> not resubmitted")
            return "cancelled"
        att = us["att"]
        what = f"{end.get('state')} exit {end.get('exit')}"
        if cls == "code":
            att["code"] = att.get("code", 0) + 1
            if att["code"] > RETRY["code"]:
                j["handled"] = "halt"
                self.halt(S, st, f"{u.key}: the job-start guard refused the code {att['code']} times (exit 4: a script "
                                 f"changed after submission)", numbers=dict(job=j["id"], unit=u.key), kind="ops")
                return "halt"
            j["handled"] = "retry"
            self.log(f"{S.name} {u.key}: {j['id']} guard exit 4 (code changed after submission) -> resubmit when committed")
            return "todo"
        if cls == "cont":
            att["cont"] = att.get("cont", 0) + 1
            prog = self.progress(S)
            if prog is not None and us.get("prog_sub") is not None and prog <= us["prog_sub"]:
                us["noprog"] = us.get("noprog", 0) + 1
            else:
                us["noprog"] = 0
            if att["cont"] > RETRY["cont"] or us["noprog"] >= RETRY["noprog"]:
                j["handled"] = "halt"
                self.halt(S, st, f"{u.key}: {what} — 재제출 {att['cont'] - 1}회, 진전 없는 재제출 {us['noprog']}회 연속",
                          numbers=dict(job=j["id"], unit=u.key, progress=prog, progress_at_submit=us.get("prog_sub"),
                                       attempts=att), kind="ops")
                return "halt"
        elif cls == "oom":
            att["oom"] = att.get("oom", 0) + 1
            if att["oom"] > RETRY["oom"]:
                j["handled"] = "halt"
                self.halt(S, st, f"{u.key}: OUT_OF_MEMORY twice (mem x {us.get('mem_mult')})", numbers=dict(job=j["id"]),
                          kind="ops")
                return "halt"
            us["mem_mult"] = round(float(us.get("mem_mult") or 1.0) * 1.5, 3)
        else:
            att["infra"] = att.get("infra", 0) + 1
            if att["infra"] > RETRY["infra"]:
                j["handled"] = "halt"
                self.halt(S, st, f"{u.key}: {what} again (infra / exit 1 retry used)", numbers=dict(job=j["id"], end=end),
                          kind="ops")
                return "halt"
        j["handled"] = "retry"
        self.log(f"{S.name} {u.key}: {j['id']} {what} -> resubmit ({cls})")
        return "todo"

    def cancel_stop(self, S, st, keys):
        st.update(status="cancelled", why=f"{', '.join(keys)}: CANCELLED (lead), outputs incomplete — not resubmitted",
                  halt_kind="ops")
        self.write_stop(S.name, st["why"], numbers=dict(units=keys),
                        options=["`retry` — 새 라운드로 다시 낸다(취소한 작업 포함)", "`accept` — 이대로 끝난 것으로 본다",
                                 "`skip` — 이 단계를 건너뛴다"],
                        recommended="취소한 사람(lead)이 고른다. 산출물이 나중에 다 차면(예: 손으로 낸 job) 다음 pass가 저절로 푼다.")
        self.log(f"{S.name}: cancelled unit(s) {keys}, outputs incomplete -> STOP_{S.name}.md")
        self.to_commit.append((S.name, "cancelled"))

    def epoch_end(self, S, st):
        att = st.setdefault("att", dict(cont=0, infra=0))
        att["cont"] = att.get("cont", 0) + 1
        prog = self.progress(S)
        if prog is not None and st.get("prog_epoch") is not None and prog <= st["prog_epoch"]:
            st["noprog"] = st.get("noprog", 0) + 1
        else:
            st["noprog"] = 0
        if att["cont"] > RETRY["cont"] or st["noprog"] >= RETRY["noprog"]:
            if S.partial is not None:
                try:
                    ok, txt = S.partial(self)
                except Exception as e:                                      # noqa: BLE001
                    ok, txt = False, f"partial(): {type(e).__name__}: {e}"
                if ok:
                    st["partial"] = txt
                    self.dec(S.name, txt, key=f"partial:{st.get('epoch')}")
                    self.mark_done(S, st)
                    return
            self.halt(S, st, f"단계 산출물이 끝나지 않음: 라운드 {att['cont']}회, 진전 없는 라운드 {st['noprog']}회 연속",
                      numbers=dict(progress=prog, progress_at_round_start=st.get("prog_epoch"), epoch=st.get("epoch")),
                      kind="ops")
            return
        st["epoch"] = int(st.get("epoch") or 0) + 1
        st["prog_epoch"] = prog
        self.log(f"{S.name}: outputs incomplete -> round {st['epoch']}")
        st["status"] = "ready"

    def on_exit3(self, S, st, key, j):
        if S.on_exit3 == "decide" and S.done is not None and self.safe(S.done):
            self.log(f"{S.name} {key}: exit 3 with its report written -> decision rule")
            self.mark_done(S, st)
            return
        self.halt(S, st, f"{key}: job {j['id']} exit 3 (STOP) — log {self.c.mlt}/logs/mltp_{S.name}_*{j['id'].split('_')[0]}*",
                  numbers=dict(job=j["id"], unit=key, end=j.get("end")))

    def mark_done(self, S, st):
        st.update(status="done", why="", t_done=self.now, halt_kind=None)
        self.log(f"{S.name} done")
        r = {}
        if S.decide:
            try:
                r = S.decide(self, S, st) or {}
            except Exception as e:                                          # noqa: BLE001
                r = dict(key="decide error", halt=f"decision rule failed: {type(e).__name__}: {e}"[:300])
            st["decision"] = {k: v for k, v in r.items() if k in ("key", "numbers", "line", "halt")}
            if r.get("line"):
                self.dec(S.name, r["line"], key=f"done:{st.get('epoch')}:{r.get('key')}")
        if r.get("halt"):
            self.halt(S, st, r["halt"], numbers=r.get("numbers"), options=r.get("options"), recommended=r.get("recommended"))
        else:
            self.to_commit.append((S.name, f"done ({r.get('key') or 'complete'})"))
        for name in r.get("reset", []):
            self.reset_stage(name, f"decision of {S.name}")

    def halt(self, S, st, why, numbers=None, options=None, recommended=None, kind="decision"):
        """kind 'decision' = a spec STOP / decision rule / exit 3 (waits for RESOLVE); 'ops' = retries used up
        (also clears itself once the outputs are complete). Scope: a filler (report-only) stage's HALT stops only its
        dependants; any other HALT also stops new stages from starting."""
        st.update(status="halted", why=why, t_halt=self.now, halt_kind=kind)
        if S.name not in self.state["halted"]:
            self.state["halted"].append(S.name)
        self.write_stop(S.name, why, numbers, options, recommended, scope=S.halt_scope)
        eff = ("이 단계와 이에 의존하는 단계는 RESOLVE까지 멈추고, HALT 동안 새 단계도 시작하지 않는다." if S.halt_scope == "global"
               else "보고용(filler) 단계라 이 단계에 의존하는 단계만 멈추고 다른 단계는 계속한다.")
        self.dec(S.name, f"**HALT** — {why}. `results/STOP_{S.name}.md`에 조건 · 수치 · 선택지를 적었다. 정해진 추천안이 "
                         f"없어 {eff}", key=f"halt:{st.get('epoch')}:{why[:160]}")
        self.log(f"{S.name} HALT: {why}"[:300])
        self.to_commit.append((S.name, "halted"))

    def write_stop(self, name, why, numbers=None, options=None, recommended=None, scope="global"):
        L = [f"# STOP: {name} ({self.now}, pipeline.py)", "", f"- **조건 / condition**: {why}"]
        if numbers:
            L.append("- **수치 / numbers**:")
            L += [f"  - {k}: {json.dumps(v, ensure_ascii=False, default=str)}" for k, v in numbers.items()]
        L.append(f"- **선택지 / options** — 이 파일 첫 줄에 `RESOLVE: <선택지>`를 쓰면 다음 pass가 적용하고 DECISIONS.md에 "
                 f"적은 뒤 이 파일을 `STOP_{name}.resolved.md`로 이름을 바꾼다(지우지 않음):")
        L += [f"  - {o}" for o in (options or DEFAULT_OPTIONS)]
        L.append(f"- **추천 / recommended**: {recommended or '정해진 추천안 없음 — 명령서 STOP이므로 사용자(lead)가 고른다.'}")
        L += [("- **영향 / effect**: 이 단계와 이 단계에 의존하는 단계는 시작하지 않고, HALT 동안 새 단계도 시작하지 않는다. "
               "이미 시작했고 이 단계에 의존하지 않는 단계는 계속한다." if scope == "global" else
               "- **영향 / effect**: 보고용(filler) 단계: 이 단계에 의존하는 단계만 시작하지 않는다. 다른 단계는 계속한다."), ""]
        if self.dry:
            self.planned.append(f"(dry) would write results/STOP_{name}.md: {why}")
            return
        M.write_text(self.c.res / f"STOP_{name}.md", "\n".join(L))

    # ---------------------------------------------------------------- git
    def commit(self):
        items = [tuple(x) for x in self.state.get("commit_pending") or []] + [x for x in self.to_commit]
        seen, items_u = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                items_u.append(x)
        items = items_u
        if not items and not self.state.get("push_pending"):
            return
        if self.dry:
            if items:
                self.planned.append("(dry) would commit + push: " + "; ".join(f"{n} {k}" for n, k in items))
            return
        g = self.c.git
        if items:
            br = g.branch()
            if br != "mlts":
                self.attn.append(f"git branch is {br!r}, not mlts: commits deferred")
                self.state["commit_pending"] = [list(x) for x in items]
                return
            if g.index_locked():
                age = g.index_lock_age()
                self.log("git index.lock present: commit next pass")
                if age is not None and age > 600:
                    self.attn.append(f".git index.lock is {age / 60:.0f} min old: commits deferred — lead: check for a "
                                     "running git, remove the stale lock from login if none")
                self.state["commit_pending"] = [list(x) for x in items]
                return
            res_root = (HERE / "results").resolve()
            if not getattr(g, "fake", False) and self.c.res.resolve() != res_root:
                self.attn.append(f"results root {self.c.res} is not analysis/mlts/results: no commit")
                self.state["commit_pending"] = []
                return
            paths = set()
            for name, _ in items:
                S = self.by.get(name)
                for pat in (S.results if S else []):
                    paths |= {p for p in self.c.res.glob(pat) if p.is_file()}
            paths |= {p for p in self.c.res.glob("STOP_*.md") if p.is_file()}
            # STOP files renamed to .resolved.md: commit the removal of the old tracked path (never a data deletion)
            gone = [Path(p) for p in self.state.get("stop_removed") or [] if not Path(p).exists() and g.tracked(p)]
            self.state["commit_pending"] = []
            self.flush_log()
            self.write_state()
            paths |= {p for p in (self.c.state_path, self.c.log_path, self.c.dec_path) if p.is_file()}
            root = str(self.c.res.resolve()) + os.sep
            paths = sorted(str(p) for p in paths if str(p.resolve()).startswith(root))
            gone = sorted(str(p) for p in gone if str(p.resolve()).startswith(root))
            title = ("MLTS pipeline: " + "; ".join(f"{n} {k}" for n, k in items))[:200]
            g.add(paths + gone)
            rc, out = g.commit(paths + gone, title, TRAILER)
            if rc == 0:
                sha = g.head()
                for name, _ in items:
                    if name in self.by:
                        self.st(name)["commit"] = dict(sha=sha, pushed=False)
                self.state["push_pending"] = True
                self.state["stop_removed"] = []
                self.log(f"commit {sha}: {title[15:80]}")
            elif "nothing to commit" in (out or "") or "no changes added" in (out or ""):
                self.log("commit: nothing to commit")
            else:
                self.state["commit_pending"] = [list(x) for x in items]
                self.attn.append(f"git commit failed: {(out or '').strip()[:160]}")
                return
        if self.state.get("push_pending"):
            rc, out = g.push()
            if rc == 0:
                self.state["push_pending"] = False
                for s in self.state["stages"].values():
                    if s.get("commit") and not s["commit"].get("pushed"):
                        s["commit"]["pushed"] = True
                self.log("push origin mlts ok")
            else:
                self.attn.append(f"git push pending ({(out or '').strip()[:120]}); retried next pass — lead may push from login")
                self.log("push failed (pending)")

    # ---------------------------------------------------------------- submission
    def stage_started(self, S):
        st = self.st(S.name)
        return bool(st.get("started")) or any(us.get("jobs") for us in st["units"].values())

    def room_queue(self):
        """places left under MaxSubmit: 20 - 1 spare (the lead) - every task now - the successor launcher if none is
        pending yet (its place is reserved so the chain can never be crowded out)."""
        succ = 0 if (self.succ_pending or self.pause_exit or self.state.get("finished")) else 1
        return SUBMIT_CAP - SPARE - self.total - succ

    def room_eligible(self, filler):
        """FIFO cluster: a task that is eligible now starts before anything submitted later, so eligible pending
        work is kept to the free running slots (+ PEND_BUF for critical work; filler only into free slots)."""
        free = MAX_RUN - (self.Rn - self.self_run) - self.EP
        return free if filler else free + PEND_BUF

    def submit_all(self):
        if not self.cands:
            return
        if self.paused:
            for S, _, _ in self.cands:
                self.st(S.name)["why"] += " | paused"
            return
        if self.state.get("finished"):
            return
        if not self.successor_ok:
            self.attn.append("no submissions this pass: launcher successor missing")
            return
        if getattr(self, "no_submit", None):
            for S, _, _ in self.cands:
                self.st(S.name)["why"] += f" | no submissions: {self.no_submit}"
            return
        halted = set(self.state.get("halted") or [])
        ghalt = {h for h in halted if h in self.by and self.by[h].halt_scope == "global"}
        order = sorted(range(len(self.cands)), key=lambda i: (self.cands[i][0].filler, self.cands[i][0].crit, i))
        groups = []
        for i in order:
            S, u, deps = self.cands[i]
            mm = float((self.st(S.name)["units"][u.key]).get("mem_mult") or 1.0)
            if u.akey is not None and groups and groups[-1][0] is S and groups[-1][1][0].akey == u.akey \
                    and groups[-1][3] == mm and groups[-1][2] == deps:
                groups[-1][1].append(u)
            else:
                groups.append((S, [u], deps, mm))
        for S, us, deps, mm in groups:
            if self.stop_submit or self.n_sub >= MAX_TASKS_PER_PASS:
                break
            st = self.st(S.name)
            # the status is left as computed (ready): only the reason changes, so a HALT adds no log line per pass
            if self.anc[S.name] & halted:
                st["why"] = f"depends on halted {sorted(self.anc[S.name] & halted)}"
                continue
            if ghalt and not self.stage_started(S):
                st["why"] = f"HALT ({', '.join(sorted(ghalt))}): no new stages"
                continue
            u0 = us[0]
            rq, re_ = self.room_queue(), (self.room_eligible(S.filler) if not deps else None)
            room = rq if re_ is None else min(rq, re_)
            if u0.c:
                room = min(room, (CPU_CAP - self.CPU) // max(1, u0.c))
            if u0.gpu:
                room = min(room, (GPU_CAP - self.GPU) // max(1, u0.gpu))
            room = min(room, MAX_TASKS_PER_PASS - self.n_sub)
            if room <= 0:
                if " | no room" not in st["why"]:
                    st["why"] += (f" | no room (queue {self.total}/{SUBMIT_CAP - SPARE}, running {self.Rn - self.self_run}/"
                                  f"{MAX_RUN}, eligible pending {self.EP}"
                                  f"{'' if deps else ' (filler: free slots only)' if S.filler else f' (+{PEND_BUF})'}, "
                                  f"CPU {self.CPU}/{CPU_CAP}, GPU {self.GPU}/{GPU_CAP})")
                continue
            if u0.akey is None:
                for u in us[:room]:
                    if self.stop_submit:
                        break
                    self.submit(S, [u], deps, mm)
            else:
                self.submit(S, us[:room], deps, mm)

    def guard_files(self, u):
        return self.closure(cmd_scripts(u.cmd))

    def sbatch_argv(self, S, us, deps, mm, iid):
        u0 = us[0]
        mem = max(1, int(math.ceil(u0.mem * mm)))
        arrv = u0.akey is not None
        spec = f"{u0.akey}[{'+'.join(str(u.idx) for u in us)}]" if arrv else u0.key
        logs = self.c.mlt / "logs"
        argv = ["-p", PART[u0.part], "-J", f"mltp_{S.name}", "--time=48:00:00", f"--cpus-per-task={u0.c}", f"--mem={mem}G"]
        if u0.gpu:
            argv.append(f"--gres=gpu:{u0.gpu}")
        if arrv:
            argv.append(f"--array={','.join(str(u.idx) for u in us)}")
        argv += [f"--output={logs}/mltp_{S.name}_" + ("%A_%a.out" if arrv else "%j.out"),
                 f"--comment=mltp:{S.name}:{spec}:{iid}"]
        if deps:
            argv.append(f"--dependency=afterany:{':'.join(deps)}")
        guard = ["pipeline.py", "guard"] + self.guard_files(u0) + [":::"]
        argv += [f"--export=ALL,MLTENV={u0.env}", "jobs/mlt.slurm"] + guard + u0.cmd
        return argv, mem

    def submit(self, S, us, deps, mm):
        st = self.st(S.name)
        iid = secrets.token_hex(4)
        argv, mem = self.sbatch_argv(S, us, deps, mm, iid)
        line = "sbatch --parsable " + shlex.join(argv)
        u0, n = us[0], len(us)
        if self.dry:
            self.planned.append(f"[{S.name}] {line}")
            self._count(S, us, bool(deps))
            return
        if S.on_start and not st.get("started"):
            try:
                txt = S.on_start(self, S, st)
            except Exception:                                               # noqa: BLE001
                txt = None
            if txt:
                self.dec(S.name, txt, key=f"start:{st.get('epoch')}")
        it = dict(id=iid, stage=S.name, units=[u.key for u in us], array=u0.akey is not None, t=self.now,
                  status="submitting", cmd=line)
        self.state["intents"].append(it)
        self.write_state()
        rc, out, err = self.c.slurm.sbatch(argv, cwd=str(self.c.code), env=child_env(self.c.env))
        if rc != 0:
            kind = sbatch_error_kind(rc, err)
            if kind == "transient":
                # the controller may have created the job: the intent stays 'submitting' (matched by its squeue
                # comment next pass, voided after 2 min if absent); nothing more this pass; never counted
                it["err"] = err[:300]
                self.stop_submit = "sbatch transient error"
                self._count(S, us, bool(deps))
                self.log(f"{S.name}: sbatch transient error ({err[:120]}) -> intent kept, no more submissions this pass")
            elif kind == "maxsubmit":
                it.update(status="failed", err=err[:300])
                self.stop_submit = "MaxSubmit"
                self.log(f"{S.name}: sbatch refused (MaxSubmit) -> no more submissions this pass")
            elif kind == "dependency":
                it.update(status="failed", err=err[:300])
                self.log(f"{S.name}: sbatch dependency error ({err[:120]}) -> recomputed next pass")
            else:
                it.update(status="failed", err=err[:300])
                pn = (self.state.get("pass") or {}).get("n")
                if st.get("sbatch_err_pass") != pn:                    # once per pass: "N passes in a row"
                    st["sbatch_err"] = int(st.get("sbatch_err") or 0) + 1
                    st["sbatch_err_pass"] = pn
                self.log(f"{S.name}: sbatch failed: {err[:160]}")
                if st["sbatch_err"] >= RETRY["sbatch"]:
                    self.halt(S, st, f"sbatch fails {st['sbatch_err']} passes in a row: {err[:200]}", kind="ops")
            self.write_state()
            return
        jid = out.split(";")[0].strip()
        it.update(status="submitted", job=jid)
        prog = self.progress(S)
        for u in us:
            uss = st["units"][u.key]
            uss["jobs"].append(dict(id=f"{jid}_{u.idx}" if u.akey is not None else jid, intent=iid, t_sub=self.now,
                                    epoch=int(st.get("epoch") or 0), mem_gb=mem, disk_gb=u.disk_gb,
                                    cmd_sha=hashlib_sha(" ".join(u.cmd)), end=None))
            uss["prog_sub"] = prog
            self.jobidx[uss["jobs"][-1]["id"]] = (S.name, u.key)
        if st.get("prog_epoch") is None:
            st["prog_epoch"] = prog
        st["sbatch_err"] = 0
        st["started"] = True
        st["status"] = "queued"
        self._count(S, us, bool(deps))
        tag = f"{jid}[{','.join(str(u.idx) for u in us)}]" if u0.akey is not None else jid
        self.log(f"{S.name} submit {tag} {u0.akey or u0.key} {'filler' if S.filler else 'crit'}")
        self.write_state()

    def _count(self, S, us, has_deps=False):
        n, u0 = len(us), us[0]
        self.T += n
        self.total += n
        if not has_deps:
            self.EP += n
        self.CPU += n * u0.c
        self.GPU += n * u0.gpu
        if S.filler:
            self.T_filler += n
        self.n_sub += n

    # ---------------------------------------------------------------- end of pass
    def finish(self):
        self.state["pass"]["t1"] = iso(self.c.clock())
        self.state["pass"]["submitted_tasks"] = self.n_sub
        items = list(self.attn)
        if self.state.get("halted"):
            items.insert(0, "HALT " + ", ".join(f"{h} (results/STOP_{h}.md)" for h in self.state["halted"]))
        canc = [n for n, s in self.state["stages"].items() if s.get("status") == "cancelled"]
        if canc:
            items.append("cancelled (lead): " + ", ".join(canc))
        if self.paused:
            items.append("PIPELINE_PAUSE present: no submissions")
        if self.state.get("finished"):
            items.append("finished (STOP_S.md)")
        self.state["attention"] = items
        att = f"{self.now} " + (" | ".join(items) if items else "ok") + "\n"
        if not self.dry:
            self.flush_log()
            self.write_state()
            try:
                M.write_json(self.c.pdir / "state.prev.json", self.state)     # fallback copy (load_state)
            except Exception:                                               # noqa: BLE001
                pass
            M.write_text(self.c.ctl / "PIPELINE_ATTENTION", att)
        self.report()

    def report(self):
        s = self.state["slots"]
        print(f"[pipeline{' dry' if self.dry else ''}] pass {self.state['pass']['n']} job {self.c.job} {self.now} | squeue "
              f"total {s['total']} (non-launcher {s['T']}, running {s['running']}, eligible pending "
              f"{s['eligible_pending']}) CPU {s['CPU']} GPU {s['GPU']} launcher {s['launcher']} | after: total "
              f"{self.total} eligible pending {self.EP} CPU {self.CPU} GPU {self.GPU} | submitted {self.n_sub} tasks",
              flush=True)
        for S in self.reg:
            st = self.st(S.name)
            print(f"  {S.name:9s} {st.get('status') or '-':13s} ep{st.get('epoch', 0)} {st.get('why', '')}"[:260], flush=True)
        for x in self.planned:
            print(("  would: " if self.dry else "  ") + x, flush=True)
        for x in self.declines:
            print(("  DECISION (dry, not written): " if self.dry else "  DECISION: ") + x.rstrip(), flush=True)
        for x in self.events:
            print(f"  event: {x}", flush=True)
        print("  attention: " + (" | ".join(self.state.get("attention") or []) or "ok"), flush=True)


def hashlib_sha(s):
    import hashlib
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ================================================================ commands
def cmd_status(ctx):
    S = json.loads(ctx.state_path.read_text()) if ctx.state_path.is_file() else None
    if not S:
        print(f"no state file {ctx.state_path}")
        return 2
    try:
        sq = ctx.slurm.squeue()
    except SnapshotError as e:
        print(f"squeue failed: {e}")
        sq = []
    owner = {}
    for name, st in S.get("stages", {}).items():
        for us in st.get("units", {}).values():
            for j in us.get("jobs", []):
                owner[j["id"]] = name
    cnt = {}
    for l in sq:
        n = owner.get(l["id"]) or (l["name"][5:] if l["name"].startswith("mltp_") else None)
        if n:
            cnt[n] = cnt.get(n, 0) + 1
    p = S.get("pass", {})
    print(f"pass {p.get('n')} {p.get('t0')} job {p.get('job')} | slots {S.get('slots')} | halted {S.get('halted')} | "
          f"paused {S.get('paused')} | quota {S.get('quota', {}).get('est_free')} GB (stale {S.get('quota', {}).get('stale')})")
    print(f"squeue now: {len(sq)} tasks, running {sum(l['state'] == 'RUNNING' for l in sq)}, eligible pending "
          f"{sum(eligible(l) for l in sq if l['name'] != LAUNCH)}, launchers {sum(l['name'] == LAUNCH for l in sq)}")
    for s in registry():
        st = S.get("stages", {}).get(s.name, {})
        print(f"{s.name:9s} {st.get('status') or '-':13s} ep{st.get('epoch', 0)} live {cnt.get(s.name, 0):2d} | "
              f"{st.get('why', '')}"[:240])
    print("attention: " + (" | ".join(S.get("attention") or []) or "ok"))
    return 0


def cmd_guard(files, code=None, git=None):
    """first segment of every launcher job: exit 0 when every file is tracked and equal to HEAD, else exit 4 (the job
    runs nothing; the launcher resubmits it once the code is committed — exit 4 uses no retry)."""
    code = Path(code or HERE)
    g = git or Git(code, readonly=True)
    bad = []
    for f in files:
        p = code / f
        if not p.is_file():
            bad.append(f"{f} missing")
        elif not g.tracked(p):
            bad.append(f"{f} not committed")
        elif not g.clean(p):
            bad.append(f"{f} modified since HEAD")
    if bad:
        print(f"[mltp guard] code changed after submission: {'; '.join(bad)} -> exit {EXIT_GUARD} (resubmitted by the "
              f"launcher once committed)", flush=True)
        return EXIT_GUARD
    print(f"[mltp guard] {len(files)} files == HEAD {g.head()}", flush=True)
    return 0


def cmd_lists(ctx):
    """fold test lists (FT-2), dipole lists (X), the A-3 pilot list (60 per fold, split as evenly as possible over the
    dipoles present in that fold — folds 0-3 lack a dipole, so not always 12 per dipole; seeded)."""
    import numpy as np
    import pandas as pd
    T = M.targets()
    cols = ["rxn_id", "job_id", "dipole", "fold_T1", "n_atoms"]
    out = ctx.lists
    info = dict(written=iso(time.time()), code=CODE_VERSION, source=str(M.LABELS), n_targets=int(len(T)),
                n_fold_minus1=int((T.fold_T1 == -1).sum()), files={},
                rules=dict(f_test="fold_T1 == k (all test reactions of fold k; FT-2 spec 5)",
                           dip="lodo == d (mlt_common.DIPOLES order; X spec 12)",
                           a3=f"per fold k (0..4): 60 fold-k test reactions split as evenly as possible over the dipoles "
                              f"present in that fold (one per dipole in M.DIPOLES order, round-robin), seeded draws "
                              f"(numpy default_rng({M.SEED} + 3)) = 300 total (A-3 spec 6)"))
    for k in range(5):
        S = T[T.fold_T1 == k][cols].sort_values("rxn_id")
        p = out / f"f{k}_test.csv"
        M.write_csv(p, S)
        info["files"][f"f{k}_test"] = dict(path=str(p), n=int(len(S)))
    for d, dn in enumerate(M.DIPOLES):
        S = T[T.lodo == d][cols].sort_values("rxn_id")
        p = out / f"dip{d}.csv"
        M.write_csv(p, S)
        info["files"][f"dip{d}"] = dict(path=str(p), n=int(len(S)), dipole=dn)
    rng = np.random.default_rng(M.SEED + 3)
    parts = []
    for k in range(5):
        avail = {dn: int(((T.fold_T1 == k) & (T.dipole == dn)).sum()) for dn in M.DIPOLES}
        quota, left = {dn: 0 for dn in M.DIPOLES}, 60
        while left > 0 and any(quota[d] < avail[d] for d in M.DIPOLES):      # even split over the dipoles present
            for dn in M.DIPOLES:
                if left > 0 and quota[dn] < avail[dn]:
                    quota[dn] += 1
                    left -= 1
        for dn in M.DIPOLES:
            g = T[(T.fold_T1 == k) & (T.dipole == dn)].sort_values("rxn_id")
            if quota[dn]:
                parts.append(g.iloc[sorted(rng.choice(len(g), size=quota[dn], replace=False))][cols])
    A = pd.concat(parts).sort_values(["fold_T1", "rxn_id"])
    p = out / "a3.csv"
    M.write_csv(p, A)
    info["files"]["a3"] = dict(path=str(p), n=int(len(A)), by_fold=A.fold_T1.value_counts().sort_index().to_dict(),
                               by_fold_dipole={f"{k}/{d}": int(v) for (k, d), v in
                                               A.groupby(["fold_T1", "dipole"]).size().items()})
    M.write_json(out / "lists.json", info)
    print(json.dumps({k: v["n"] for k, v in info["files"].items()}), flush=True)
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else ""
    if cmd == "once":
        ctx = Ctx()
        try:
            return Pass(ctx).run()
        except Exception as e:                                              # noqa: BLE001
            traceback.print_exc()
            try:
                M.write_text(ctx.ctl / "PIPELINE_ATTENTION", f"{iso(time.time())} pass error: {type(e).__name__}: {e}\n")
            except Exception:                                               # noqa: BLE001
                pass
            return 1
    if cmd == "dry":
        return Pass(Ctx(), dry=True).run()
    if cmd == "status":
        return cmd_status(Ctx())
    if cmd == "lists":
        return cmd_lists(Ctx())
    if cmd == "guard":
        return cmd_guard(a[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
