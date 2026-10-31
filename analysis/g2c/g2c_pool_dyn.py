#!/usr/bin/env python3
"""G2c dynamic pool (user 2026-10-31 "진행": jobs that finish early take over the remaining reactions). D1 run env,
sbatch only (jobs/g2c.slurm with G2C_ROOT=none). Replaces the static i::n slices of g2c_d1.py pool / d0g2.py pool:
every job pulls the next unfinished, unclaimed reaction of the four rerun roots from ONE shared list.

The computation is unchanged: each reaction runs exactly the per-reaction command of the static pools,
  d1_r<k>: python analysis/g2c/g2c_d1.py one <T> <cores>   env D1G2_SCRATCH=$G2C/d1_r<k>
  d0_r<k>: python analysis/geo/d0g2.py one <T> <cores>     env D0G2_SCRATCH=D1G2_SCRATCH=$G2C/d0_r<k>
(same scripts and environment as jobs/g2c.slurm; their files are not modified). Finished = .done / .fail_* marker in
the root's g2_final/<rxn_id>/, exactly as the static pools.

Claims ($G2C/claims/<root>/<T>; content "<SLURM_JOB_ID> <host> <time> <attempt>"):
  * created atomically (temp file + os.link, never empty); a claimed reaction is skipped by the other jobs;
  * stale = no final marker, older than MIN_STALE_S and the owner job id is absent from a FRESH `squeue -o %A`;
    the takeover is atomic: only the job that creates the token <T>.take.<owner> (O_EXCL) may os.replace the claim;
  * a run that ends without a marker is an error: the claim gets attempt+1 and owner 'retry' (any job may retry it
    once, through the same token rule); after MAX_ATTEMPTS it becomes 'error' (skipped; counted by 'plan');
  * circuit breaker: a child exit 3 (STOP) or MAX_CONSEC consecutive errors in this job -> stop and exit 3.
Order: largest expected cost first over all four roots (D1: C_compare g2_core_h_total; D0: GEO1 Z_status core_h).
Refuses to start while a static G2c pool job (name g2c_d1_r* / g2c_d0_r*) is in the queue.

  python g2c_pool_dyn.py plan                  # counts per root (targets / final / live / stale / retry / error / open)
  python g2c_pool_dyn.py run <n_par> <cores>   # stops starting new reactions after $G2C_STOP_AFTER_H h (default 44)
run exit: 0 = every reaction final (or error after MAX_ATTEMPTS), nothing held by live jobs; 2 = time limit or a
reaction ended in error; 3 = STOP (static pool running, child STOP, too many consecutive errors)."""
from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))
CLAIMS = G2C / "claims"
ROOTS = ("d1_r1", "d1_r2", "d0_r1", "d0_r2")
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
Z_STATUS = REPO / "analysis" / "geo" / "results" / "Z_status.csv"
D1_SCRIPT = REPO / "analysis" / "g2c" / "g2c_d1.py"
D0_SCRIPT = REPO / "analysis" / "geo" / "d0g2.py"
STOP_H = float(os.environ.get("G2C_STOP_AFTER_H", "44"))
JOB = os.environ.get("SLURM_JOB_ID", "nojob")
USER = os.environ.get("USER", "yeseo1ee")
MIN_STALE_S, MAX_ATTEMPTS, MAX_CONSEC, WAIT_S = 600, 2, 5, 300


def fval(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def tasks():
    """[(cost, root, job_id, rxn_id)] for all four roots, largest expected cost first."""
    cost1, cost0 = {}, {}
    with open(C_COMPARE, newline="") as f:
        for r in csv.DictReader(f):
            cost1[int(r["rxn_id"])] = fval(r.get("g2_core_h_total")) or (fval(r.get("n_atoms")) or 0) / 100
    with open(Z_STATUS, newline="") as f:
        for r in csv.DictReader(f):
            cost0[int(r["rxn_id"])] = fval(r.get("core_h")) or (fval(r.get("n_atoms")) or 0) / 100
    with open(G2C / "lists" / "d1_targets.csv", newline="") as f:
        d1 = [(r["job_id"], int(r["rxn_id"])) for r in csv.DictReader(f)]
    out = []
    for root in ROOTS:
        if root.startswith("d1"):
            out += [(cost1.get(rid, 0.0), root, T, rid) for T, rid in d1]
        else:
            labs = json.loads((G2C / root / "G" / "labels_d0_g2.json").read_text())
            out += [(cost0.get(int(d["rxn_id"]), 0.0), root, d["job_id"], int(d["rxn_id"])) for d in labs]
    out.sort(key=lambda t: (-t[0], ROOTS.index(t[1]), t[2]))
    return out


def final(root, rid):
    fin = G2C / root / "g2_final" / str(rid)
    return (fin / ".done").is_file() or any(fin.glob(".fail_*"))


def squeue(fmt):
    try:
        out = subprocess.run(["squeue", "-h", "-u", USER, "-o", fmt], capture_output=True, text=True, check=True,
                             timeout=120).stdout
        return out.split("\n")
    except (subprocess.SubprocessError, OSError):
        return None


def live_jobs():
    """job ids (unique per array task: %A) in the queue now; None if squeue failed (then nothing is ever stale)."""
    out = squeue("%A")
    return None if out is None else {x.strip() for x in out if x.strip()}


def static_pools_queued():
    out = squeue("%j")
    if out is None:
        return None
    return sorted(n.strip() for n in out if n.strip().startswith(("g2c_d1_r", "g2c_d0_r")))


def read_claim(p):
    """(owner, attempt, mtime) or None if there is no claim."""
    try:
        st = p.stat()
        txt = p.read_text().split()
    except FileNotFoundError:
        return None
    owner = txt[0] if txt else "?"
    att = int(txt[4]) if len(txt) > 4 and txt[4].isdigit() else 0
    return owner, att, st.st_mtime


def write_claim_new(p, attempt=0):
    """atomic create: True if this job now owns a brand-new claim."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{JOB}.{threading.get_ident()}.tmp")
    tmp.write_text(f"{JOB} {socket.gethostname()} {time.strftime('%Y-%m-%dT%H:%M:%S')} - {attempt}\n")
    try:
        os.link(tmp, p)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def replace_claim(p, owner_word, attempt):
    tmp = p.with_name(f".{p.name}.{JOB}.{threading.get_ident()}.tmp")
    tmp.write_text(f"{owner_word} {socket.gethostname()} {time.strftime('%Y-%m-%dT%H:%M:%S')} {JOB} {attempt}\n")
    os.replace(tmp, p)


def take_over(p, old_owner, attempt):
    """atomic takeover of a stale / retry claim: only the creator of the token may replace the claim."""
    tok = p.with_name(f"{p.name}.take.{old_owner}.{attempt}")
    try:
        fd = os.open(tok, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.close(fd)
    cur = read_claim(p)
    if cur is None or cur[0] != old_owner or cur[1] != attempt:      # changed meanwhile: leave it
        return False
    tmp = p.with_name(f".{p.name}.{JOB}.{threading.get_ident()}.tmp")
    tmp.write_text(f"{JOB} {socket.gethostname()} {time.strftime('%Y-%m-%dT%H:%M:%S')} - {attempt}\n")
    os.replace(tmp, p)
    return True


def state(p):
    """'none' | 'mine' | 'retry' | 'error' | 'held' (owner word, attempt, mtime kept for the caller)."""
    c = read_claim(p)
    if c is None:
        return "none", None
    if c[0] == "retry":
        return "retry", c
    if c[0] == "error":
        return "error", c
    if c[0] == JOB:
        return "mine", c
    return "held", c


def task_env(root):
    e = dict(os.environ)
    rp = str(G2C / root)
    e.update(D1G2_SCRATCH=rp, G2C_ROOT=rp, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    if root.startswith("d0"):
        e.update(D0G2_SCRATCH=rp, GEO_SCRATCH="/gpfs/tmp_cpu2/yeseo1ee/geo")
    return e


def run_one(root, T, rid, cores, attempt):
    """-> ('.done' | '.fail' | 'error', child rc)."""
    script, cwd = (D1_SCRIPT, D1_SCRIPT.parent) if root.startswith("d1") else (D0_SCRIPT, D0_SCRIPT.parent)
    log = G2C / root / "logs" / "rxn" / f"{T}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, str(script), "one", T, str(cores)], cwd=cwd, stdout=f,
                           stderr=subprocess.STDOUT, env=task_env(root))
    if final(root, rid):
        fin = G2C / root / "g2_final" / str(rid)
        return (".done" if (fin / ".done").is_file() else ".fail"), r.returncode
    p = CLAIMS / root / T
    replace_claim(p, "retry" if attempt + 1 < MAX_ATTEMPTS else "error", attempt + 1)
    return "error", r.returncode


def cmd_plan():
    live = live_jobs()
    rep = {}
    for _, root, T, rid in tasks():
        d = rep.setdefault(root, dict(targets=0, final=0, live=0, stale=0, retry=0, error=0, open=0))
        d["targets"] += 1
        if final(root, rid):
            d["final"] += 1
            continue
        s, c = state(CLAIMS / root / T)
        if s == "held":
            s = "live" if live is None or c[0] in live else "stale"
        d[{"none": "open", "mine": "live"}.get(s, s)] += 1
    print(json.dumps(rep, indent=1), flush=True)
    return 0


def cmd_run(n_par, cores):
    if not (D1_SCRIPT.is_file() and D0_SCRIPT.is_file()):
        print("STOP: per-reaction scripts missing", file=sys.stderr)
        return 3
    sp = static_pools_queued()
    if sp is None or sp:
        print(f"STOP: static G2c pool jobs still queued (or squeue failed): {sp}", file=sys.stderr)
        return 3
    T_all, t0 = tasks(), time.time()
    lock, pos, res = threading.Lock(), [0], {}
    stop = threading.Event()
    consec = [0]

    def acquire(i):
        """try to own reaction i: unfinished and unclaimed / stale / retry. -> attempt or None."""
        _, root, T, rid = T_all[i]
        if final(root, rid):
            return None
        p = CLAIMS / root / T
        if write_claim_new(p):
            return 0
        s, c = state(p)
        if s == "retry":
            return c[1] if take_over(p, "retry", c[1]) else None
        if s == "held" and time.time() - c[2] > MIN_STALE_S:
            live = live_jobs()                                   # fresh list for every takeover decision
            if live is not None and c[0] not in live and not final(root, rid):
                if take_over(p, c[0], c[1]):
                    print(f"  stale claim of job {c[0]} taken over: {root} {T}", flush=True)
                    return c[1]
        return None

    def next_task(rescan):
        with lock:
            i = 0 if rescan else pos[0]
            while i < len(T_all) and not stop.is_set():
                j = i
                i += 1
                if not rescan:
                    pos[0] = i
                a = acquire(j)
                if a is not None:
                    return j, a
            return None

    def others_hold_work():
        live = live_jobs()
        for _, root, T, rid in T_all:
            if final(root, rid):
                continue
            s, c = state(CLAIMS / root / T)
            if s in ("retry", "none") or (s == "held" and (live is None or c[0] in live)):
                return True
        return False

    def worker():
        rescan = False
        while not stop.is_set():
            if time.time() - t0 > STOP_H * 3600:
                with lock:
                    res["not_started"] = res.get("not_started", 0) + 1
                return
            try:
                t = next_task(rescan)
            except OSError as e:                                 # transient filesystem error: wait and retry
                print(f"  next_task: {e!r}; retry in 60 s", flush=True)
                time.sleep(60)
                continue
            if t is None:
                if not rescan:
                    rescan = True                                # whole list again: stale / retry claims
                    continue
                if others_hold_work():                           # other jobs still running: wait, then rescan
                    time.sleep(WAIT_S)
                    continue
                return
            j, att = t
            _, root, T, rid = T_all[j]
            try:
                s, rc = run_one(root, T, rid, cores, att)
            except Exception as e:                               # noqa: BLE001  (never leave a held claim behind)
                s, rc = "error", -1
                print(f"  {root} {T}: exception {e!r}", flush=True)
                try:
                    replace_claim(CLAIMS / root / T, "retry" if att + 1 < MAX_ATTEMPTS else "error", att + 1)
                except OSError:
                    pass
            with lock:
                res[s] = res.get(s, 0) + 1
                consec[0] = consec[0] + 1 if s == "error" else 0
                if rc == 3 or consec[0] >= MAX_CONSEC:
                    res["stopped"] = f"rc={rc} consecutive_errors={consec[0]}"
                    stop.set()
            print(f"  {root} {T}: {s} (attempt {att}, rc {rc})", flush=True)

    print(f"[g2c dyn] job {JOB}: {len(T_all)} reaction-runs over {ROOTS}, {n_par} at a time x {cores} cores", flush=True)
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for f in [ex.submit(worker) for _ in range(int(n_par))]:
            f.result()
    print(f"[g2c dyn] {res}", flush=True)
    if res.get("stopped"):
        return 3
    return 2 if res.get("not_started") or res.get("error") else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["plan"]:
        sys.exit(cmd_plan())
    if len(a) == 3 and a[0] == "run":
        sys.exit(cmd_run(a[1], a[2]))
    raise SystemExit(__doc__)
