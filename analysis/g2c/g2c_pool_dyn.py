#!/usr/bin/env python3
"""G2c dynamic pool (user 2026-10-31 "진행": jobs that finish early take over the remaining reactions). D1 run env,
sbatch only (jobs/g2c.slurm with G2C_ROOT=none). Replaces the static i::n slices of g2c_d1.py pool / d0g2.py pool:
every job pulls the next unfinished, unclaimed reaction of the four rerun roots from ONE shared list.

The computation is unchanged: each reaction runs exactly the per-reaction command of the static pools,
  d1_r<k>: python analysis/g2c/g2c_d1.py one <T> <cores>   env D1G2_SCRATCH=$G2C/d1_r<k>
  d0_r<k>: python analysis/geo/d0g2.py one <T> <cores>     env D0G2_SCRATCH=D1G2_SCRATCH=$G2C/d0_r<k>
(same scripts and environment as jobs/g2c.slurm; their files are not modified). Finished = .done / .fail_* marker in
the root's g2_final/<rxn_id>/, exactly as the static pools.

Claims ($G2C/claims/<root>/<T>; content "<owner> <host> <time> <by> <attempt>"):
  * created atomically (temp file + os.link, never empty); the owner's run touches it every HEARTBEAT_S (mtime);
  * stale = no final marker, heartbeat older than MIN_STALE_S AND owner absent from a fresh `squeue -o %A`;
  * takeover (stale or 'retry') is atomic: only the creator of the token <T>.take.<owner>.<attempt> (O_EXCL, holding
    the creator's job id) may os.replace the claim; the token is removed if the takeover fails, and an orphan token
    (creator dead, older than MIN_STALE_S) is retired by an atomic rename so the claim can be taken again;
  * a run that ends without a marker (or exceeds RUN_TIMEOUT_S) -> owner 'retry' with attempt+1 (any job retries it
    once); after MAX_ATTEMPTS -> 'error' (skipped, counted by 'plan'); a claim is only rewritten while this job owns it;
  * circuit breaker: a child exit 3 (STOP) or MAX_CONSEC consecutive errors in this job -> stop, exit 3.
Order: largest expected cost first over all four roots (D1: C_compare g2_core_h_total; D0: GEO1 Z_status core_h).
A job leaves when nothing is claimable (unclaimed, retry, or stale); reactions held by live jobs are theirs.
Refuses to start while a static G2c pool job (name g2c_d1_r* / g2c_d0_r*) is queued, or outside SLURM.

  python g2c_pool_dyn.py plan                  # per root: targets / final / live / stale / retry / error / open / tokens
  python g2c_pool_dyn.py run <n_par> <cores>   # stops starting new reactions after $G2C_STOP_AFTER_H h (default 44)
run exit: 0 = nothing claimable is left and this job ended no run in error; 2 = time limit, an error run, or claimable
work left at exit; 3 = STOP."""
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
JOB = os.environ.get("SLURM_JOB_ID", "")
HOST = socket.gethostname()
USER = os.environ.get("USER", "yeseo1ee")
MIN_STALE_S, HEARTBEAT_S, RUN_TIMEOUT_S = 900, 120, 6 * 3600
MAX_ATTEMPTS, MAX_CONSEC = 2, 5


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


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
        return [x.strip() for x in out.split("\n") if x.strip()]
    except (subprocess.SubprocessError, OSError):
        return None


def live_jobs():
    """job ids (unique per array task: %A) queued now; None if squeue failed (then nothing is ever stale)."""
    out = squeue("%A")
    return None if out is None else set(out)


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


def _tmp(p):
    return p.with_name(f".{p.name}.{JOB}.{threading.get_ident()}.tmp")


def write_claim_new(p):
    """atomic create: True if this job now owns a brand-new claim (attempt 0)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(p)
    tmp.write_text(f"{JOB} {HOST} {now()} - 0\n")
    try:
        os.link(tmp, p)
        return True
    except FileExistsError:
        return False
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def set_claim(p, owner_word, attempt):
    """rewrite the claim, but only while this job owns it (atomic replace)."""
    c = read_claim(p)
    if c is None or c[0] != JOB:
        return False
    tmp = _tmp(p)
    tmp.write_text(f"{owner_word} {HOST} {now()} {JOB} {attempt}\n")
    os.replace(tmp, p)
    return True


def take_over(p, old_owner, attempt, live):
    """atomic takeover of a stale / 'retry' claim. Only the creator of the token may replace the claim."""
    tok = p.with_name(f"{p.name}.take.{old_owner}.{attempt}")
    for _ in range(2):
        try:
            fd = os.open(tok, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:                                              # orphan token: creator dead and old -> retire it
                creator = tok.read_text().split()[0]
                age = time.time() - tok.stat().st_mtime
            except (FileNotFoundError, IndexError):
                return False
            if live is None or creator in live or age < MIN_STALE_S:
                return False
            try:
                os.rename(tok, tok.with_name(f"{tok.name}.dead.{JOB}.{threading.get_ident()}"))
            except FileNotFoundError:
                return False                                  # someone else retired it first
    else:
        return False
    ok = False
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"{JOB} {HOST} {now()}\n")
        cur = read_claim(p)
        if cur is not None and cur[0] == old_owner and cur[1] == attempt:
            tmp = _tmp(p)
            tmp.write_text(f"{JOB} {HOST} {now()} - {attempt}\n")
            os.replace(tmp, p)
            ok = True
        return ok
    finally:
        if not ok:
            try:
                tok.unlink()
            except FileNotFoundError:
                pass


def task_env(root):
    e = dict(os.environ)
    rp = str(G2C / root)
    e.update(D1G2_SCRATCH=rp, G2C_ROOT=rp, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    if root.startswith("d0"):
        e.update(D0G2_SCRATCH=rp, GEO_SCRATCH="/gpfs/tmp_cpu2/yeseo1ee/geo")
    return e


def run_one(root, T, rid, cores, attempt):
    """-> ('.done' | '.fail' | 'error', child rc). Heartbeat on the claim while the child runs."""
    script, cwd = (D1_SCRIPT, D1_SCRIPT.parent) if root.startswith("d1") else (D0_SCRIPT, D0_SCRIPT.parent)
    p = CLAIMS / root / T
    log = G2C / root / "logs" / "rxn" / f"{T}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        child = subprocess.Popen([sys.executable, str(script), "one", T, str(cores)], cwd=cwd, stdout=f,
                                 stderr=subprocess.STDOUT, env=task_env(root))
        while True:
            try:
                rc = child.wait(timeout=HEARTBEAT_S)
                break
            except subprocess.TimeoutExpired:
                try:
                    os.utime(p)
                except OSError:
                    pass
                if time.time() - t0 > RUN_TIMEOUT_S:
                    child.kill()
                    rc = child.wait()
                    print(f"  {root} {T}: killed after {RUN_TIMEOUT_S} s", flush=True)
                    break
    if final(root, rid):
        fin = G2C / root / "g2_final" / str(rid)
        return (".done" if (fin / ".done").is_file() else ".fail"), rc
    set_claim(p, "retry" if attempt + 1 < MAX_ATTEMPTS else "error", attempt + 1)
    return "error", rc


def classify(root, T, rid, live):
    """'final' | 'open' | 'live' | 'stale' | 'retry' | 'error' (for plan and the claimable-work check)."""
    if final(root, rid):
        return "final"
    c = read_claim(CLAIMS / root / T)
    if c is None:
        return "open"
    if c[0] in ("retry", "error"):
        return c[0]
    if live is not None and c[0] not in live and time.time() - c[2] > MIN_STALE_S:
        return "stale"
    return "live"


def cmd_plan():
    live = live_jobs()
    rep = {}
    for _, root, T, rid in tasks():
        d = rep.setdefault(root, dict(targets=0, final=0, live=0, stale=0, retry=0, error=0, open=0))
        d["targets"] += 1
        d[classify(root, T, rid, live)] += 1
    for root in ROOTS:
        rep.setdefault(root, {})["tokens"] = len(list((CLAIMS / root).glob("*.take.*"))) if (CLAIMS / root).is_dir() else 0
    print(json.dumps(rep, indent=1), flush=True)
    return 0


def cmd_run(n_par, cores):
    if not JOB:
        print("STOP: run only inside a SLURM job (SLURM_JOB_ID unset)", file=sys.stderr)
        return 3
    if not (D1_SCRIPT.is_file() and D0_SCRIPT.is_file()):
        print("STOP: per-reaction scripts missing", file=sys.stderr)
        return 3
    names = squeue("%j")
    static = None if names is None else sorted(n for n in names if n.startswith(("g2c_d1_r", "g2c_d0_r")))
    if static is None or static:
        print(f"STOP: static G2c pool jobs still queued (or squeue failed): {static}", file=sys.stderr)
        return 3
    T_all, t0 = tasks(), time.time()
    lock, pos, res = threading.Lock(), [0], {}
    stop = threading.Event()
    consec = [0]

    def acquire(i, live):
        """try to own reaction i (unfinished and unclaimed / stale / retry) -> attempt or None."""
        _, root, T, rid = T_all[i]
        if final(root, rid):
            return None
        p = CLAIMS / root / T
        c = read_claim(p)
        if c is None:
            return 0 if write_claim_new(p) else None
        if c[0] == "retry":
            return c[1] if take_over(p, "retry", c[1], live) else None
        if c[0] in ("error", JOB):
            return None
        if live is not None and c[0] not in live and time.time() - c[2] > MIN_STALE_S and not final(root, rid):
            if take_over(p, c[0], c[1], live):
                print(f"  stale claim of job {c[0]} taken over: {root} {T}", flush=True)
                return c[1]
        return None

    def next_task(rescan):
        with lock:
            live = live_jobs()                               # fresh once per call (seconds old at most)
            i = 0 if rescan else pos[0]
            while i < len(T_all) and not stop.is_set():
                j = i
                i += 1
                if not rescan:
                    pos[0] = i
                a = acquire(j, live)
                if a is not None:
                    return j, a
            return None

    def claimable_left():
        live = live_jobs()
        return sum(classify(root, T, rid, live) in ("open", "retry", "stale") for _, root, T, rid in T_all)

    def worker():
        rescan = False
        while not stop.is_set():
            if time.time() - t0 > STOP_H * 3600:
                with lock:
                    res["not_started"] = res.get("not_started", 0) + 1
                return
            try:
                t = next_task(rescan)
            except OSError as e:                             # transient filesystem error: wait and retry
                print(f"  next_task: {e!r}; retry in 60 s", flush=True)
                time.sleep(60)
                continue
            if t is None:
                if not rescan:
                    rescan = True                            # whole list once more: stale / retry claims
                    continue
                return                                       # nothing claimable: what is left belongs to live jobs
            j, att = t
            _, root, T, rid = T_all[j]
            try:
                s, rc = run_one(root, T, rid, cores, att)
            except Exception as e:                           # noqa: BLE001  (never leave a held claim behind)
                s, rc = "error", -1
                print(f"  {root} {T}: exception {e!r}", flush=True)
                try:
                    set_claim(CLAIMS / root / T, "retry" if att + 1 < MAX_ATTEMPTS else "error", att + 1)
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
    left = claimable_left()
    print(f"[g2c dyn] {res}; claimable left {left}", flush=True)
    if res.get("stopped"):
        return 3
    return 2 if res.get("not_started") or res.get("error") or left else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["plan"]:
        sys.exit(cmd_plan())
    if len(a) == 3 and a[0] == "run":
        sys.exit(cmd_run(a[1], a[2]))
    raise SystemExit(__doc__)
