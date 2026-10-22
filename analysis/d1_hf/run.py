#!/usr/bin/env python3
"""D1-HF runner (docs/specs/D1HF.md section 4). Run through sbatch only (jobs/hf.slurm).

  python run.py run <list.csv> <geom> <level> [N] [i n]
      reactions = the job_id column of <list.csv> (slice i of n), N at a time, each ORCA process on one core;
      geom dft = the label structures (5 calculations: frag1_dist, frag2_dist, frag1_rel, frag2_rel, then eda),
      geom g1 = the G1 structures (eda only, $HF/inputs/<T>/g1/eda.inp); level hf = $HF/inputs/<T>/hf/,
      level orig1 = $HF/inputs/<T>/orig1/ (S-1: the original B3LYP / def2-TZVP inputs, %pal nprocs 1).
  python run.py status <geom> <level>    # counts of done / failed / missing reactions

Per reaction: work folder ${TMPDIR:-/tmp}/d1hf_<SLURM_JOB_ID>/<run>/<T> (node local), results in $HF/runs/<run>/<T>/
(<run> = geom for level hf, else geom_level): <calc>.inp, <calc>.out.gz (eda also eda_frag{1,2}.{inp,out}.gz),
<calc>.err.gz, timing.tsv, attempts.tsv, meta.json, .done or .fail. A calculation whose .out.gz terminated normally is
skipped and a reaction with .done is skipped (idempotent). An SCF convergence failure is rerun once with SlowConv (the
'!' line and, for eda, the FRAG strings) and '%scf MaxIter 500 end' (spec 4; nothing else changes); any other failure
stops the reaction (.fail). .gbw / .tmp / .densities files are never copied back (the work folder is deleted).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_common as H  # noqa: E402

RE_NOCONV = re.compile(r"SCF NOT CONVERGED|NOT FULLY CONVERGED|did not converge|error termination in SCF", re.I)
RE_MAXITER = re.compile(r"Maximum # iterations\s+MaxIter\s+\.+\s+(\d+)")
KEEP = ("inp", "out", "err")                     # copied back (gzip for out / err); everything else is dropped
T0 = float(os.environ.get("D1HF_JOB_T0", time.time()))              # job start (hf.slurm), else process start
STOP_AFTER_H = float(os.environ.get("D1HF_STOP_AFTER_H", "45"))      # do not start a reaction after this (48 h wall)


def run_name(geom, level):
    return geom if level == "hf" else f"{geom}_{level}"


# D1-G2 Phase H (docs/specs/D1G2.md 9): geom g2 = the G2 structures, level hf only: $HF/inputs/<T>/g2/ written by
# 'prep.py geominputs g2', the 5 calculations as for g1 with its fragment inputs (not in the module docstring = the
# usage text, unchanged).
def src_dir(T, geom, level):
    """g1 / g2: the geometry folder $HF/inputs/<T>/<geom>/; dft: the level folder (hf, orig1)."""
    return H.INPUTS / T / (geom if geom in ("g1", "g2") else level)


def calcs(geom, src=None):
    """dft: the 5 calculations; g1: the 5 as well when the G1 fragment inputs exist (additional instruction 2,
    2026-10-22: frag*_dist from the G1 TS, frag*_rel = the G1 reactants), else the EDA only; g2: as g1 ('prep.py
    geominputs' writes all 5 inputs or none)."""
    if geom == "dft" or (src is not None and all((src / f"{c}.inp").is_file() for c in H.CALCS)):
        return H.CALCS
    return ("eda",)


def scf_names(calc):
    return [calc] + ([f"eda_frag{k}" for k in (1, 2)] if calc == "eda" else [])


def scf_good(text):
    """terminated normally, one 'SCF CONVERGED AFTER', no non-convergence message."""
    return H.terminated(text) and text.count("SCF CONVERGED AFTER") == 1 and not RE_NOCONV.search(text)


def cached(dest, calc):
    return all((dest / f"{n}.out.gz").is_file() and scf_good(H.read_text(dest / f"{n}.out.gz")) for n in scf_names(calc))


def retry_text(txt, calc):
    """spec 4 retry: SlowConv on the '!' line (and in the FRAG strings of eda) + '%scf MaxIter 500 end'. ORCA does not
    copy the %scf block into the fragment inputs it writes for the EDA, so the EDA fragment SCFs get SlowConv only; the
    MaxIter each SCF actually used is read from the outputs and written to attempts.tsv."""
    out = []
    for ln in txt.splitlines(keepends=True):
        s = ln.strip()
        if s.startswith("!"):
            ln = ln.rstrip("\n") + " SlowConv\n"
        elif calc == "eda" and s.split()[:1] in (["FRAG1"], ["FRAG2"]):
            ln = re.sub(r'"\s*$', ' SlowConv"', ln.rstrip("\n")) + "\n"
        elif s.lower().startswith("%maxcore"):
            ln = ln + "%scf MaxIter 500 end\n"
        out.append(ln)
    return "".join(out)


def orca(work, calc):
    """one ORCA process (serial) in the work folder -> (return code, wall seconds)."""
    t = time.time()
    with open(work / f"{calc}.out", "w") as fo, open(work / f"{calc}.err", "w") as fe:
        p = subprocess.run([str(H.ORCA_BIN), f"{calc}.inp"], cwd=work, stdout=fo, stderr=fe, env=os.environ.copy())
    return p.returncode, time.time() - t


def copy_back(work, dest, calc):
    """gzip <calc>.out / .err (and eda_frag*.out / .err) and copy the .inp files; atomic per file."""
    for n in scf_names(calc):
        for ext in KEEP:
            p = work / f"{n}.{ext}"
            if not p.is_file():
                continue
            if ext == "inp":
                H.write_atomic(dest / p.name, p.read_bytes())
            else:
                H.write_atomic(dest / f"{p.name}.gz", lambda t, p=p: t.write_bytes(gzip.compress(p.read_bytes(), 6)))


def append_tsv(p, row, cols):
    new = not p.is_file()
    with open(H.assert_path(p), "a") as f:
        if new:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(str(row[c]) for c in cols) + "\n")


def run_rxn(T, geom, level, root):
    """one reaction; returns (T, status). Statuses: skip, failed_before, not_started, no_input, done, fail, killed,
    error:<exception> (the last two leave no .fail, so a resubmission runs the reaction again)."""
    try:
        return _run_rxn(T, geom, level, root)
    except Exception as e:                               # one reaction must not stop the pool
        import traceback
        print(f"  {T}: ERROR {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        return T, f"error:{type(e).__name__}"


def _run_rxn(T, geom, level, root):
    dest = H.RUNS / run_name(geom, level) / T
    src = src_dir(T, geom, level)
    todo = calcs(geom, src)
    if (dest / ".done").is_file() and all(cached(dest, c) for c in todo):
        return T, "skip"                                # a .done reaction missing calculations (G1 fragments) is filled
    if (dest / ".fail").is_file() and os.environ.get("D1HF_RETRY_FAILED") != "1":
        return T, "failed_before"                       # spec 4: one retry only; a resubmission does not retry again
    if not all((src / f"{c}.inp").is_file() for c in todo):
        return T, "no_input"
    if time.time() - T0 > STOP_AFTER_H * 3600:
        return T, "not_started"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / ".fail").is_file():
        (dest / ".fail").unlink()
    work = root / run_name(geom, level) / T
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    meta = dict(job_id=T, geom=geom, level=level, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                orca_bin=str(H.ORCA_BIN), calcs={})
    cols = ["calc", "attempt", "change", "returncode", "ok", "scf_not_converged", "maxiter"]
    try:
        for c in todo:
            if cached(dest, c):
                meta["calcs"][c] = "cached"
                continue
            txt = (src / f"{c}.inp").read_text()
            for att in (1, 2):
                for p in work.glob(f"{c}*"):
                    p.unlink() if p.is_file() else shutil.rmtree(p)
                (work / f"{c}.inp").write_text(txt)
                rc, wall = orca(work, c)
                outs = {n: (work / f"{n}.out").read_text(errors="replace") if (work / f"{n}.out").is_file() else ""
                        for n in scf_names(c)}
                ok = all(scf_good(t) for t in outs.values())
                conv = any(RE_NOCONV.search(t) for t in outs.values())
                killed = rc < 0 or not outs[c].strip()
                append_tsv(dest / "timing.tsv", dict(calc=c, attempt=att, orca_seconds=H.run_seconds(outs[c]),
                                                     wall_seconds=round(wall, 3), host=meta["host"]),
                           ["calc", "attempt", "orca_seconds", "wall_seconds", "host"])
                append_tsv(dest / "attempts.tsv", dict(
                    calc=c, attempt=att, change="none" if att == 1 else
                    ("SlowConv (! line + FRAG strings) + %scf MaxIter 500 end (parent SCF only)" if c == "eda"
                     else "SlowConv + %scf MaxIter 500 end"), returncode=rc, ok=ok, scf_not_converged=conv,
                    maxiter=";".join(f"{n}:{','.join(RE_MAXITER.findall(t)) or '-'}" for n, t in outs.items())), cols)
                if ok:
                    break
                if killed:
                    return T, "killed"
                if att == 1 and conv:
                    txt = retry_text(txt, c)
                    continue
                copy_back(work, dest / "failed", c)
                H.write_atomic(dest / ".fail", f"{c}: " + ("scf_not_converged" if conv else f"orca_error rc={rc}") +
                               (" (after retry)" if att == 2 else "") + "\n")
                return T, "fail"
            copy_back(work, dest, c)
            meta["calcs"][c] = f"ok (attempt {att})"
        H.write_json(dest / "meta.json", meta)
        H.write_atomic(dest / ".done", "")
        return T, "done"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_run(lst, geom, level, n, i=0, m=1):
    if geom not in ("dft", "g1", "g2") or level not in ("hf", "orig1") or (geom in ("g1", "g2") and level != "hf"):
        H.stop(f"bad geom / level {geom} / {level}")
    if not os.access(H.ORCA_BIN, os.X_OK):
        H.stop(f"{H.ORCA_BIN} not executable")
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[i::m]
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"d1hf_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"[run] {len(ids)} reactions ({lst}, slice {i}/{m}), {geom} / {level}, {n} workers, work {root}, "
          f"ORCA {H.ORCA_BIN} sha256 {H.sha256(H.ORCA_BIN)[:16]}", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(run_rxn, T, geom, level, root) for T in ids]
        for f in as_completed(futs):
            T, s = f.result()
            res[s] = res.get(s, 0) + 1
            print(f"  {T}: {s}  ({sum(res.values())}/{len(ids)}, {time.time() - T0:.0f} s)", flush=True)
    shutil.rmtree(root, ignore_errors=True)
    print(f"[run] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith(("error", "killed")) for k in res) else 0


def cmd_status(geom, level):
    d = H.RUNS / run_name(geom, level)
    rx = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    n_done = sum((d / T / ".done").is_file() for T in rx)
    fails = {T: (d / T / ".fail").read_text().strip() for T in rx if (d / T / ".fail").is_file()}
    print(json.dumps(dict(run=run_name(geom, level), n_folders=len(rx), n_done=n_done, n_fail=len(fails),
                          fails=dict(list(fails.items())[:20])), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"] and len(a) >= 4:
        n = int(a[4]) if len(a) > 4 else int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        sl = (int(a[5]), int(a[6])) if len(a) > 6 else (0, 1)
        sys.exit(cmd_run(a[1], a[2], a[3], n, *sl))
    if a[:1] == ["status"] and len(a) == 3:
        sys.exit(cmd_status(a[1], a[2]))
    raise SystemExit(__doc__)
