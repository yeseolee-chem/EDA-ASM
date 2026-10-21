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

RE_NOCONV = re.compile(r"SCF NOT CONVERGED|did not converge|NOT CONVERGED|error termination in SCF", re.I)
KEEP = ("inp", "out", "err")                     # copied back (gzip for out / err); everything else is dropped
T0 = time.time()
STOP_AFTER_H = float(os.environ.get("D1HF_STOP_AFTER_H", "45"))      # do not start a reaction after this (48 h wall)


def run_name(geom, level):
    return geom if level == "hf" else f"{geom}_{level}"


def calcs(geom):
    return H.CALCS if geom == "dft" else ("eda",)


def src_dir(T, geom, level):
    return H.INPUTS / T / ("g1" if geom == "g1" else level)


def out_ok(p):
    return p.is_file() and H.terminated(H.read_text(p))


def retry_text(txt, calc):
    """spec 4 retry: SlowConv on the '!' line (and in the FRAG strings of eda) + '%scf MaxIter 500 end'."""
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
    """one ORCA process (serial) in the work folder -> (terminated normally, wall seconds)."""
    t = time.time()
    with open(work / f"{calc}.out", "w") as fo, open(work / f"{calc}.err", "w") as fe:
        subprocess.run([str(H.ORCA_BIN), f"{calc}.inp"], cwd=work, stdout=fo, stderr=fe, env=os.environ.copy())
    return H.terminated((work / f"{calc}.out").read_text(errors="replace")), time.time() - t


def copy_back(work, dest, calc):
    """gzip <calc>.out / .err (and eda_frag*.out / .err) and copy the .inp files; atomic per file."""
    names = [calc] + ([f"eda_frag{k}" for k in (1, 2)] if calc == "eda" else [])
    for n in names:
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
    dest = H.RUNS / run_name(geom, level) / T
    if (dest / ".done").is_file():
        return T, "skip"
    if (dest / ".fail").is_file() and os.environ.get("D1HF_RETRY_FAILED") != "1":
        return T, "failed_before"                       # spec 4: one retry only; a resubmission does not retry again
    if time.time() - T0 > STOP_AFTER_H * 3600:
        return T, "not_started"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / ".fail").is_file():
        (dest / ".fail").unlink()
    work = root / run_name(geom, level) / T
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    src = src_dir(T, geom, level)
    meta = dict(job_id=T, geom=geom, level=level, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                orca_bin=str(H.ORCA_BIN), calcs={})
    try:
        for c in calcs(geom):
            if out_ok(dest / f"{c}.out.gz") and (c != "eda" or all(out_ok(dest / f"eda_frag{k}.out.gz") for k in (1, 2))):
                meta["calcs"][c] = "cached"
                continue
            txt = (src / f"{c}.inp").read_text()
            for att in (1, 2):
                for p in work.glob(f"{c}*"):
                    p.unlink() if p.is_file() else shutil.rmtree(p)
                (work / f"{c}.inp").write_text(txt)
                ok, wall = orca(work, c)
                out = (work / f"{c}.out").read_text(errors="replace")
                if c == "eda":
                    ok &= all((work / f"eda_frag{k}.out").is_file()
                              and H.terminated((work / f"eda_frag{k}.out").read_text(errors="replace")) for k in (1, 2))
                conv = bool(RE_NOCONV.search(out)) or (c == "eda" and any(
                    (work / f"eda_frag{k}.out").is_file() and RE_NOCONV.search(
                        (work / f"eda_frag{k}.out").read_text(errors="replace")) for k in (1, 2)))
                append_tsv(dest / "timing.tsv", dict(calc=c, attempt=att, orca_seconds=H.run_seconds(out),
                                                     wall_seconds=round(wall, 3), host=meta["host"]),
                           ["calc", "attempt", "orca_seconds", "wall_seconds", "host"])
                append_tsv(dest / "attempts.tsv", dict(calc=c, attempt=att, change="none" if att == 1 else
                                                       "SlowConv + %scf MaxIter 500 end", ok=ok,
                                                       scf_not_converged=conv),
                           ["calc", "attempt", "change", "ok", "scf_not_converged"])
                if ok:
                    break
                if att == 1 and conv:
                    txt = retry_text(txt, c)
                    continue
                copy_back(work, dest / "failed", c)
                H.write_atomic(dest / ".fail", f"{c}: " + ("scf_not_converged" if conv else "orca_error") +
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
    if geom not in ("dft", "g1") or level not in ("hf", "orig1") or (geom == "g1" and level != "hf"):
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
    return 0 if not res.get("not_started") else 2


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
