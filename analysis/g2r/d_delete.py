#!/usr/bin/env python3
"""D1-G2R Phase D (docs/specs/D1G2R.md 12): checks before deleting the old g2, the before-record, and the after-check.
The deletion itself is the one-line wrap job of spec 12-2-3, submitted only after `check` exits 0:
    sbatch --time=48:00:00 --wrap "rm -rf /gpfs/tmp_cpu2/yeseo1ee/d1_g2 /gpfs/tmp_cpu2/yeseo1ee/ch3/lf"

    python analysis/g2r/d_delete.py check <pushed_sha>   # D-a .. D-e + realpaths -> results/D_check.json,
                                                         # results/D_before.json, $G2R/x/D_listing_<name>.txt.gz; exit 0/3
    python analysis/g2r/d_delete.py verify <rm_job_id>   # both paths gone, $G2R unchanged -> results/D_report.json
env_d1fu2 or any python >= 3.8 (stdlib); py_g2r.slurm. Deletes nothing itself.

D-a  re-read every STOP gate: S0_start, C_report, X_report, R_tests + R_report (+ R_diagnosis.md), lf/L_report +
     L_check, lf/M_report + M_check, lf/F_report (stop false) + F_check, lf/T_lf_check (gate / tokens stop false) +
     T_check.
D-b  `git ls-remote origin refs/heads/g2r` = <pushed_sha> (from the login-node push) = HEAD of this checkout.
D-c  $G2R/jobs against $G2R/x/C_manifest.csv.gz (recorded decision for the Phase R changes): the folders Phase R did
     not touch have exactly the manifest files (count, bytes); in the 20 repair folders every manifest file except
     ref_result.json and .fail_ref is unchanged (bytes) and ref_result.orig.json has the manifest sha256 of
     ref_result.json; 200 random manifest files (seed 42, R-touched files excluded) have their manifest sha256.
D-d  no queued / running job of mine (except this one) has $G2 or $CH3/lf in its command, work dir or output paths
     (squeue + scontrol show job), and none runs a script whose default root is $G2 (cpu_d1.slurm, g.slurm, g2.slurm).
D-e  no symlink under /gpfs/tmp_cpu2/yeseo1ee or /gpfs/home1/yeseo1ee/projects resolves into $G2 or $CH3/lf (links
     inside the two paths themselves are skipped); found links are listed.
realpath($G2) and realpath($CH3/lf) must equal the two constants (spec 12-2-2).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

TARGETS = {"d1_g2": "/gpfs/tmp_cpu2/yeseo1ee/d1_g2", "ch3_lf": "/gpfs/tmp_cpu2/yeseo1ee/ch3/lf"}
LINK_ROOTS = ("/gpfs/tmp_cpu2/yeseo1ee", "/gpfs/home1/yeseo1ee/projects")
R_TOUCHED = {"ref_result.json", ".fail_ref"}
DEFAULT_G2_SCRIPTS = ("cpu_d1.slurm", "g.slurm", "g2.slurm")
N_SAMPLE, SEED = 200, 42


def sh(*a, timeout=300):
    r = subprocess.run(list(a), capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def gate(name, ok, **kw):
    return dict(ok=bool(ok), **kw)


def rj(p):
    p = Path(p)
    return C.read_json(p) if p.is_file() else {}


def d_a():
    R = C.RES
    s0, cr, xr = rj(R / "S0_start.json"), rj(R / "C_report.json"), rj(R / "X_report.json")
    rt, rr = rj(R / "R_tests.json"), rj(R / "R_report.json")
    lr, lc = rj(R / "lf" / "L_report.json"), rj(R / "L_check.json")
    mr, mc = rj(R / "lf" / "M_report.json"), rj(R / "M_check.json")
    fr, fc = rj(R / "lf" / "F_report.json"), rj(R / "F_check.json")
    tl, tc = rj(R / "lf" / "T_lf_check.json"), rj(R / "T_check.json")
    v = {"1": s0.get("all_pass") is True, "C": cr.get("all_pass") is True, "X": xr.get("all_pass") is True,
         "R": rt.get("all_pass") is True and rr.get("all_pass") is True and (R / "R_diagnosis.md").is_file(),
         "L": lr.get("all_pass") is True and lc.get("all_pass") is True,
         "M": mr.get("all_pass") is True and mc.get("all_pass") is True,
         "F": fr.get("stop") is False and fc.get("all_pass") is True,
         "T": (tl.get("gate") or {}).get("stop") is False and (tl.get("gate") or {}).get("gate_rc") == 0
         and (tl.get("tokens") or {}).get("stop") is False and tc.get("all_pass") is True}
    return gate("D-a", all(v.values()), phases=v)


def d_b(sha):
    rc, out, err = sh("git", "-C", str(C.REPO), "ls-remote", "origin", "refs/heads/g2r", timeout=120)
    remote = out.split()[0] if rc == 0 and out.split() else None
    head = sh("git", "-C", str(C.REPO), "rev-parse", "HEAD")[1].strip()
    return gate("D-b", remote is not None and remote == sha == head, ls_remote=remote, ls_remote_rc=rc,
                ls_remote_err=err.strip()[:300], pushed_sha=sha, head=head)


def d_c():
    rows = list(csv.reader(io.StringIO(gzip.decompress((C.G2R_X / "C_manifest.csv.gz").read_bytes()).decode())))[1:]
    man = {}
    for j, rel, b, s in rows:
        man.setdefault(j, {})[rel] = (int(b), s)
    with open(C.G2R_X / "repair_ids.csv", newline="") as fh:
        repair = {r["job_id"] for r in csv.DictReader(fh)}
    bad, n_files, n_bytes = [], 0, 0
    for j, files in man.items():
        root = C.G2R_JOBS / j
        now = {}
        for folder, _d, names in os.walk(root):
            for nm in names:
                p = Path(folder) / nm
                now[str(p.relative_to(root))] = 0 if p.is_symlink() else p.stat().st_size
        if j in repair:
            for rel, (b, _s) in files.items():
                if rel in R_TOUCHED:
                    continue
                if now.get(rel) != b:
                    bad.append(dict(job_id=j, rel=rel, want=b, now=now.get(rel)))
            o = root / "ref_result.orig.json"
            if not o.is_file() or C.sha256(o) != files.get("ref_result.json", (0, None))[1]:
                bad.append(dict(job_id=j, rel="ref_result.orig.json", problem="missing or sha256 != manifest ref_result.json"))
        else:
            if set(now) != set(files) or any(now[r] != files[r][0] for r in files):
                bad.append(dict(job_id=j, missing=sorted(set(files) - set(now))[:5], extra=sorted(set(now) - set(files))[:5],
                                size_differs=[r for r in files if r in now and now[r] != files[r][0]][:5]))
        n_files += len(files)
        n_bytes += sum(b for b, _ in files.values())
    pool = sorted((j, rel, s) for j, files in man.items() for rel, (_b, s) in files.items()
                  if not (j in repair and rel in R_TOUCHED) and not s.startswith("link:"))
    sample = random.Random(SEED).sample(pool, min(N_SAMPLE, len(pool)))
    sbad = [dict(job_id=j, rel=rel) for j, rel, s in sample
            if not (C.G2R_JOBS / j / rel).is_file() or C.sha256(C.G2R_JOBS / j / rel) != s]
    n_folders = sum(1 for p in C.G2R_JOBS.iterdir() if p.is_dir() and not p.name.startswith("."))
    return gate("D-c", not bad and not sbad and n_folders == len(man) == 9266, n_manifest_folders=len(man),
                n_folders_now=n_folders, n_manifest_files=n_files, manifest_bytes=n_bytes, n_differ=len(bad),
                differ=bad[:20], n_sample=len(sample), n_sample_sha_differ=len(sbad), sample_differ=sbad[:20],
                rule="untouched folders exact; repair folders all but ref_result.json/.fail_ref + orig sha; seed 42")


def d_d():
    me = os.environ.get("SLURM_JOB_ID", "")
    user = os.environ.get("USER") or "yeseo1ee"
    rc, out, err = sh("squeue", "-r", "-u", user, "-h", "-o", "%i %j %o")      # -r: one line per array task
    hits, seen, not_inspected = [], [], []
    for line in out.splitlines():
        jid = line.split()[0] if line.split() else ""
        base = jid.split("_")[0]
        if not jid or base == me or jid == me:
            continue
        rc2, sc, _ = sh("scontrol", "show", "job", jid if "[" not in jid else base)
        if rc2 != 0 or not sc.strip():
            rc3, still, _ = sh("squeue", "-h", "-j", base, "-o", "%i")
            if rc3 != 0 or still.strip():
                not_inspected.append(jid)                         # still queued but its paths could not be read
            continue
        text = line + "\n" + sc
        seen.append(jid)
        why = [t for t in TARGETS.values() if t in text]
        why += [s for s in DEFAULT_G2_SCRIPTS if f"/{s}" in text or f" {s}" in text]
        if why:
            hits.append(dict(job=jid, why=why, line=line[:300]))
    return gate("D-d", rc == 0 and not hits and not not_inspected, squeue_rc=rc, squeue_err=err.strip()[:300],
                own_job=me, n_jobs_checked=len(seen), jobs=seen[:50], hits=hits, not_inspected=not_inspected)


def d_e():
    tg = [os.path.realpath(t) for t in TARGETS.values()]
    found, n_links = [], 0
    for root in LINK_ROOTS:
        for folder, dirs, names in os.walk(root, followlinks=False):
            if any(folder == t or folder.startswith(t + "/") for t in TARGETS.values()):
                dirs[:] = []
                continue
            for nm in dirs + names:
                p = os.path.join(folder, nm)
                if os.path.islink(p):
                    n_links += 1
                    r = os.path.realpath(p)
                    if any(r == t or r.startswith(t + "/") for t in tg) or any(
                            os.readlink(p).startswith(t) for t in TARGETS.values()):
                        found.append(dict(link=p, target=os.readlink(p), realpath=r))
    return gate("D-e", not found, roots=list(LINK_ROOTS), n_links_seen=n_links, n_found=len(found), found=found[:200])


def tree(root, skip=()):
    n, b, lines = 0, 0, []
    for folder, _d, names in os.walk(root):
        _d[:] = [x for x in _d if Path(folder, x) not in set(map(Path, skip))]
        for nm in names:
            p = Path(folder) / nm
            s = 0 if p.is_symlink() else p.lstat().st_size
            n += 1
            b += s
            lines.append(f"{p.relative_to(root)}\t{s}")
    return n, b, lines


def cmd_check(sha):
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rp = {k: os.path.realpath(v) for k, v in TARGETS.items()}
    real_ok = all(rp[k] == v and os.path.isdir(v) and not os.path.islink(v) for k, v in TARGETS.items())
    gates = {"D-a": d_a(), "D-b": d_b(sha), "D-c": d_c(), "D-d": d_d(), "D-e": d_e(),
             "realpath": gate("realpath", real_ok, realpath=rp, constants=TARGETS)}
    before = {}
    if all(g["ok"] for g in gates.values()):
        for k, v in TARGETS.items():
            n, b, lines = tree(v)
            C.write_atomic(C.G2R_X / f"D_listing_{k}.txt.gz", gzip.compress(("\n".join(sorted(lines)) + "\n").encode()))
            before[k] = dict(path=v, realpath=rp[k], n_files=n, bytes=b, top_level=sorted(os.listdir(v)),
                             listing=str(C.G2R_X / f"D_listing_{k}.txt.gz"))
        n, b, _ = tree(C.G2R, skip=(C.G2R_LOGS,))
        before["g2r"] = dict(path=str(C.G2R), n_files=n, bytes=b, excluded=[str(C.G2R_LOGS)],
                             note="$G2R/logs excluded: job logs keep growing (check, rm, verify jobs)")
        C.write_json(C.RES / "D_before.json", dict(spec="docs/specs/D1G2R.md 12-2-1", written=time.strftime("%F %T"),
                                                   slurm_job=os.environ.get("SLURM_JOB_ID"), **before))
    rep = dict(spec="docs/specs/D1G2R.md 12-1", started=started, written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), gates=gates, all_pass=all(g["ok"] for g in gates.values()),
               rm_command='sbatch --time=48:00:00 --wrap "rm -rf /gpfs/tmp_cpu2/yeseo1ee/d1_g2 /gpfs/tmp_cpu2/yeseo1ee/ch3/lf"')
    C.write_json(C.RES / "D_check.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], **{k: g["ok"] for k, g in gates.items()}), indent=1))
    return 0 if rep["all_pass"] else 3


def cmd_verify(rm_job):
    b = C.read_json(C.RES / "D_before.json")
    gone = {k: not os.path.lexists(v) for k, v in TARGETS.items()}
    n, by, _ = tree(C.G2R, skip=(C.G2R_LOGS,))
    same = n == b["g2r"]["n_files"] and by == b["g2r"]["bytes"]
    rep = dict(spec="docs/specs/D1G2R.md 12-2-4/5", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), rm_job=rm_job,
               deleted={k: dict(path=v, n_files=b[k]["n_files"], bytes=b[k]["bytes"], gone=gone[k])
                        for k, v in TARGETS.items()},
               g2r_before=b["g2r"], g2r_after=dict(n_files=n, bytes=by, excluded=[str(C.G2R_LOGS)]),
               g2r_unchanged=same,
               all_pass=all(gone.values()) and same)
    C.write_json(C.RES / "D_report.json", rep)
    print(json.dumps(rep, indent=1))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check"] and len(a) == 2:
        sys.exit(cmd_check(a[1]))
    if a[:1] == ["verify"] and len(a) == 2:
        sys.exit(cmd_verify(a[1]))
    raise SystemExit(__doc__)
