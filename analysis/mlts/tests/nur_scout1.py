#!/usr/bin/env python3
"""tests/nur_scout1.py — DEPLOY-V2 (MLTS) Nurion READ-ONLY SCOUT 1 (tar anatomy). Sandbox / compute node only.

  anatomy <tar> [--jobs N]                 stream the outer tar (mode 'r|'); every outer member recorded as a generalised
                                           path pattern; the first N jobs' archive.tgz (0 = all) are opened in memory,
                                           every inner member (and the nested ts.tar / sp.tar members) listed, the ORCA
                                           files written to a node-local $TMPDIR scratch dir (deleted after each job)
                                           and parsed with inv_dft.parse_file / orca_io.read_hess; TS-near check vs the
                                           label TS for D1 targets; small top-level files of the scanned jobs shown.
  count <tar> [--workers K] [--budget S]   the same per-job scan (no per-file detail) for every job, K worker processes,
                                           stop submitting after S seconds; timings + totals per pattern.
The tar is opened read only. Writes only $MLT/tmp/nurtest/scout1_<tar>_<mode>.json; prints a summary JSON."""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import inv_dft as I  # noqa: E402
import mlt_common as C  # noqa: E402
import orca_io as O  # noqa: E402

MANI = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/D1_build/nurion")
OUTDIR = C.MLT / "tmp" / "nurtest"
NEAR_A = 0.3

RE_T = re.compile(r"(?<![A-Za-z0-9])[TCS]\d{4,5}(?![0-9])")
RE_SPEC = re.compile(r"(?<![A-Za-z0-9])([rp])\d+_[A-Za-z0-9]+")
RE_RAND = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]{6}(?![A-Za-z0-9]))(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])"
                     r"[A-Za-z0-9]{6}")
RE_BONDS = re.compile(r"_\d+-\d+(?:_\d+-\d+)*")
RE_NUM = re.compile(r"\d+")
DISK_SUFFIX = (".out", ".out.gz", "_trj.xyz", ".hess", ".inp", ".engrad")


def gen(p):
    s = RE_T.sub("<J>", p)
    s = RE_SPEC.sub(lambda m: m.group(1) + "#_<F>", s)
    s = RE_RAND.sub("<N>", s)
    s = RE_BONDS.sub("_<b>", s)
    return RE_NUM.sub("#", s)


def _safe(name):
    parts = name.split("/")
    return not (name.startswith("/") or ".." in parts)


def _bang(text):
    return " ".join(" ".join(ln.split("#")[0].split()) for ln in text.splitlines() if ln.strip().startswith("!"))


def _member(m, tf, prefix, d, res, detail):
    name = m.name[2:] if m.name.startswith("./") else m.name
    full = prefix + name
    if m.isdir():
        res["n_dirs"] += 1
        return
    if m.issym() or m.islnk():
        res["links"].append([full, m.linkname])
        return
    if not m.isfile():
        res["other_types"].append([full, int(m.type[0]) if m.type else -1])
        return
    if not _safe(name):
        res["err"].append(f"unsafe name {full}")
        return
    b = tf.extractfile(m).read()
    res["n_files"] += 1
    res["bytes_unc"] += len(b)
    g = gen(full)
    pc = res["pat"].setdefault(g, [0, 0])
    pc[0] += 1
    pc[1] += len(b)
    if detail:
        res["inner"].append([full, len(b)])
    if name.endswith(".tar") and prefix == "":
        res["nested"].append([name, len(b)])
        try:
            with tarfile.open(fileobj=io.BytesIO(b), mode="r:") as t2:
                for m2 in t2:
                    _member(m2, t2, name + "::", d / ("_" + name), res, detail)
        except tarfile.TarError as e:
            res["err"].append(f"nested {name}: {e}")
        return
    if name.endswith(DISK_SUFFIX) and "_xtb." not in name:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        res["_disk"].append([full, str(p)])
    elif name.endswith(".out") and "_xtb." in name:
        res["n_xtb_out"] += 1


KEEP = ("job_type", "level_ok", "level_reason", "bang", "smd_block", "orca_version", "converged", "terminated_normally",
        "n_opt_cycles", "n_coord_blocks", "n_energies", "n_gradients", "n_steps", "kept", "reason", "n_atoms",
        "fmax_first", "fmax_last", "de_kcal", "trj_frames", "trj_ok", "trj_reason", "bytes")


def _dist(pos, pairs):
    return np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs])


def scan_archive(job, blob, ctx, detail, tmproot):
    """one archive.tgz (bytes) -> dict; ctx = None or dict(symbols, pos, pairs) of the label TS."""
    t0 = time.time()
    res = dict(job=job, arc_bytes=len(blob), n_files=0, n_dirs=0, bytes_unc=0, n_xtb_out=0, pat={}, nested=[], links=[],
               other_types=[], err=[], inner=[], _disk=[], orca=[], hess=[], trj_unpaired=[], inp={})
    d = Path(tempfile.mkdtemp(prefix=f"{job}_", dir=tmproot))
    try:
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for m in tf:
                    _member(m, tf, "", d, res, detail)
        except (tarfile.TarError, OSError, EOFError) as e:
            res["err"].append(f"archive: {type(e).__name__}: {e}")
        res["t_unpack_s"] = time.time() - t0
        t1 = time.time()
        outs = {p: f for f, p in res["_disk"] if f.endswith((".out", ".out.gz"))}
        lvl = {}
        for p, full in sorted(outs.items(), key=lambda x: x[1]):
            ta = time.time()
            rec, steps = I.parse_file(dict(path=p, source="nurion"))
            r = {k: rec.get(k) for k in KEEP if k in rec}
            r.update(file=full, pat=gen(full), parse_s=round(time.time() - ta, 3))
            lvl[I._stem(p)] = bool(rec.get("level_ok"))
            if rec.get("level_ok") and not steps and rec.get("n_gradients"):     # rejected label-level file: salvage?
                st2, _ = O.read_out_steps(p)
                r["n_steps_unkept"] = len(st2)
                steps_for_near = st2
            else:
                steps_for_near = steps
            if ctx and steps_for_near:
                same = [s["symbols"] == ctx["symbols"] for s in steps_for_near]
                r["same_symbols"] = all(same)
                if all(same):
                    dl = _dist(ctx["pos"], ctx["pairs"])
                    dd = np.array([np.abs(_dist(s["pos_A"], ctx["pairs"]) - dl).max() for s in steps_for_near])
                    r["n_near"] = int((dd <= NEAR_A).sum())
                    r["dform_min"] = float(dd.min())
                    r["dform_last"] = float(dd[-1])
                    r["dpos_last_vs_label"] = float(np.abs(steps_for_near[-1]["pos_A"] - ctx["pos"]).max())
            res["orca"].append(r)
        for full, p in res["_disk"]:
            if full.endswith(".hess"):
                h = dict(file=full, pat=gen(full), bytes=Path(p).stat().st_size)
                try:
                    hh = O.read_hess(p)
                    H = hh["hessian"]
                    h.update(n_atoms=len(hh["symbols"]), finite=bool(np.isfinite(H).all()),
                             asym=float(np.abs(H - H.T).max()), level_ok_out=lvl.get(p[: -len(".hess")]))
                    if ctx and hh["symbols"] == ctx["symbols"]:
                        pos = hh["pos_bohr"] * 0.529177210903
                        dl = _dist(ctx["pos"], ctx["pairs"])
                        h["dform_vs_label"] = float(np.abs(_dist(pos, ctx["pairs"]) - dl).max())
                except Exception as e:                       # noqa: BLE001
                    h["error"] = f"{type(e).__name__}: {e}"[:200]
                res["hess"].append(h)
            elif full.endswith("_trj.xyz"):
                stem = p[: -len("_trj.xyz")]
                if stem + ".out" not in outs and stem + ".out.gz" not in outs:
                    try:
                        nfr = len(I.read_trj(p))
                    except Exception:                        # noqa: BLE001
                        nfr = -1
                    res["trj_unpaired"].append([full, nfr])
            elif full.endswith(".inp"):
                bg = _bang(Path(p).read_text(errors="replace"))
                res["inp"].setdefault(gen(full), Counter())[bg] += 1
        res["t_parse_s"] = time.time() - t1
    finally:
        shutil.rmtree(d, ignore_errors=True)
    res["inp"] = {k: dict(v) for k, v in res["inp"].items()}
    del res["_disk"]
    res["t_total_s"] = time.time() - t0
    return res


def load_manifest(stem):
    out = {}
    f = MANI / f"{stem}.manifest.tsv"
    if f.exists():
        lines = f.read_text().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            r = dict(zip(hdr, ln.split("\t")))
            out[r["job_id"]] = r
    return out


class Ctx:
    def __init__(self):
        T = C.targets()
        self.t = {r.job_id: r for r in T.itertuples()}

    def get(self, job):
        r = self.t.get(job)
        if r is None:
            return None
        p = C.label_ts(r.rxn_id)
        if not p.exists():
            return dict(missing=str(p))
        sym, pos = C.read_xyz(p)
        return dict(symbols=sym, pos=np.array(pos), pairs=C.formed_pairs(r.formed_pairs_ts), rxn_id=int(r.rxn_id),
                    fold=int(r.fold_T1), dipole=r.dipole)


RE_OUT = re.compile(r"^(jobs|logs|attempts)/([^/]+?)(\.log)?(/|$)")


def agg_add(A, res):
    for g, (n, b) in res["pat"].items():
        a = A["pat"].setdefault(g, [0, 0])
        a[0] += n
        a[1] += b
    for r in res["orca"]:
        a = A["orca"].setdefault(r["pat"], dict(n=0, level_ok=0, term=0, kept=0, steps_kept=0, grads=0, cycles=0,
                                                grads_level=0, near=0, jt=Counter(), reasons=Counter(), bang=Counter(),
                                                bytes=0, parse_s=0.0))
        a["n"] += 1
        a["bytes"] += r.get("bytes") or 0
        a["parse_s"] += r.get("parse_s") or 0
        a["jt"][r.get("job_type")] += 1
        a["bang"][(r.get("bang") or "")[:140]] += 1
        a["grads"] += r.get("n_gradients") or 0
        a["cycles"] += r.get("n_opt_cycles") or 0
        if r.get("level_ok"):
            a["level_ok"] += 1
            a["grads_level"] += r.get("n_gradients") or 0
        else:
            a["reasons"][(r.get("level_reason") or "")[:60]] += 1
        a["term"] += bool(r.get("terminated_normally"))
        if r.get("kept"):
            a["kept"] += 1
            a["steps_kept"] += r.get("n_steps") or 0
        elif r.get("level_ok"):
            a["reasons"]["L:" + (r.get("reason") or "")[:70]] += 1
        a["near"] += r.get("n_near") or 0
    for h in res["hess"]:
        a = A["hess"].setdefault(h["pat"], dict(n=0, ok=0, level_ok_out=0, n_atoms=Counter()))
        a["n"] += 1
        a["ok"] += "error" not in h and bool(h.get("finite"))
        a["level_ok_out"] += bool(h.get("level_ok_out"))
        a["n_atoms"][h.get("n_atoms")] += 1
    A["n_jobs"] += 1
    A["per_job"].append([res["job"], sum(r.get("n_steps") or 0 for r in res["orca"] if r.get("kept")),
                         sum(r.get("n_near") or 0 for r in res["orca"]), round(res["arc_bytes"] / 1e6, 2),
                         round(res.get("t_total_s", 0), 2), len(res["orca"]), len(res["hess"]),
                         sum(r.get("n_gradients") or 0 for r in res["orca"] if r.get("level_ok"))])
    A["arc_bytes"] += res["arc_bytes"]
    A["bytes_unc"] += res["bytes_unc"]
    A["n_files"] += res["n_files"]
    A["t_unpack_s"] += res.get("t_unpack_s", 0)
    A["t_parse_s"] += res.get("t_parse_s", 0)
    ks = sum(r.get("n_steps") or 0 for r in res["orca"] if r.get("kept"))
    A["steps_per_job"].append(ks)
    A["near_per_job"].append(sum(r.get("n_near") or 0 for r in res["orca"]))
    A["errors"] += res["err"][:3]
    A["links"] += res["links"][:3]


def new_agg():
    return dict(n_jobs=0, arc_bytes=0, bytes_unc=0, n_files=0, t_unpack_s=0.0, t_parse_s=0.0, pat={}, orca={}, hess={},
                steps_per_job=[], near_per_job=[], errors=[], links=[], per_job=[])


def finish_agg(A):
    for a in A["orca"].values():
        a["jt"] = dict(a["jt"])
        a["reasons"] = dict(a["reasons"].most_common(6))
        a["bang"] = dict(a["bang"].most_common(3))
        a["parse_s"] = round(a["parse_s"], 2)
    for a in A["hess"].values():
        a["n_atoms"] = dict(a["n_atoms"])
    s = np.array(A["steps_per_job"] or [0])
    A["steps_per_job_q"] = [int(np.quantile(s, q)) for q in (0, 0.25, 0.5, 0.75, 1.0)] + [float(s.mean())]
    A["steps_total"] = int(s.sum())
    nn = np.array(A["near_per_job"] or [0])
    A["near_total"] = int(nn.sum())
    del A["steps_per_job"], A["near_per_job"]
    A["pat"] = dict(sorted(A["pat"].items(), key=lambda x: -x[1][0]))
    return A


def run(args):
    tar = Path(args.tar)
    stem = tar.name[:-4] if tar.name.endswith(".tar") else tar.name
    mani = load_manifest(stem)
    ctx = Ctx()
    tmproot = Path(tempfile.mkdtemp(prefix="nurscout_", dir=os.environ.get("TMPDIR", "/tmp")))
    detail = args.mode == "anatomy"
    outer = {}
    outer_first = []
    small = defaultdict(dict)
    job_order = []
    jobs_res = []
    A = new_agg()
    t0 = time.time()
    n_members = 0
    last_off = 0
    stopped = None
    stale_arch = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.mode == "count" else None
    inflight = set()
    t_read = 0.0
    try:
        with tarfile.open(tar, mode="r|") as tf:
            for m in tf:
                n_members += 1
                name = m.name[2:] if m.name.startswith("./") else m.name
                last_off = m.offset_data + m.size
                if len(outer_first) < 80:
                    outer_first.append([name, m.size, m.type.decode() if isinstance(m.type, bytes) else str(m.type)])
                g = gen(name)
                oc = outer.setdefault(g, [0, 0])
                oc[0] += 1
                oc[1] += m.size
                mm = RE_OUT.match(name)
                job = mm.group(2) if mm else None
                if job and job not in job_order and mm.group(1) == "jobs":
                    job_order.append(job)
                in_scope = job is not None and (args.jobs == 0 or (job in job_order[: args.jobs]))
                if detail and args.jobs and job and mm.group(1) == "jobs" and not in_scope:
                    stopped = f"after {args.jobs} jobs (member {n_members})"
                    break
                if args.budget and time.time() - t0 > args.budget:
                    stopped = f"budget {args.budget}s (member {n_members}, offset {last_off})"
                    break
                if not m.isfile():
                    continue
                if mm and mm.group(1) == "jobs" and re.search(r"/archive[^/]*\.tgz$", name):
                    if not name.endswith("/archive.tgz"):
                        stale_arch.append([name, m.size])
                        continue
                    tr = time.time()
                    blob = tf.extractfile(m).read()
                    t_read += time.time() - tr
                    c = ctx.get(job)
                    c = c if c and "missing" not in c else None
                    if pool is None:
                        res = scan_archive(job, blob, c, detail, tmproot)
                        res["manifest"] = mani.get(job)
                        res["target"] = None if c is None else dict(rxn_id=c["rxn_id"], fold=c["fold"], dipole=c["dipole"])
                        jobs_res.append(res)
                        agg_add(A, res)
                    else:
                        while len(inflight) >= 3 * args.workers:
                            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                            for f in done:
                                agg_add(A, f.result())
                        inflight.add(pool.submit(scan_archive, job, blob, c, False, tmproot))
                elif in_scope and detail and m.size < 2_000_000:
                    b = tf.extractfile(m).read()
                    small[job][name] = b[:600].decode(errors="replace") if not name.endswith(".log") else \
                        (b[:400] + b"\n...\n" + b[-400:]).decode(errors="replace")
                    small[job][name + "#size"] = m.size
        t_stream = time.time() - t0
        if pool is not None:
            for f in inflight:
                agg_add(A, f.result())
    finally:
        if pool is not None:
            pool.shutdown()
        shutil.rmtree(tmproot, ignore_errors=True)
    t_all = time.time() - t0
    A = finish_agg(A)
    per_job = A.pop("per_job")
    by_cls = defaultdict(lambda: [0, 0, 0, 0])               # class -> jobs, kept steps, near, level-ok gradients
    for j, ks, nn, *_rest in per_job:
        b = by_cls[(mani.get(j) or {}).get("class", "?")]
        b[0] += 1
        b[1] += ks
        b[2] += nn
        b[3] += _rest[-1]
    A["by_class"] = dict(by_cls)
    A["t_job_q"] = [round(float(np.quantile([p[4] for p in per_job] or [0], q)), 2) for q in (0, 0.5, 0.9, 1.0)]
    cls = Counter((mani.get(r["job"]) or {}).get("class", "?") for r in jobs_res) if detail else None
    summ = dict(tar=str(tar), tar_bytes=tar.stat().st_size, mode=args.mode, host=os.uname().nodename,
                tmpdir=os.environ.get("TMPDIR", "/tmp"), workers=args.workers, n_members=n_members,
                bytes_through=last_off, frac_of_tar=round(last_off / tar.stat().st_size, 4), stopped=stopped,
                t_stream_s=round(t_stream, 1), t_all_s=round(t_all, 1), t_read_archives_s=round(t_read, 1),
                n_manifest=len(mani), n_jobs_seen=len(job_order), stale_archives=stale_arch[:10],
                n_stale_archives=len(stale_arch), outer_patterns=dict(sorted(outer.items(), key=lambda x: -x[1][0])),
                agg=A, classes=dict(cls) if cls else None)
    full = dict(summary=summ, outer_first=outer_first, per_job=per_job, jobs=jobs_res if detail else None,
                small={k: v for k, v in small.items()} if detail else None)
    out = OUTDIR / f"scout1_{stem}_{args.mode}{'_j' + str(args.jobs) if detail else ''}.json"
    C.write_json(out, full)
    summ["written"] = str(out)
    print(json.dumps(summ, indent=1, default=C._json_default)[:60000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["anatomy", "count"])
    ap.add_argument("tar")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget", type=float, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
