#!/usr/bin/env python3
"""DEPLOY-MODEL Phase L (docs/specs/DEPLOY_MODEL.md 3): label-free finalisation of G2 autodE work folders.
Run through sbatch only (jobs/cpu_d1.slurm = the D1 run env, G2ENV=d1 of analysis/d1_g2/jobs/g2.slurm).

  python finalize_lf.py run <i|@I> <n> <cores> [--jobs J1,J2,..|ids.csv]   # D1 targets [i::n]: $G2/jobs -> $DEP/g2lf_final
  python finalize_lf.py report [--jobs ...]                                # results/L_report.json, results/L_status.csv
  python finalize_lf.py selftest                                           # C-7 -> results/C7_finalize_selftest.json
  exit 0 ok, 2 not ready / not finished (resubmit), 3 STOP (L-1, L-2, L-4 or C-7 failed)

finalize_lf(job_row, src_dir, out_root, cores) is also the deployment finaliser (spec 3-1, 14-2). It is
analysis/d1_g2/g2_autode.py finalize() without the label: no re-mapping (autodE atom order), formed pairs and fragment
= the run's own ts_result.json formed_pairs_ts / A_idx, charge = job_row["charge"] (manifest), multiplicity 1.
Imported, not copied (paths in the D1 env: $PILOT = /gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/
d1_autode_pilot):
  g2_autode.py   FREQ_HEADER (l.62), CONFIG (l.60, config_g2.yaml: orca_bin / maxcore_mb 3000), Fail (l.362);
                 the import also puts $PILOT and d1_run on sys.path (l.53-58)
  $PILOT/orca_direct.py  write_inp (l.121), run (l.131), terminated (l.143), last_fspe (l.151), parse_hess (l.164),
                         imag_and_mode (l.186)
  $PILOT/run_ts.py       forming_share (l.64)
  $PILOT/pilot_common.py done (l.78), failed (l.82), read_xyz (l.119), write_xyz (l.130), _jsonable (l.103)
  g2_common.py   MANIFEST_SHA (l.26)
Mirrored (inline code in g2_autode.finalize, cannot be imported): the stage-failure reasons (l.266-274:
'<stage>_<text before the first colon>' or '<stage>_not_run'), the marker name (l.354: '.fail_' + the first reason
with [^A-Za-z0-9_]+ -> '_'), the 'freq_not_terminated' / 'hess_no_imaginary_frequency' / 'error_<Exception>' reasons
(l.319-334), the ts/stereo/ref result subsets (l.275-281) and the tmp folder -> os.replace publication (l.339-358).
Decisions (recommended options, recorded): ORCA runs inside out_root/<rid>.tmp/work (every write stays under $DEP;
ORCA scratch files other than ts.inp/ts.out/ts.hess(+ts.err on failure) are deleted before publication); ts.hess is a
hard link of work/ts.hess (identical bytes, the state G_compact left in $G2/g2_final; copy if linking fails);
work/ts.out stays plain text (spec 3-1 item 7); freq failures keep work/ts.inp, ts.out, ts.err (and ts.hess); the run
pool sets OMP_NUM_THREADS = SLURM_CPUS_PER_TASK = cores in each worker as g2_autode.cmd_one did before finalize.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "analysis" / "d1_g2"))
import g2_autode as GA  # noqa: E402            (import-time: D1 pilot code on sys.path, autodE patches unused here)

pc, RT, OD, G = GA.pc, GA.RT, GA.OD, GA.G

MAXCORE_MB = 3000                                    # spec 3-1 item 4 (= config_g2.yaml maxcore_mb, asserted in cfg())
MULT = 1
STAGES = ("ts", "stereo", "ref")
SRC_FILES = ("ts.xyz", "ts_result.json", "ref_dipole.xyz", "ref_dipolarophile.xyz", "ref_result.json",
             "stereo_result.json")
SUBSET_KEYS = ("autode_version", "keywords", "n_cores", "imag_freqs_cm", "formed_pairs_ts", "formed_d", "checks",
               "A_idx", "status", "found", "requested", "alt_used", "source", "attempts", "n_orca_outputs", "xtb_calls",
               "wall_s")                             # = g2_autode.finalize l.278-281
RESULT_KEYS = ("rxn_id", "job_id", "status", "fail_reasons", "formed_pairs_ts", "A_idx", "imag_freqs_cm", "n_imag",
               "imag_mode_forming_share", "d_form", "e_ts_freq_eh", "remapped", "source_dir", "inputs_sha256",
               "freq_header", "core_h")              # spec 3-1 item 8
DONE_FILES = ("ts.xyz", "rel1.xyz", "rel2.xyz", "ts.hess", "work/ts.hess", "work/ts.inp", "work/ts.out", "result.json",
              ".done")
KEEP_WORK_OK = ("ts.inp", "ts.out", "ts.hess")
KEEP_WORK_FAIL = ("ts.inp", "ts.out", "ts.err", "ts.hess")
LEVEL = ("G2 label-free (DEPLOY-MODEL L): autodE 1.4.5 work folder as is (autodE atom order, own formed pairs / A_idx); "
         "ORCA XTB2 ALPB(water) Freq at ts.xyz")
G_STATUS = C.REPO / "analysis" / "d1_g2" / "results" / "G_status.csv"
LABEL_META = C.G2 / "G" / "input_meta_d1_g2.csv"      # label A_idx (sp/eda.inp (1) atoms, label order) - L-3 only
N_TARGETS, N_G2_OK = 9266, 8616
TOL_PERM_A = 1e-6                                    # L-2
L1_MAX_FREQ_FAIL = 0.005
L4_TOL_CM, L4_MAX_FRAC = 1.0, 0.01
STOP_AFTER_H = float(os.environ.get("DEP_STOP_AFTER_H", "44"))
T0 = float(os.environ.get("DEP_JOB_T0", time.time()))
_CFG = None


def cfg():
    """config_g2.yaml (read only): OD.run takes orca_bin from it when $ORCA_BIN is unset."""
    global _CFG
    if _CFG is None:
        _CFG = yaml.safe_load(Path(GA.CONFIG).read_text())
        if int(_CFG["maxcore_mb"]) != MAXCORE_MB:
            C.stop(f"{GA.CONFIG} maxcore_mb {_CFG['maxcore_mb']} != {MAXCORE_MB} (spec 3-1 item 4)")
    return _CFG


def rxn_key(job_row):
    """output folder name: rxn_id / meta_rxn_id of the row (D1 manifest: meta_rxn_id), else the job_id."""
    for k in ("rxn_id", "meta_rxn_id"):
        v = job_row.get(k)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v)):
            continue
        return int(float(v))
    return str(job_row["job_id"])


def marker(d):
    """'.done', the first sorted '.fail_*' name, or None (= xtb_slice.geom_marker)."""
    d = Path(d)
    if (d / ".done").is_file():
        return ".done"
    f = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
    return f[0] if f else None


def marker_name(reason):
    return ".fail_" + re.sub(r"[^A-Za-z0-9_]+", "_", reason)          # = g2_autode.finalize l.354


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _clean_work(wd, keep):
    for p in list(wd.iterdir()):
        if p.name in keep:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()


# ================================================================ spec 3-1
def finalize_lf(job_row, src_dir, out_root, cores):
    """out_root/<rxn_id>/ (G1 folder format, autodE atom order) from a G2 autodE work folder; returns the status
    ('ok', 'fail:<reason>' or 'skip<marker>' when the folder is already final)."""
    job_row, src, root, cores = dict(job_row), Path(src_dir), Path(out_root), int(cores)
    rid = rxn_key(job_row)
    out, tmp = root / str(rid), root / f"{rid}.tmp"
    C.assert_write_ok(tmp)
    m = marker(out)
    if m is not None:
        return "skip" + m                                             # idempotent (item 7)
    res = dict(rxn_id=rid, job_id=str(job_row["job_id"]), level=LEVEL, code="analysis/deploy/finalize_lf.py",
               cores=cores, source_dir=os.path.abspath(src), remapped=False)
    reasons, times = [], {}
    for st in STAGES:                                                 # item 1 (= g2_autode.finalize l.266-274)
        f = pc.failed(src, st)
        if f is not None:
            reasons.append(f"{st}_" + (f.split(":")[0] if f else "fail"))
            res[f"{st}_fail"] = f
            break
        if not pc.done(src, st):
            reasons.append(f"{st}_not_run")
            break
    res["inputs_sha256"] = {f: C.sha256(src / f) for f in SRC_FILES if (src / f).is_file()}
    for f in ("ts_result.json", "stereo_result.json", "ref_result.json"):
        r = _read_json(src / f)
        if isinstance(r, dict):
            res[f.split(".")[0]] = {k: r.get(k) for k in SUBSET_KEYS if k in r}
    if tmp.exists():
        shutil.rmtree(tmp)                                            # stale partial build (crash / wall time)
    wd = tmp / "work"
    wd.mkdir(parents=True)
    ts = rels = None
    if not reasons:
        try:
            ts = pc.read_xyz(src / "ts.xyz")                          # item 2: autodE order, no re-mapping
            rels = [pc.read_xyz(src / "ref_dipole.xyz"), pc.read_xyz(src / "ref_dipolarophile.xyz")]
            tsr = json.loads((src / "ts_result.json").read_text())
            fp = sorted(tuple(sorted(int(x) for x in p)) for p in tsr["formed_pairs_ts"])     # item 3
            A = sorted(int(i) for i in tsr["A_idx"])
            X = np.asarray(ts[1], dtype=float)
            D = np.linalg.norm(X[:, None] - X[None], axis=-1)
            charge = int(job_row["charge"])
            hdr = GA.FREQ_HEADER.format(n=cores, m=MAXCORE_MB)
            res.update(formed_pairs_ts=[list(p) for p in fp], A_idx=A, d_form=[float(D[p]) for p in fp],
                       n_atoms=len(ts[0]), charge=charge, mult=MULT, freq_header=GA.FREQ_HEADER.strip().splitlines()[0],
                       freq_input_header=hdr)
            inp = wd / "ts.inp"
            OD.write_inp(inp, hdr, *ts, charge=charge, mult=MULT)    # item 4: ts.xyz as is, no optimisation
            t1 = time.time()
            outp = OD.run(cfg(), inp)
            times["freq"] = time.time() - t1
            txt = outp.read_text(errors="replace") if outp.is_file() else ""
            res.update(orca_terminated=OD.terminated(txt), e_ts_freq_eh=OD.last_fspe(txt))
            if not OD.terminated(txt) or not (wd / "ts.hess").is_file():
                raise GA.Fail("freq_not_terminated")                  # item 6
            imag, mode = OD.imag_and_mode(wd / "ts.hess")
            heavy = [i for i, s in enumerate(ts[0]) if s != "H"]
            share = None if mode is None else RT.forming_share(X, mode, fp, heavy)     # item 5: own pairs
            res.update(imag_freqs_cm=imag, n_imag=len(imag), imag_mode_forming_share=share)
            if not imag:
                raise GA.Fail("hess_no_imaginary_frequency")
        except GA.Fail as e:
            reasons.append(str(e))
        except Exception as e:                                        # noqa: BLE001
            res["traceback"] = traceback.format_exc()
            reasons.append(f"error_{type(e).__name__}")
    _clean_work(wd, KEEP_WORK_OK if not reasons else KEEP_WORK_FAIL)
    res["wall_s"] = times
    res["core_h"] = {k: v * cores / 3600 for k, v in times.items()}
    res["core_h_total"] = float(sum(res["core_h"].values()))
    res["fail_reasons"] = reasons
    res["status"] = "ok" if not reasons else "fail:" + reasons[0]
    if not reasons:
        pc.write_xyz(tmp / "ts.xyz", *ts, f"G2 TS rxn {rid} (autodE order, label-free)")
        for k, g in enumerate(rels, 1):
            pc.write_xyz(tmp / f"rel{k}.xyz", *g, f"G2 reference {k} rxn {rid} (autodE order, label-free)")
        try:
            os.link(wd / "ts.hess", tmp / "ts.hess")
        except OSError:
            shutil.copy(wd / "ts.hess", tmp / "ts.hess")
    (tmp / "result.json").write_text(json.dumps(res, indent=1, default=pc._jsonable))
    mk = ".done" if not reasons else marker_name(reasons[0])
    (tmp / mk).write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n" + "\n".join(reasons) + "\n")
    if out.exists():
        shutil.rmtree(out)                                            # a folder without marker (never published here)
    os.replace(tmp, out)                                              # item 7: atomic publication
    return res["status"]


# ================================================================ targets
def load_targets(jobs=None):
    """the 9,266 D1 targets (analysis/d1_g2/results/G_status.csv) in job_id order; jobs = optional subset."""
    with open(G_STATUS, newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["job_id"])
    if len(rows) != N_TARGETS or len({r["job_id"] for r in rows}) != N_TARGETS:
        C.stop(f"{G_STATUS}: {len(rows)} rows, expected {N_TARGETS} distinct job_ids")
    if jobs:
        want = set(jobs)
        rows = [r for r in rows if r["job_id"] in want]
        if len(rows) != len(want):
            C.stop(f"--jobs not in G_status.csv: {sorted(want - {r['job_id'] for r in rows})}")
    return rows


def load_manifest():
    """job_id -> manifest row (same read as pilot_common.load_job); sha256 = g2_common.MANIFEST_SHA (start 0-7)."""
    s = C.sha256(C.D1_MANIFEST)
    if s != G.MANIFEST_SHA:
        C.stop(f"{C.D1_MANIFEST} sha256 {s} != g2_common.MANIFEST_SHA {G.MANIFEST_SHA}")
    m = pd.read_csv(C.D1_MANIFEST, keep_default_na=False)
    return {str(r["job_id"]): r for r in m.to_dict("records")}


def parse_jobs(a):
    """--jobs J1,J2 or a csv file with a job_id column."""
    if "--jobs" not in a:
        return a, None
    k = a.index("--jobs")
    v = a[k + 1]
    rest = a[:k] + a[k + 2:]
    if Path(v).is_file():
        return rest, pd.read_csv(v, dtype=str).job_id.tolist()
    return rest, [x for x in v.split(",") if x]


def _init_worker(cores):
    os.environ["SLURM_CPUS_PER_TASK"] = os.environ["OMP_NUM_THREADS"] = str(cores)   # as g2_autode.cmd_one


def _one(job_row, cores):
    T = str(job_row["job_id"])
    if marker(C.LF_FINAL / str(rxn_key(job_row))) is None and time.time() - T0 > STOP_AFTER_H * 3600:
        return T, "not_started", 0.0
    t = time.time()
    st = finalize_lf(job_row, C.G2_JOBS / T, C.LF_FINAL, cores)
    return T, st, time.time() - t


def cmd_run(i, n, cores, jobs=None):
    i, n, cores = C.array_task(i), int(n), int(cores)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    n_par = max(1, cpus // cores)
    tg = load_targets(jobs)[i::n]
    man = load_manifest()
    C.assert_write_ok(C.LF_FINAL / "x")
    C.LF_FINAL.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tg:
        r = man.get(t["job_id"])
        if r is None or int(r["meta_rxn_id"]) != int(t["rxn_id"]):
            C.stop(f"{t['job_id']}: manifest row missing or meta_rxn_id != G_status rxn_id {t['rxn_id']}")
        rows.append(r)
    print(f"[finalize_lf run] slice {i}/{n}: {len(rows)} reactions, {n_par} at a time x {cores} cores "
          f"(cpus {cpus}); {C.G2_JOBS} -> {C.LF_FINAL}", flush=True)
    cnt, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=n_par, initializer=_init_worker, initargs=(cores,)) as ex:
        futs = {ex.submit(_one, r, cores): str(r["job_id"]) for r in rows}
        for f in as_completed(futs):
            try:
                T, st, w = f.result()
            except Exception as e:                                    # noqa: BLE001  (worker crash)
                T, st, w = futs[f], f"error_worker:{type(e).__name__}:{e}", 0.0
            key = st.split(":")[0] if st.startswith(("fail", "error")) else st
            cnt[key] = cnt.get(key, 0) + 1
            print(f"  {T}: {st}  {w:.1f} s  ({sum(cnt.values())}/{len(rows)}, {time.time() - t0:.0f} s)", flush=True)
    print(f"[finalize_lf run] {cnt}", flush=True)
    return 2 if cnt.get("not_started") or any(k.startswith("error") for k in cnt) else 0


# ================================================================ spec 3-3 report
def match_perm(a, b, tol=TOL_PERM_A):
    """p with b[p[i]] == a[i]: same element and |x_a - x_b| <= tol (A), one candidate per atom, a bijection.
    -> (p | None, max deviation of the matched atoms | None, why)."""
    sa, xa = list(a[0]), np.asarray(a[1], dtype=float)
    sb, xb = list(b[0]), np.asarray(b[1], dtype=float)
    if len(sa) != len(sb):
        return None, None, "n_atoms"
    if sorted(sa) != sorted(sb):
        return None, None, "elements"
    D = np.linalg.norm(xa[:, None] - xb[None], axis=-1)
    eb = np.array(sb)
    p = []
    for k, s in enumerate(sa):
        c = np.where((D[k] <= tol) & (eb == s))[0]
        if len(c) != 1:
            return None, float(D[k][eb == s].min()), f"atom_{k}_candidates_{len(c)}"
        p.append(int(c[0]))
    if len(set(p)) != len(p):
        return None, None, "not_bijective"
    return p, float(max(D[k, p[k]] for k in range(len(p)))), ""


def load_label_A():
    m = pd.read_csv(LABEL_META, dtype={"A_idx": str})
    return {int(r.rxn_id): sorted(int(x) for x in str(r.A_idx).split()) for r in m.itertuples()}


def _same_file(a, b):
    try:
        if os.stat(a).st_ino == os.stat(b).st_ino:
            return True
        return Path(a).read_bytes() == Path(b).read_bytes()
    except OSError:
        return False


def status_row(t, labA):
    """one L_status.csv row (L-1 .. L-4 per reaction)."""
    rid = int(t["rxn_id"])
    d, g = C.LF_FINAL / str(rid), C.G2_FINAL / str(rid)
    row = dict(rxn_id=rid, job_id=t["job_id"], dipole=t.get("dipole"), skeleton=t.get("skeleton"),
               n_atoms=t.get("n_atoms"), g2_status=t["status"], g2_marker=t["marker"], g2_remapped=t.get("remapped"))
    mk = marker(d)
    row["lf_marker"] = mk
    if mk is None:
        row["lf_status"] = "missing"
        return row
    res = _read_json(d / "result.json") or {}
    row.update(lf_status=res.get("status"), lf_fail_reason=(res.get("fail_reasons") or [None])[0],
               core_h_freq=(res.get("core_h") or {}).get("freq"), wall_freq_s=(res.get("wall_s") or {}).get("freq"))
    if mk != ".done":
        return row
    prob = [f"missing:{f}" for f in DONE_FILES if not (d / f).is_file()]
    prob += [f"key:{k}" for k in RESULT_KEYS if k not in res]
    if not _same_file(d / "ts.hess", d / "work" / "ts.hess"):
        prob.append("ts.hess!=work/ts.hess")
    imag = res.get("imag_freqs_cm") or []
    if res.get("status") != "ok" or not imag or res.get("remapped") is not False:
        prob.append("result_status")
    try:
        same_in = all(C.sha256(Path(res["source_dir"]) / f) == s for f, s in res["inputs_sha256"].items())
    except (OSError, KeyError, TypeError):
        same_in = False
    row.update(done_complete=not prob, done_problems=";".join(prob), inputs_unchanged=same_in,
               lf_n_imag=len(imag), lf_imag1_cm=imag[0] if imag else None,
               lf_share=res.get("imag_mode_forming_share"))
    if prob:
        return row
    # ---- L-2: same (element, coordinates) sets as $G2/g2_final (label order)
    if not (g / ".done").is_file():
        row["l2"] = "fail:no_g2_final_done"
        return row
    gres = _read_json(g / "result.json") or {}
    perms, bad = {}, []
    for name in ("ts", "rel1", "rel2"):
        try:
            p, dev, why = match_perm(pc.read_xyz(d / f"{name}.xyz"), pc.read_xyz(g / f"{name}.xyz"))
        except Exception as e:                                        # noqa: BLE001
            p, dev, why = None, None, f"read:{type(e).__name__}"
        perms[name] = p
        row[f"l2_{name}_dev_A"] = dev
        row[f"l2_{name}_identity"] = None if p is None else p == list(range(len(p)))
        if p is None:
            bad.append(f"{name}:{why}")
    row["l2"] = "ok" if not bad else "fail:" + ",".join(bad)
    # ---- L-3: own formed pairs / A_idx through the TS permutation vs LADDER formed_pairs_label / label A_idx
    p = perms["ts"]
    if p is not None:
        fp = sorted(tuple(sorted((p[i], p[j]))) for i, j in res["formed_pairs_ts"])
        fl = sorted(tuple(sorted(int(x) for x in q)) for q in (gres.get("formed_pairs_label") or []))
        A = sorted(p[a] for a in res["A_idx"])
        row["l3_fp_same"] = fp == fl
        row["l3_A_same"] = None if labA is None else A == labA
        row["l3_A_is_label_B"] = None if labA is None else A == sorted(set(range(len(p))) - set(labA))
    # ---- L-4: lowest imaginary frequency vs $G2/g2_final result.json
    gi = gres.get("imag_freqs_cm") or []
    row["g2_imag1_cm"] = gi[0] if gi else None
    row["g2_n_imag"] = len(gi)
    row["l4_d_imag_cm"] = abs(float(imag[0]) - float(gi[0])) if gi else None
    gs, ls = gres.get("imag_mode_forming_share"), res.get("imag_mode_forming_share")
    row["g2_share"] = gs
    row["l4_d_share"] = abs(float(ls) - float(gs)) if gs is not None and ls is not None else None
    return row


def _flag(x):
    """True / False / None from a bool-ish cell (None, NaN -> None)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return bool(x)


def _q(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def cmd_report(jobs=None):
    tg = load_targets(jobs)
    labA = load_label_A()
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    if cpus > 1:
        with ProcessPoolExecutor(max_workers=min(cpus, 16)) as ex:
            rows = list(ex.map(status_row, tg, [labA.get(int(t["rxn_id"])) for t in tg], chunksize=32))
    else:
        rows = [status_row(t, labA.get(int(t["rxn_id"]))) for t in tg]
    S = pd.DataFrame(rows)
    for c in ("lf_status", "lf_fail_reason", "core_h_freq", "wall_freq_s", "done_complete", "done_problems",
              "inputs_unchanged", "lf_n_imag", "lf_imag1_cm", "lf_share", "l2", "l3_fp_same", "l3_A_same",
              "l3_A_is_label_B", "g2_imag1_cm", "g2_n_imag", "l4_d_imag_cm", "g2_share", "l4_d_share"):
        if c not in S:
            S[c] = None
    C.write_csv(C.RES / "L_status.csv", S)
    ok, done = S.g2_status == "ok", S.lf_marker == ".done"
    missing = S.rxn_id[S.lf_marker.isna()].tolist()
    rep = dict(spec="docs/specs/DEPLOY_MODEL.md 3-3", lf_root=str(C.LF_FINAL), g2_jobs=str(C.G2_JOBS),
               g2_final=str(C.G2_FINAL), subset=sorted(jobs) if jobs else None, n_targets=len(S),
               complete=not missing, n_missing=len(missing), missing=missing[:50],
               lf_markers={k: int(v) for k, v in S.lf_marker.fillna("missing").value_counts().items()})
    # ---- L-1
    n_ok = int(ok.sum())
    fail_ok = S[ok & ~done & S.lf_marker.notna()]
    stage_mask = ~ok & S.g2_marker.str.match(r"^\.fail_(ts|stereo|ref)_")
    stage_diff = S[stage_mask & (S.lf_marker != S.g2_marker)]
    incomplete = S[done & (S.done_complete != True)]                                         # noqa: E712
    n_freq_fail = int(n_ok - (ok & done).sum())
    l1 = dict(n_g2_ok=n_ok, n_done=int(done.sum()), n_done_g2_ok=int((ok & done).sum()),
              n_done_g2_not_ok=int((~ok & done).sum()), done_g2_not_ok=S.rxn_id[~ok & done].tolist()[:50],
              n_freq_fail=n_freq_fail, freq_fail_frac=n_freq_fail / n_ok if n_ok else 0.0,
              freq_fail_reasons={str(k): int(v) for k, v in fail_ok.lf_marker.value_counts().items()},
              freq_fail_ids=fail_ok.rxn_id.tolist()[:50], max_freq_fail_frac=L1_MAX_FREQ_FAIL,
              n_stage_fail=int(stage_mask.sum()), n_stage_marker_differs=len(stage_diff),
              stage_marker_differs=stage_diff[["rxn_id", "g2_marker", "lf_marker"]].to_dict("records")[:20],
              n_done_incomplete=len(incomplete),
              done_incomplete=incomplete[["rxn_id", "done_problems"]].to_dict("records")[:20],
              n_inputs_changed=int((done & (S.inputs_unchanged == False)).sum()),             # noqa: E712
              g2_not_ok_lf_markers={f"{a} -> {b}": int(v) for (a, b), v in
                                    S[~ok].groupby(["g2_marker", S.lf_marker[~ok].fillna("missing")]).size().items()})
    if not jobs:
        l1["n_g2_ok_expected"] = N_G2_OK
    l1["pass"] = bool(rep["complete"] and l1["n_done"] == n_ok - n_freq_fail and l1["freq_fail_frac"] <= L1_MAX_FREQ_FAIL
                      and not len(stage_diff) and not len(incomplete) and (jobs or n_ok == N_G2_OK))
    # ---- L-2
    D = S[done & (S.done_complete == True)]                                                  # noqa: E712
    for k in ("ts", "rel1", "rel2"):
        for c in (f"l2_{k}_identity", f"l2_{k}_dev_A"):
            if c not in D:
                D = D.assign(**{c: None})
    l2bad = D[D.l2 != "ok"]
    is_id = lambda k, v: D[f"l2_{k}_identity"].map(lambda x: _flag(x) is v)                  # noqa: E731
    nonid = D[(D.l2 == "ok") & is_id("ts", False)]
    g2rem = set(S.rxn_id[S.g2_remapped.astype(str) == "True"])
    l2 = dict(n_checked=len(D), n_pass=int((D.l2 == "ok").sum()), n_fail=len(l2bad),
              fail=l2bad[["rxn_id", "l2"]].to_dict("records")[:50], tol_A=TOL_PERM_A,
              max_dev_A={k: _q(D[f"l2_{k}_dev_A"].astype(float).tolist()).get("max") for k in ("ts", "rel1", "rel2")},
              n_ts_perm_identity=int(is_id("ts", True).sum()),
              n_ts_perm_not_identity=len(nonid), ts_perm_not_identity=nonid.rxn_id.tolist()[:50],
              g2_remapped_done=sorted(g2rem & set(D.rxn_id)),
              ts_perm_not_identity_equals_g2_remapped=set(nonid.rxn_id) == (g2rem & set(D.rxn_id)),
              n_rel_perm_not_identity={k: int(is_id(k, False).sum()) for k in ("rel1", "rel2")})
    l2["pass"] = bool(rep["complete"] and l2["n_fail"] == 0 and len(D) == int(done.sum()))
    # ---- L-3 (report only)
    E = D[D.l2 == "ok"]
    fpd, Ad = E[E.l3_fp_same != True], E[E.l3_A_same != True]                                # noqa: E712
    l3 = dict(n_compared=len(E), n_fp_differs=len(fpd), n_A_differs=len(Ad),
              n_A_is_label_B=int((E.l3_A_is_label_B == True).sum()),                           # noqa: E712
              n_any_differs=int(((E.l3_fp_same != True) | (E.l3_A_same != True)).sum()),       # noqa: E712
              fp_differs=fpd.rxn_id.tolist()[:100], A_differs=Ad.rxn_id.tolist()[:100],
              note="not a STOP (spec 3-3); cause analysis in Phase Q")
    # ---- L-4
    dn = E.l4_d_imag_cm.astype(float)
    n_gt = int((dn > L4_TOL_CM).sum())
    l4 = dict(n=int(dn.notna().sum()), d_imag_cm=_q(dn.tolist()), n_gt_1cm=n_gt,
              frac_gt_1cm=n_gt / max(1, int(dn.notna().sum())), gt_1cm=E.rxn_id[dn > L4_TOL_CM].tolist()[:50],
              n_imag_differs=int((E.lf_n_imag != E.g2_n_imag).sum()) if "g2_n_imag" in E else None,
              d_share=_q(E.l4_d_share.astype(float).tolist()), tol_cm=L4_TOL_CM, max_frac=L4_MAX_FRAC)
    l4["pass"] = bool(rep["complete"] and l4["n"] == len(E) and l4["frac_gt_1cm"] <= L4_MAX_FRAC)
    ch = S.core_h_freq[done].astype(float)
    rep.update(L1=l1, L2=l2, L3=l3, L4=l4,
               budget=dict(core_h_freq_total=float(ch.sum()), core_h_freq=_q(ch.tolist()),
                           wall_freq_s=_q(S.wall_freq_s[done].astype(float).tolist())),
               inputs={str(G_STATUS): C.sha256(G_STATUS), str(LABEL_META): C.sha256(LABEL_META)},
               written=time.strftime("%F %T"))
    rep["all_pass"] = bool(l1["pass"] and l2["pass"] and l4["pass"])
    C.write_json(C.RES / "L_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("n_targets", "complete", "lf_markers", "all_pass")}, indent=1), flush=True)
    for k in ("L1", "L2", "L3", "L4"):
        print(k, json.dumps({a: b for a, b in rep[k].items() if not isinstance(b, (list, dict)) or a in
                             ("freq_fail_reasons", "d_imag_cm", "max_dev_A")}), flush=True)
    if not rep["complete"]:
        C.not_ready(f"{len(missing)} targets have no .done / .fail_* under {C.LF_FINAL}")
    if not rep["all_pass"]:
        C.stop("L gates failed: " + ", ".join(k for k in ("L1", "L2", "L4") if not rep[k]["pass"]))
    return 0


# ================================================================ C-7 synthetic check
C7_SUCCESS = ("T00001", "T00397", "T00669")                # G2 ok (T00397 = a G2 remapped reaction)
C7_STAGE_FAIL = (("ts", "T00009"), ("stereo", "T10341"), ("ref", "T09757"))     # copies of real G2 failed folders
C7_IMAG = -321.0


def _fake_hess(path, syms, xyz, pairs, imag):
    """an ORCA-format .hess (sections parse_hess reads): 6 zero modes, `imag` (or none) and positive modes; the imaginary
    mode stretches the formed pairs + deterministic noise."""
    n = len(syms)
    m = 3 * n
    nv = m - 6 - (1 if imag is not None else 0)
    freqs = [0.0] * 6 + ([float(imag)] if imag is not None else []) + [100.0 + 7.0 * k for k in range(nv)]
    modes = np.eye(m)
    if imag is not None:
        rng = np.random.default_rng(C.SEED)
        v = 0.01 * rng.standard_normal((n, 3))
        for i, j in pairs:
            u = xyz[j] - xyz[i]
            u = u / np.linalg.norm(u)
            v[i] += 0.3 * u
            v[j] -= 0.3 * u
        modes[:, 6] = (v / np.linalg.norm(v)).ravel()
    L = ["", "$orca_hessian_file", "", "$vibrational_frequencies", str(m)]
    L += [f"{k:5d} {f:16.6f}" for k, f in enumerate(freqs)]
    L += ["", "$normal_modes", f"{m} {m}"]
    for c0 in range(0, m, 5):
        cols = list(range(c0, min(c0 + 5, m)))
        L.append(" " * 6 + "".join(f"{c:14d}" for c in cols))
        L += [f"{r:6d}" + "".join(f"{modes[r, c]:14.8f}" for c in cols) for r in range(m)]
    L += ["", "$atoms", str(n)] + [f" {s:<2} {1.0:10.5f} {x / 0.52917721:16.10f} {y / 0.52917721:16.10f} "
                                     f"{z / 0.52917721:16.10f}" for s, (x, y, z) in zip(syms, xyz)]
    L += ["", "$end", ""]
    Path(path).write_text("\n".join(L))


class _Stub:
    """OD.run replacement for C-7: mode per rxn folder ('ok', 'noimag', 'noterm', 'crash'); records calls and whether
    the final folder existed while the 'calculation' ran (atomic publication)."""

    def __init__(self, out_root, pairs):
        self.out_root, self.pairs, self.mode, self.calls, self.final_seen = Path(out_root), pairs, {}, {}, {}

    def __call__(self, cfg_, inp):
        inp = Path(inp)
        rid = inp.parent.parent.name[:-len(".tmp")]
        self.calls[rid] = self.calls.get(rid, 0) + 1
        self.final_seen[rid] = (self.out_root / rid).exists() and marker(self.out_root / rid) is not None
        mode = self.mode.get(rid, "ok")
        lines = inp.read_text().splitlines()
        k = next(i for i, ln in enumerate(lines) if ln.startswith("* xyz"))
        body = [ln.split() for ln in lines[k + 1:] if ln.strip() and not ln.startswith("*")]
        syms, xyz = [b[0] for b in body], np.array([[float(v) for v in b[1:4]] for b in body])
        out = inp.with_suffix(".out")
        (inp.parent / "ts.engrad").write_text("scratch\n")                       # ORCA scratch -> must be cleaned
        (inp.parent / "ts_tmpdir").mkdir(exist_ok=True)
        (inp.parent / "ts_tmpdir" / "x.tmp").write_text("scratch\n")
        txt = "C-7 stub ORCA output\nFINAL SINGLE POINT ENERGY       -12.345678901\n"
        if mode == "crash":
            out.write_text(txt)
            raise KeyboardInterrupt("C-7 simulated kill during ORCA")
        if mode == "noterm":
            out.write_text(txt + "ORCA finished by error termination in XTB\n")
            return out
        _fake_hess(inp.parent / "ts.hess", syms, xyz, self.pairs[rid], None if mode == "noimag" else C7_IMAG)
        out.write_text(txt + "                             ****ORCA TERMINATED NORMALLY****\n")
        return out


def selftest_c7(out_json=None, cores=4):
    """C-7 (spec 9): fake work folders (3 successes from G2 ok folders, 3 failures = copies of real G2 ts / stereo / ref
    failures, + freq_not_terminated / no-imaginary / crash / stale variants) through finalize_lf with ORCA stubbed.
    Checks markers (stage failures = the marker g2_autode.finalize wrote for the same folder in $G2/g2_final),
    contents, idempotent rerun, atomic publication and the write guard. Returns the report dict (all_pass)."""
    base = C.DEP / "smoke" / "c7_selftest"
    C.assert_write_ok(base)
    if base.exists():
        shutil.rmtree(base)
    src_root, out_root = base / "src", base / "out"
    man = load_manifest()
    st = {r["job_id"]: r for r in load_targets()}
    cases = []

    def copy_src(T, name):
        d = src_root / name
        d.mkdir(parents=True)
        for p in (C.G2_JOBS / T).iterdir():
            if p.is_file() and p.name != "inputs.tar.gz":
                shutil.copy(p, d / p.name)
        return d

    for T in C7_SUCCESS:
        cases.append(dict(name=f"ok_{T}", job=T, row=dict(man[T]), src=copy_src(T, T), expect=".done", mode="ok"))
    for stg, T in C7_STAGE_FAIL:
        g2m = marker(C.G2_FINAL / st[T]["rxn_id"])
        cases.append(dict(name=f"{stg}_fail_{T}", job=T, row=dict(man[T]), src=copy_src(T, T), expect=g2m, mode="ok",
                          g2_status_marker=st[T]["marker"]))
    extra = (("noterm", "T00001", ".fail_freq_not_terminated"), ("noimag", "T00001", ".fail_hess_no_imaginary_frequency"),
             ("crash", "T00669", ".done"), ("stale", "T00397", ".done"))
    for k, (mode, T, exp) in enumerate(extra, 1):
        row = dict(man[T], job_id=f"C7_{mode}", meta_rxn_id=9000000 + k)
        cases.append(dict(name=f"{mode}_{T}", job=row["job_id"], row=row, src=copy_src(T, row["job_id"]), expect=exp,
                          mode=mode))
    pairs = {}
    for c in cases:
        r = _read_json(c["src"] / "ts_result.json") or {}
        c["rid"] = str(rxn_key(c["row"]))
        pairs[c["rid"]] = [tuple(p) for p in r.get("formed_pairs_ts") or []]
    stub = _Stub(out_root, pairs)
    for c in cases:
        stub.mode[c["rid"]] = c["mode"]
    checks = {}

    def chk(name, ok, detail=None):
        checks[name] = dict(ok=bool(ok), detail=detail)

    orig_run = OD.run
    OD.run = stub
    try:
        out_root.mkdir(parents=True)
        stale = next(c for c in cases if c["mode"] == "stale")
        (out_root / stale["rid"]).mkdir()
        (out_root / stale["rid"] / "junk_unmarked.txt").write_text("partial folder without marker\n")
        (out_root / f"{stale['rid']}.tmp").mkdir()
        (out_root / f"{stale['rid']}.tmp" / "junk.txt").write_text("stale tmp\n")
        first = {}
        for c in cases:
            try:
                first[c["name"]] = finalize_lf(c["row"], c["src"], out_root, cores)
            except KeyboardInterrupt as e:
                first[c["name"]] = f"interrupted:{e}"
        crash = next(c for c in cases if c["mode"] == "crash")
        chk("crash_leaves_no_final_folder", first[crash["name"]].startswith("interrupted")
            and not (out_root / crash["rid"]).exists() and (out_root / f"{crash['rid']}.tmp").is_dir(),
            dict(status=first[crash["name"]]))
        stub.mode[crash["rid"]] = "ok"
        first[crash["name"] + "_rerun"] = finalize_lf(crash["row"], crash["src"], out_root, cores)
        calls_after_first = dict(stub.calls)
        # ---- markers + contents
        for c in cases:
            d = out_root / c["rid"]
            mk = sorted(p.name for p in d.glob(".done")) + sorted(p.name for p in d.glob(".fail_*"))
            chk(f"marker:{c['name']}", mk == [c["expect"]], dict(found=mk, expect=c["expect"]))
            res = _read_json(d / "result.json") or {}
            if c["expect"] == ".done":
                src_ts, tsr = pc.read_xyz(c["src"] / "ts.xyz"), _read_json(c["src"] / "ts_result.json")
                new_ts = pc.read_xyz(d / "ts.xyz")
                same = lambda a, b: list(a[0]) == list(b[0]) and np.array_equal(a[1], b[1])   # noqa: E731
                imag, mode = OD.imag_and_mode(d / "work" / "ts.hess")
                heavy = [i for i, s in enumerate(new_ts[0]) if s != "H"]
                fp = sorted(tuple(sorted(p)) for p in tsr["formed_pairs_ts"])
                share = RT.forming_share(np.asarray(new_ts[1], dtype=float), mode, fp, heavy)
                inp = (d / "work" / "ts.inp").read_text()
                hdr = GA.FREQ_HEADER.format(n=cores, m=MAXCORE_MB)
                det = dict(
                    files=[f for f in DONE_FILES if not (d / f).is_file()],
                    work_extra=sorted(p.name for p in (d / "work").iterdir() if p.name not in KEEP_WORK_OK),
                    keys=[k for k in RESULT_KEYS if k not in res],
                    hess_same=_same_file(d / "ts.hess", d / "work" / "ts.hess"),
                    ts_same=same(new_ts, src_ts),
                    rel1_same=same(pc.read_xyz(d / "rel1.xyz"), pc.read_xyz(c["src"] / "ref_dipole.xyz")),
                    rel2_same=same(pc.read_xyz(d / "rel2.xyz"), pc.read_xyz(c["src"] / "ref_dipolarophile.xyz")),
                    remapped_false=res.get("remapped") is False,
                    fp_own=res.get("formed_pairs_ts") == [list(p) for p in fp],
                    A_own=res.get("A_idx") == sorted(tsr["A_idx"]),
                    imag=res.get("imag_freqs_cm") == [C7_IMAG] and imag == [C7_IMAG],
                    share=res.get("imag_mode_forming_share") == share,
                    inp_header=inp.startswith(hdr + f"* xyz {int(c['row']['charge'])} 1\n"),
                    sha=all(C.sha256(c["src"] / f) == s for f, s in res.get("inputs_sha256", {}).items())
                    and "ts.xyz" in res.get("inputs_sha256", {}),
                    status=res.get("status") == "ok" and res.get("fail_reasons") == [])
                ok = not det["files"] and not det["work_extra"] and not det["keys"] and all(
                    v for k, v in det.items() if k not in ("files", "work_extra", "keys"))
                chk(f"contents:{c['name']}", ok, det)
            else:
                reason = (res.get("fail_reasons") or [""])[0]
                det = dict(status=res.get("status"), reason=reason,
                           marker_from_reason=marker_name(reason) == c["expect"],
                           no_top_xyz=not any((d / f).exists() for f in ("ts.xyz", "rel1.xyz", "rel2.xyz", "ts.hess")),
                           orca_called=stub.calls.get(c["rid"], 0))
                ok = det["status"] == "fail:" + reason and det["marker_from_reason"] and det["no_top_xyz"]
                if c["mode"] == "ok":                                   # stage failures: ORCA never called
                    ok = ok and det["orca_called"] == 0 and c["expect"] == c.get("g2_status_marker")
                    det["g2_status_csv_marker"] = c.get("g2_status_marker")
                chk(f"contents:{c['name']}", ok, det)
        chk("stale_partial_replaced", not (out_root / stale["rid"] / "junk_unmarked.txt").exists()
            and marker(out_root / stale["rid"]) == ".done")
        chk("atomic_final_absent_during_orca", not any(stub.final_seen.values()), stub.final_seen)
        chk("no_tmp_left", not list(out_root.glob("*.tmp")), [p.name for p in out_root.glob("*.tmp")])
        # ---- idempotent rerun
        before = {c["rid"]: (C.sha256(out_root / c["rid"] / "result.json"), os.stat(out_root / c["rid"]).st_ino)
                  for c in cases}
        again = {c["name"]: finalize_lf(c["row"], c["src"], out_root, cores) for c in cases}
        after = {c["rid"]: (C.sha256(out_root / c["rid"] / "result.json"), os.stat(out_root / c["rid"]).st_ino)
                 for c in cases}
        chk("idempotent_rerun", all(s.startswith("skip") for s in again.values()) and before == after
            and stub.calls == calls_after_first, dict(statuses=again))
        # ---- write guard: nothing may be written outside $DEP / analysis/deploy/results
        bad_root = C.G2 / "never_written_by_deploy_c7"
        print("[C-7] write-guard test: the next 'STOP: write outside ...' line is expected", file=sys.stderr, flush=True)
        try:
            finalize_lf(cases[0]["row"], cases[0]["src"], bad_root, cores)
            guard = False
        except C.Stop as e:
            guard = e.code == 3
        chk("write_guard_stop", guard and not bad_root.exists(), str(bad_root))
    finally:
        OD.run = orig_run
    rep = dict(spec="docs/specs/DEPLOY_MODEL.md 9 C-7", base=str(base), cores=cores, orca="stub (OD.run replaced)",
               cases=[dict(name=c["name"], job=c["job"], rid=c["rid"], mode=c["mode"], expect=c["expect"],
                           first=first.get(c["name"])) for c in cases],
               crash_rerun=first.get(crash["name"] + "_rerun"), stub_calls=stub.calls, checks=checks,
               n_checks=len(checks), n_fail=sum(not v["ok"] for v in checks.values()),
               all_pass=all(v["ok"] for v in checks.values()), written=time.strftime("%F %T"))
    C.write_json(Path(out_json) if out_json else C.RES / "C7_finalize_selftest.json", rep)
    return rep


def cmd_selftest():
    rep = selftest_c7()
    print(json.dumps({k: rep[k] for k in ("n_checks", "n_fail", "all_pass")}), flush=True)
    for k, v in rep["checks"].items():
        print(f"  {'ok  ' if v['ok'] else 'FAIL'} {k}" + ("" if v["ok"] else f"  {v['detail']}"), flush=True)
    if not rep["all_pass"]:
        C.stop("C-7 finalize_lf selftest failed")
    return 0


if __name__ == "__main__":
    a, jobs = parse_jobs(sys.argv[1:])
    if a[:1] == ["run"] and len(a) == 4:
        sys.exit(cmd_run(a[1], a[2], a[3], jobs))
    if a[:1] == ["report"] and len(a) == 1:
        sys.exit(cmd_report(jobs))
    if a[:1] == ["selftest"] and len(a) == 1:
        sys.exit(cmd_selftest())
    raise SystemExit(__doc__)
