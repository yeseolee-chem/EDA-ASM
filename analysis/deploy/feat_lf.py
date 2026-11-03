#!/usr/bin/env python3
"""feat_lf.py — DEPLOY-MODEL Phase F (docs/specs/DEPLOY_MODEL.md 5): the 182 global features of every label-free g2
reaction ($DEP/g2lf_final/<rxn_id>/.done, autodE atom order) -> $DEP/X182_lf.parquet + results/F_report.json.

  python feat_lf.py slice <i|@I> <n>   # reactions [i::n] of the .done set -> $DEP/F/slices/slice<i>of<n>_{g1,ext}.parquet
  python feat_lf.py merge              # all slices of one n -> $DEP/X182_lf.parquet, results/F_report.json (F-1..F-4)

Job: analysis/d1_pair/jobs/g.slurm submitted from analysis/d1_pair with --export=ALL,PYENV=venv5,D1PAIR_PHASE=F (the
D1PAIR / D1-G2 Phase F job environment: rev 5 venv = xtb 6.7.1 + tblite 0.7.0, site_si Si entries); parent_env()
checks it with d1_g1.check_env() and d1_features.check_engines().

A thin runner (spec 5-1) around the unchanged D1PAIR / D1-G2 code; it mirrors d1_features.py runroot -> _rootrun ->
run_ids and mergeroot: the slice command starts a child interpreter with the 5-2 environment (the rev 4-6 modules read
their roots at import), the child checks that the modules picked up that environment (child_env) and calls
d1_features.run_ids (= r6_features._proc_one = xtb_slice.process_one(label, "g1") in a spawn pool, aggregate.derive,
reindex to the rev 4 G1 columns, r6_features._ext = ext_features B1..B6, CODE_VERSION r5-ext-3); merge calls
d1_features.merge_rows (concat, aggregate.derive, rev 4 reindex, rev6_common.our_columns 182) with the mergeroot
checks. Only three things differ from d1_features runroot / _rootrun / mergeroot (spec 5-1):
  (1) partition mode: xtb_slice.PARTITION_MODE must be "redetect" (ESPLEY_PARTITION=redetect), not "label";
  (2) the g_env() comparison of the child allows atoms.ROOT_ENV (6 roots) + ESPLEY_PARTITION to differ;
  (3) writes: dep_common.assert_write_ok ($DEP, analysis/deploy/results) instead of d1_features.ROOT_WRITE_OK, on top
      of the unchanged rev 4-6 / D1 guard d1pair_common.assert_write_ok.
Every other check is copied: the modules read the same roots, the Si entries, the engine versions per slice, the
slice records (root / labels / meta sha256 / selection), the .done set unchanged at merge.
5-2 environment of the child (lf_env): G1_ROOT=$DEP/g2lf_final, ESPLEY_LABELS=$DEP/labels_lf.json,
ESPLEY_META=$DEP/input_meta_lf.csv, ESPLEY_PROF=$DEP/prof_empty (must be an EMPTY folder: no DFT structure can be
read), ESPLEY_PARTITION=redetect, ESPLEY_OUT=$DEP/F/xtb, R5_SCRATCH=$DEP/F/r5 (ext cache $DEP/F/r5/ext),
D1PAIR_PHASE=FROOT; all other g_env() values (ESPLEY_SMILES, G1_ROOT_REV4, G1_ROOT_R6, G2_ROOT, R6_SCRATCH, R4_SCRATCH,
D1_SNAPSHOT) and D1PAIR_SI_PATCH=1 unchanged from the job environment.
Gates (merge): F-1 columns / order = $G2/X182_d1_g2.parquet (STOP); F-2 redetection failures = xtb_status partition:*
or partition_mismatch_g1 (count + rate, judged by parity.py G-A); F-3 other feature failures <= 2 % of the .done
reactions (STOP); F-4 wall / core time per reaction (slice timing records + ext t_total_s).
Exit codes: 0 done, 2 not ready (resubmit later), 3 STOP. Idempotent: run_ids skips existing slice parquets, the ext
cache is per reaction, every write is atomic; resubmit the same command after a 48 h wall.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                  # no __pycache__ in the read-only analysis/d1_pair, espley_xtb_repro trees

import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
REPO = HERE.parents[1]
P_DIR = REPO / "analysis" / "d1_pair"
FU2_DIR = P_DIR / "fu2"
for _p in (str(FU2_DIR), str(P_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import dep_common as D  # noqa: E402
import d1_features as DF  # noqa: E402     run_ids, merge_rows, root_ids, check_engines (D1PAIR / D1-G2, unchanged)
import d1_g1 as DG  # noqa: E402           g_env, check_env
import atoms as A  # noqa: E402            ROOT_ENV (the 6 FROOT roots), FU2_ENV

C = DF.C                                       # d1pair_common (rev 4-6 / D1 write guard, SI_DECISION)

# ---------------------------------------------------------------- spec 5 constants
PARTITION = "redetect"                         # spec 5-1 (1)
CHILD_PHASE = "FROOT"                          # = d1_features runroot child / atoms.py root
ENV_EXCEPT = tuple(A.ROOT_ENV) + ("ESPLEY_PARTITION",)              # spec 5-1 (2)
LF_ENV_KEYS = tuple(A.ROOT_ENV) + ("ESPLEY_PARTITION", "D1PAIR_PHASE")
SI_WANT = ("1", 1.11, 14, 2.10, 4)            # (D1PAIR_SI_PATCH, fr.COV, fr.Z, R5.BONDI, R5.GFN2_VALENCE)["Si"]
F_DIR = D.DEP / "F"
X182_G2 = D.G2 / "X182_d1_g2.parquet"          # F-1 column reference (LADDER g2 input, read only)
F_REPORT = D.RES / "F_report.json"
F3_MAX = 0.02                                  # spec 5-4 F-3
REDETECT_FAIL = "xtb_status 'partition:*' or 'partition_mismatch_g1' (spec 5-4 F-2)"


def fdirs(fdir):
    """the layout of one feature run under fdir: slices/ (run_ids outputs + records), xtb/ (ESPLEY_OUT), r5/ (R5_SCRATCH)."""
    fdir = Path(os.path.abspath(fdir))
    return dict(slices=fdir / "slices", xtb=fdir / "xtb", r5=fdir / "r5")


def is_redetect_fail(st) -> bool:
    st = str(st)
    return st.startswith("partition:") or st == "partition_mismatch_g1"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def lf_env(root, labels, meta, prof, fdir, base=None):
    """spec 5-2: the child environment (job environment + the lf roots)."""
    d = fdirs(fdir)
    env = dict(os.environ if base is None else base)
    env.update(G1_ROOT=os.path.abspath(root), ESPLEY_LABELS=os.path.abspath(labels), ESPLEY_META=os.path.abspath(meta),
               ESPLEY_PROF=os.path.abspath(prof), ESPLEY_PARTITION=PARTITION, ESPLEY_OUT=str(d["xtb"]),
               R5_SCRATCH=str(d["r5"]), D1PAIR_PHASE=CHILD_PHASE, PYTHONDONTWRITEBYTECODE="1")
    return env


def parent_env():
    """the job environment = the D1PAIR Phase F one (d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F): exit 3
    otherwise (d1_g1.check_env: g_env() exactly + Si entries; d1_features.check_engines: venv5, tblite 0.7.0, xtb 6.7.1)."""
    if os.environ.get("D1PAIR_PHASE") != "F":
        D.stop(f"D1PAIR_PHASE={os.environ.get('D1PAIR_PHASE')!r}: submit from analysis/d1_pair with "
               "--export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm ../deploy/<script> ...")
    over = [k for k in A.FU2_ENV if os.environ.get(k)]
    if over:
        D.stop(f"{over} set in the job environment (the children set their own)")
    DG.check_env()
    return DF.check_engines()


def prof_is_empty(prof) -> bool:
    p = Path(prof)
    return p.is_dir() and not any(p.iterdir())


def child_env(rec_env):
    """child of a slice run: exit 3 unless the rev 4-6 modules picked up the 5-2 environment. The checks of
    d1_features._rootrun (+ the module write paths of atoms._env_root) with the spec 5-1 changes (1)-(3)."""
    import aggregate
    import ext_features as ef
    import fragmenter as fr
    import r6_partition as RP
    import rev5_common as R5
    import rev6_common as R6
    import xtb_slice as xs
    bad = {k: (os.environ.get(k), v) for k, v in rec_env.items() if os.environ.get(k) != v}
    if bad or rec_env.get("D1PAIR_PHASE") != CHILD_PHASE or rec_env.get("ESPLEY_PARTITION") != PARTITION:
        D.stop(f"child environment differs from the slice record: {bad}")
    gbad = {k: (os.environ.get(k), v) for k, v in DG.g_env().items() if k not in ENV_EXCEPT and os.environ.get(k) != v}
    if gbad:
        D.stop(f"child environment differs from d1_g1.g_env() outside {ENV_EXCEPT}: {gbad}")
    e = rec_env
    paths = dict(R5_G1_ROOT=(R5.G1_ROOT, "G1_ROOT"), xtb_slice_g1=(xs.GEOM_ROOT["g1"], "G1_ROOT"),
                 R5_SCRATCH=(R5.SCRATCH, "R5_SCRATCH"), R5_XTB_ROOT=(R5.XTB_ROOT, "ESPLEY_OUT"),
                 aggregate_ROOT=(aggregate.ROOT, "ESPLEY_OUT"), R5_LABELS=(R5.LABELS, "ESPLEY_LABELS"),
                 xtb_slice_LABELS=(xs.LABELS_PATH, "ESPLEY_LABELS"), R5_META=(R5.META, "ESPLEY_META"),
                 xtb_slice_META=(xs.META, "ESPLEY_META"), R5_PROF=(R5.PROF, "ESPLEY_PROF"),
                 r6_partition_PROF=(RP.PROF, "ESPLEY_PROF"))
    bad = {k: (str(p), e[v]) for k, (p, v) in paths.items() if Path(p) != Path(e[v])}
    si = (os.environ.get("D1PAIR_SI_PATCH"), fr.COV.get("Si"), fr.Z.get("Si"), R5.BONDI.get("Si"), R5.GFN2_VALENCE.get("Si"))
    if bad or xs.PARTITION_MODE != PARTITION or si != SI_WANT:
        D.stop(f"modules not on the lf environment: {bad}, partition {xs.PARTITION_MODE}, Si {si}")
    if not prof_is_empty(R5.PROF):
        D.stop(f"ESPLEY_PROF {R5.PROF} is not an empty folder (spec 1: prof_empty stays empty)")
    if ef.B1_TOL_OVERRIDE:
        D.stop(f"R6_B1_TOL is set ({ef.B1_TOL_OVERRIDE}): the D1-G2 runs used the rev 5 B1 gate for every reaction")
    for p in (xs.GEOM_ROOT["g1"], xs.GEOM_ROOT["g2"], R5.SCRATCH, R5.XTB_ROOT, R5.G1_ROOT, R6.SCRATCH, R6.G1_ROOT_R6,
              R6.XTB_R6):
        C.assert_write_ok(p)                              # rev 4-6 / D1 locations (atoms._env_root list)
    for p in (R5.XTB_ROOT, R5.SCRATCH):
        D.assert_write_ok(p)                              # spec 5-1 (3)
    return dict(si=si, partition=xs.PARTITION_MODE, roots={k: str(p) for k, (p, _) in paths.items()})


# ================================================================ slice
def run_slice(root, labels, meta, prof, fdir, i, n, ids=None):
    """reactions [i::n] of the .done set of root (ids = an explicit selection, e.g. C-1) through d1_features.run_ids in
    a child with the 5-2 environment -> fdir/slices/slice<i>of<n>_{g1,ext,versions}.* + _lf.json (record) + _timing.json.
    Returns 0, or raises dep_common.Stop (2 not ready, 3 STOP)."""
    root, labels, meta, prof = (Path(os.path.abspath(p)) for p in (root, labels, meta, prof))
    d = fdirs(fdir)
    for p in d.values():
        C.assert_write_ok(p)
        D.assert_write_ok(p)
    i, n = int(i), int(n)
    if not 0 <= i < n:
        D.stop(f"slice {i} of {n}?")
    if not root.is_dir():
        D.not_ready(f"root {root} is not a folder (Phase L)")
    if not (labels.is_file() and meta.is_file()):
        D.not_ready(f"labels {labels} / meta {meta} missing (Phase M)")
    if not prof_is_empty(prof):
        D.stop(f"{prof} is not an empty folder (spec 1)")
    done, _ = DF.root_ids(root)
    lab_ids = {int(r["rxn_id"]) for r in json.loads(labels.read_text())}
    meta_ids = set(int(r) for r in pd.read_csv(meta).rxn_id)
    if ids is None:
        if set(done) != lab_ids or not set(done) <= meta_ids:
            D.stop(f"labels {labels} ({len(lab_ids)} records) / meta ({len(meta_ids)} rows) do not cover exactly the "
                   f"{len(done)} .done reactions of {root} (spec 4: records = .done): .done without record "
                   f"{sorted(set(done) - lab_ids)[:10]}, record without .done {sorted(lab_ids - set(done))[:10]}, "
                   f"without meta {sorted(set(done) - meta_ids)[:10]}")
        sel, selection = done, "all .done"
    else:
        sel, selection = sorted(int(x) for x in ids), "ids"
        miss = [r for r in sel if r not in set(done) or r not in lab_ids or r not in meta_ids]
        if miss:
            D.stop(f"selected reactions without .done / label record / meta row: {miss[:10]}")
    mine = sel[i::n]
    tag = f"slice{i:02d}of{n:02d}"
    if not mine:
        D.not_ready(f"slice {tag} has no reaction ({len(sel)} selected)")
    env = lf_env(root, labels, meta, prof, fdir)
    rec = dict(root=str(root), i=i, n=n, env={k: env[k] for k in LF_ENV_KEYS}, labels=str(labels),
               labels_sha256=D.sha256(labels), meta=str(meta), meta_sha256=D.sha256(meta), prof=str(prof),
               selection=selection, ids_sha256=D.sha256_text(",".join(map(str, sel))), n_selected=len(sel),
               n_ids=len(mine), ids=mine)
    sdir = d["slices"]
    f = sdir / f"{tag}_lf.json"
    if f.is_file() and json.loads(f.read_text()) != json.loads(json.dumps(rec)):
        D.stop(f"{f} was made for another root / selection / environment / input file (use a fresh {fdir})")
    if not f.is_file():
        D.write_json(f, rec)
    g1p, exp = sdir / f"{tag}_g1.parquet", sdir / f"{tag}_ext.parquet"
    pre = (g1p.is_file(), exp.is_file())
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[F lf slice {i}/{n}] {len(mine)} of {len(sel)} reactions ({selection}), root {root}, {workers} workers, "
          f"outputs present {pre}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(THIS), "_slice", tag, str(f)], env=env)
    t1 = time.time()
    if r.returncode != 0:
        D.stop(f"slice child exit {r.returncode}", 3 if r.returncode != 2 else 2)
    if not all(pre):                                       # F-4 timing of what this call computed
        tg1 = g1p.stat().st_mtime if not pre[0] else None
        n_ok = int((pd.read_parquet(g1p, columns=["xtb_status"]).xtb_status == "ok").sum())
        g1_wall = (tg1 - t0) if tg1 else None
        ext_wall = (t1 - (tg1 or t0)) if not pre[1] else None
        D.write_json(sdir / f"{tag}_timing.json",
                     dict(tag=tag, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), workers=workers,
                          started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), wall_s=t1 - t0,
                          n_rxn=len(mine), n_xtb_ok=n_ok, g1feat_wall_s=g1_wall, ext_wall_s=ext_wall,
                          g1feat_core_s_per_rxn=g1_wall * workers / len(mine) if g1_wall else None,
                          ext_core_s_per_rxn=ext_wall * workers / n_ok if (ext_wall and n_ok) else None,
                          computed=dict(g1=not pre[0], ext=not pre[1]),
                          note="stage walls from the slice parquet mtimes (run_ids writes g1 then ext); core-s = wall x "
                               "workers / reactions (pool tail included)"))
    return 0


def _child_slice(tag, rec_path):
    rec = json.loads(Path(rec_path).read_text())
    child_env(rec["env"])
    import rev5_common as R5
    sdir = Path(rec_path).parent
    D.assert_write_ok(sdir)
    DF.run_ids([int(x) for x in rec["ids"]], tag, sdir, R5.SCRATCH / "ext", int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    return 0


def cmd_slice(i, n):
    parent_env()
    return run_slice(D.LF_FINAL, D.LABELS_LF, D.META_LF, D.PROF_EMPTY, F_DIR, D.array_task(i), int(n))


# ================================================================ merge
def merge_slices(fdir):
    """the d1_features cmd_mergeroot checks on fdir/slices + d1_features.merge_rows -> dict(g, e, m, cols, recs, ids,
    versions, timing). Raises dep_common.Stop (2 = slices incomplete, 3 = inconsistent)."""
    sd = fdirs(fdir)["slices"]
    g1p, exp = sorted(sd.glob("slice*_g1.parquet")), sorted(sd.glob("slice*_ext.parquet"))
    ns = {p.name.split("of")[1][:2] for p in g1p}
    if len(ns) != 1 or len(g1p) != int(next(iter(ns))) or len(exp) != len(g1p):
        D.not_ready(f"incomplete slices in {sd}: {len(g1p)} g1 / {len(exp)} ext of n {sorted(ns)}")
    vers = {p.name: json.loads(p.read_text()) for p in sorted(sd.glob("slice*_versions.json"))}
    v0 = DF.check_engines()
    if len(vers) != len(g1p) or any(json.loads(json.dumps(v)) != json.loads(json.dumps(v0)) for v in vers.values()):
        D.stop(f"slice engine versions differ or are missing: {vers}")
    recs = [json.loads(p.read_text()) for p in sorted(sd.glob("slice*_lf.json"))]
    keys = ("root", "n", "env", "labels", "labels_sha256", "meta", "meta_sha256", "prof", "selection", "ids_sha256")
    same = {json.dumps({k: r.get(k) for k in keys}, sort_keys=True) for r in recs}
    if len(recs) != len(g1p) or len(same) != 1 or sorted(r["i"] for r in recs) != list(range(len(g1p))):
        D.stop(f"slice records missing or from different roots / selections / inputs ({len(recs)} records)")
    r0 = recs[0]
    if D.sha256(r0["labels"]) != r0["labels_sha256"] or D.sha256(r0["meta"]) != r0["meta_sha256"]:
        D.stop(f"{r0['labels']} / {r0['meta']} changed since the slices were made")
    ids = sorted(int(x) for r in recs for x in r["ids"])
    g, e, m, cols = DF.merge_rows(g1p, exp)
    if sorted(int(r) for r in g.rxn_id) != ids:
        D.stop(f"g1feat rows {len(g)} != recorded slice reactions {len(ids)}")
    if r0["selection"] == "all .done":
        done, _ = DF.root_ids(r0["root"])
        if done != ids:
            D.stop(f".done reactions of {r0['root']} ({len(done)}) changed since the slices ({len(ids)})")
    timing = [json.loads(p.read_text()) for p in sorted(sd.glob("slice*_timing.json"))]
    return dict(g=g, e=e, m=m, cols=cols, recs=recs, ids=ids, versions=vers, timing=timing, g1p=g1p, exp=exp)


def reasons(m, ids):
    """rxn_id -> 'ok' | 'g1feat:<xtb_status>' | 'ext:<ext_status>' | 'nonfinite_182' (= d1_features cmd_mergeroot)."""
    mm = m.set_index("rxn_id")
    why = {}
    for rid in ids:
        if mm.loc[rid, "xtb_status"] != "ok":
            why[rid] = f"g1feat:{mm.loc[rid, 'xtb_status']}"
        elif mm.loc[rid, "ext_status"] != "ok":
            why[rid] = f"ext:{mm.loc[rid, 'ext_status']}"
        elif mm.loc[rid, "n_nonfinite"]:
            why[rid] = "nonfinite_182"
        else:
            why[rid] = "ok"
    return why


def x182_columns_ref():
    import pyarrow.parquet as pq
    return list(pq.read_schema(X182_G2).names)


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)),
                max=float(v.max()), sum=float(v.sum()))


def timing_summary(timing, e):
    """F-4: per-reaction cost. g1feat (xtb_slice.process_one) from the slice records (wall x workers / reactions);
    ext (B1..B6) per reaction from ext t_total_s (single-thread worker seconds = core-s) and the slice records."""
    g_core = sum((t["g1feat_wall_s"] or 0) * t["workers"] for t in timing if t.get("g1feat_wall_s"))
    g_n = sum(t["n_rxn"] for t in timing if t.get("g1feat_wall_s"))
    x_core = sum((t["ext_wall_s"] or 0) * t["workers"] for t in timing if t.get("ext_wall_s"))
    x_n = sum(t["n_xtb_ok"] for t in timing if t.get("ext_wall_s"))
    tt = e["t_total_s"].to_numpy(float) if "t_total_s" in e else np.zeros(0)
    blocks = {b: _stats(e[f"t_{b}"].to_numpy(float)) for b in ("B1", "B2", "B3", "B4", "B5", "B6") if f"t_{b}" in e}
    out = dict(n_slice_records=len(timing),
               slices=[{k: t.get(k) for k in ("tag", "host", "workers", "n_rxn", "n_xtb_ok", "wall_s", "g1feat_wall_s",
                                              "ext_wall_s", "g1feat_core_s_per_rxn", "ext_core_s_per_rxn")} for t in timing],
               g1feat_core_s_per_rxn=g_core / g_n if g_n else None, ext_pool_core_s_per_rxn=x_core / x_n if x_n else None,
               ext_t_total_s_per_rxn=_stats(tt), ext_block_s=blocks,
               note="g1feat = 5 xtb single points + D3 + SASA per reaction (pool wall x workers / n, tail included); ext "
                    "t_total_s = sum of the B1..B6 block times of one single-threaded worker (core-s)")
    if out["g1feat_core_s_per_rxn"] is not None and out["ext_pool_core_s_per_rxn"] is not None:
        out["total_core_s_per_rxn"] = out["g1feat_core_s_per_rxn"] + out["ext_pool_core_s_per_rxn"]
        out["total_core_h_per_1000_rxn"] = out["total_core_s_per_rxn"] * 1000 / 3600
    return out


def cmd_merge():
    v0 = parent_env()
    M = merge_slices(F_DIR)
    g, e, m, cols, ids = M["g"], M["e"], M["m"], M["cols"], M["ids"]
    x = m.loc[m.F_ok, ["rxn_id"] + cols].sort_values("rxn_id").reset_index(drop=True)
    ref = x182_columns_ref()
    f1 = list(x.columns) == ref and len(ref) == 183
    why = reasons(m, ids)
    st = dict(zip(m.rxn_id.astype(int), m.xtb_status.astype(str)))
    redet = {r: st[r] for r in ids if is_redetect_fail(st[r])}
    other = {r: w for r, w in why.items() if w != "ok" and r not in redet}
    n = len(ids)
    finite = bool(np.isfinite(x[cols].to_numpy(float)).all())
    stop = []
    if not f1:
        stop.append(f"F-1: X182_lf columns != {X182_G2} columns")
    if not finite:
        stop.append("non-finite values among the 182 columns of F_ok rows")
    if n and len(other) / n > F3_MAX:
        stop.append(f"F-3: {len(other)} / {n} = {len(other) / n:.4f} feature failures (other than redetection) > {F3_MAX}")
    if not n:
        stop.append("no reaction")
    xd = fdirs(F_DIR)["xtb"]
    gp, ep = xd / "xtb_features_g1_lf.parquet", xd / "xtb_features_ext_g1_lf.parquet"
    D.write_parquet(gp, g)
    D.write_parquet(ep, e)
    D.write_parquet(D.X182_LF, x)
    r0 = M["recs"][0]
    import ext_features as ef
    rep = dict(spec="docs/specs/DEPLOY_MODEL.md 5 (STOP: 16 F)", written=now(), root=r0["root"], labels=r0["labels"],
               labels_sha256=r0["labels_sha256"], meta=r0["meta"], meta_sha256=r0["meta_sha256"], prof=r0["prof"],
               env=r0["env"], n_slices=len(M["g1p"]), n_done=n, n_feat=len(x), F_fail_rate=(n - len(x)) / n if n else None,
               x182=dict(path=str(D.X182_LF), sha256=D.sha256(D.X182_LF), n=len(x), columns_equal_X182_d1_g2=f1,
                         X182_d1_g2=str(X182_G2)),
               sha256=dict(g1=D.sha256(gp), ext=D.sha256(ep)), outputs=dict(g1=str(gp), ext=str(ep)),
               reasons=dict(Counter(why.values()).most_common()),
               not_ok={str(r): w for r, w in sorted(why.items()) if w != "ok"},
               xtb_status=m.xtb_status.value_counts().to_dict(), ext_status=m.ext_status.value_counts(dropna=False).to_dict(),
               F1_all_finite=finite,
               F2=dict(rule=REDETECT_FAIL, n=len(redet), rate=len(redet) / n if n else None,
                       statuses=dict(Counter(redet.values()).most_common()), rxn=sorted(redet),
                       note="judged in parity.py (G-A: <= 2 % of the Phase L .done reactions)"),
               F3=dict(n=len(other), rate=len(other) / n if n else None, max=F3_MAX,
                       reasons=dict(Counter(other.values()).most_common(60)), stop=bool(n and len(other) / n > F3_MAX)),
               F4=timing_summary(M["timing"], e), versions=ef.versions(), versions_slices=M["versions"],
               engines_job=v0, si_decision=C.SI_DECISION,
               code=dict(feat_lf=D.sha256(THIS), d1_features=D.sha256(DF.__file__)))
    rep["stop"] = bool(stop)
    rep["stop_reasons"] = stop
    D.write_json(F_REPORT, rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("not_ok", "versions_slices")}, indent=1, default=str))
    if stop:
        print(f"STOP: {stop}", file=sys.stderr)
    return 3 if stop else 0


def need_f_report():
    """for T / Q: results/F_report.json exists, has no STOP and belongs to the current X182_lf.parquet."""
    if not (F_REPORT.is_file() and D.X182_LF.is_file()):
        D.not_ready(f"{F_REPORT} / {D.X182_LF} missing (feat_lf.py merge)")
    r = D.read_json(F_REPORT)
    if r.get("stop"):
        D.stop(f"F_report.json STOP: {r.get('stop_reasons')}")
    if (r.get("x182") or {}).get("sha256") != D.sha256(D.X182_LF):
        D.not_ready(f"{D.X182_LF} is not the one of {F_REPORT} (rerun feat_lf.py merge)")
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["slice"] and len(a) == 3:
        sys.exit(cmd_slice(a[1], a[2]))
    if a == ["merge"]:
        sys.exit(cmd_merge())
    if a[:1] == ["_slice"] and len(a) == 3:
        sys.exit(_child_slice(a[1], a[2]))
    print(__doc__)
    sys.exit(2)
