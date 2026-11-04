#!/usr/bin/env python3
"""ts_geom.py — shared TS-optimisation geometry evaluator (DEPLOY-V2 Phase Z now, FT-2 (spec 5) later).

  python ts_geom.py run <spec> <list.csv> <label|g2> <out_dir> [i n]
  python ts_geom.py summarize <out_dir>
  python ts_geom.py x_report [--root $MLT/x] [--run-tmpl FT_LODO_{d}_al] [--dir-tmpl {root}/{run}/{start}]
                             [--ref-root $MLT/ft2] [--ref-tmpl FT_{k}_al] [--lists <lists.json>] [--out <json>]
  python ts_geom.py d0_eval run <i|@I> <n> [--models FT_all_s0_al,FT_all_s1_al,FT_all_s2_al] [--starts label,g2]
                             [--root $MLT/d0eval] [--ref $MLT/dft/plan/d0_holdout_ref.json] [--workers W]
  python ts_geom.py d0_eval report [same options] [--out results/D0_eval_report.json]

run: one GPU job = one MLTS server (mlts_bridge.start_server, this job's GPU) + a pool of CPU worker threads, each
  running mlts_bridge.optimize_ts (ORCA `! ExtOpt OptTS`, D1 OptTS block, MLIP initial Hessian) for one reaction at a
  time. <spec>: a model spec (name / FT_<k> / .model / JSON, mlts_bridge.resolve_spec); 'xtb' = ORCA XTB2 ALPB(water)
  OptTS Freq with the G2 block (CPU only, no server). <list.csv>: column rxn_id. Start: 'label' = the label DFT TS
  ($LAD/dft_final/<rxn>/ts.xyz), 'g2' = the G2 TS in label order ($G2/g2_final/<rxn>/ts.xyz). [i n]: this task does
  rows k with k % n == i ('@I' = $SLURM_ARRAY_TASK_ID).
  Per reaction one atomic JSON record <out_dir>/recs/<rxn_id>.json; existing records are skipped (idempotent).
  Workers stop claiming when less than MLT_RESERVE_H (2) h of the MLT_WALL_H (48) h wall are left (MLT_JOB_T0).
  Env: MLT_WORKERS (default: allocated CPUs - 2 for MLIPs, all for xtb), MLTS_RECALC (mlts_bridge.RECALC_DEFAULT),
  MLT_KEEP_OUT=1 keeps every ORCA .out.gz (default: failures only, <out_dir>/fail/).
summarize: <out_dir>/geom.csv (one row per reaction) and <out_dir>/summary.json = the FT-2 table: median / 90th
  percentile |Δd|, median heavy-atom RMSD, same-TS fraction, n_imag == 1 fraction, convergence, cycles, wall, GPU-s;
  for g2 starts also split by the g1 shape class (analysis/d1_g2/results/C_compare.csv same_ts_g1: True = same,
  False = different, blank = none) and by dipole.
x_report (spec 12 Phase X, report only; MLTENV=mlts or reactot, CPU): per left-out dipole d the FT-2 table of
  FT_LODO_<d>_al on dipole d's reactions ($MLT/x/FT_LODO_<d>_al/{label,g2}/recs, read only), pass_a = FT-2 (a) on the
  label start, the g2 start split by g1 class, and the in-domain reference (FT_<k>_al fold-k tests, same dipole) ->
  results/X_report.json {complete, per_dipole{<dipole>: {..., pass_a}}, note}; exit 0 complete, 2 incomplete.
d0_eval (user decisions D0-plus / D0-gauss, report only, decision D0-eval): the FT_all_s<i>_al committee on the D0
  evaluation-only reactions of gen_d0 ($MLT/dft/plan/d0_holdout_ref.json): reference TS = the Coley G16 TS, starts
  'label' = that TS and 'g2' = the G2c xTB TS; records $MLT/d0eval/<model>/<start>/recs/<d0_id>.json (charged
  reactions: not_covered); report -> results/D0_eval_report.json {complete, per_model, pooled, by_class, floor_rmsd_A}.
Metrics per reaction (label-order atoms): |Δd| = max over the two label forming pairs (labels formed_pairs_ts) of
  |d_opt - d_label|; RMSD = D1 rmsd.same_ts mapped heavy-atom RMSD (also the same-order Kabsch RMSD); same_ts =
  rmsd.same_ts(label, opt) with 0.10 A / 0.02 A; n_imag = MLIP-Hessian modes below -50 cm-1 (xtb: ORCA Freq).
Convergence (decision TSG-tight = DECISIONS A-1 '수렴 기준'): `run` (FT-2, X) optimises with TightOpt (MLT_TSG_TIGHT =
  tight | verytight | none, default tight; recorded per record as 'tight'); the Python API run_list / evaluate_one keep
  tight=False unless asked (Phase Z ran through run_list without it and stays reproducible)."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as C  # noqa: E402
import mlts_bridge as B  # noqa: E402

WALL_H = float(os.environ.get("MLT_WALL_H", "48"))
RESERVE_H = float(os.environ.get("MLT_RESERVE_H", "2"))
T_START = time.time()
FT2 = dict(dd_median_A=0.010, dd_p90_A=0.030, rmsd_median_A=0.02, same_ts_frac=0.90, nimag1_frac=0.95)  # spec 5
MAX_PIPELINE_ERR = 20


def tight_kw(v):
    """MLT_TSG_TIGHT / --tight value -> mlts_bridge.optimize_ts tight argument (False | 'tight' | 'verytight')."""
    s = str(v if v is not None else "").strip().lower()
    return False if s in ("", "0", "none", "false", "default", "no") else s


TIGHT_RUN = tight_kw(os.environ.get("MLT_TSG_TIGHT", "tight"))     # decision TSG-tight (the `run` CLI: FT-2 / X)


def time_left_h():
    t0 = float(os.environ.get("MLT_JOB_T0") or T_START)
    return WALL_H - (time.time() - t0) / 3600.0


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def g1_classes():
    import pandas as pd
    if not C.C_COMPARE.is_file():
        return {}
    Cc = pd.read_csv(C.C_COMPARE, usecols=["rxn_id", "same_ts_g1"], dtype={"same_ts_g1": str}, keep_default_na=False)
    f = {"true": "same", "false": "different"}
    return {int(r): f.get(str(v).strip().lower(), "none") for r, v in zip(Cc.rxn_id, Cc.same_ts_g1)}


def target_rows(rxns=None):
    T = C.targets()
    if rxns is not None:
        T = T[T.rxn_id.isin(set(int(r) for r in rxns))]
    return {int(r.rxn_id): dict(job_id=r.job_id, dipole=r.dipole, fold_T1=int(r.fold_T1), n_atoms=int(r.n_atoms),
                                formed_pairs_ts=str(r.formed_pairs_ts), skeleton=r.skeleton)
            for r in T.itertuples()}


def start_geometry(rxn, start):
    p = C.label_ts(rxn) if start == "label" else C.g2_ts_label_order(rxn) if start == "g2" else None
    if p is None:
        raise ValueError(f"start {start!r}")
    return p


def rec_path(out_dir, rxn):
    return Path(out_dir) / "recs" / f"{int(rxn)}.json"


def evaluate_one(rxn, row, spec, start, sock, out_dir, work_root, cls=None, tight=False, lab=None, sp=None):
    """optimise one reaction's TS and compare with the label TS -> record dict (also written to recs/). tight: the
    optimize_ts convergence (decision TSG-tight); lab / sp: explicit label / start TS files (G-4 D0 reactions, whose
    label TS is the Coley TS and start the G2c xTB TS) instead of the D1 paths."""
    t0 = time.time()
    rec = dict(rxn_id=int(rxn), job_id=row["job_id"], dipole=row["dipole"], fold_T1=row["fold_T1"],
               n_atoms=row["n_atoms"], start=start, model=B.spec_name(spec), g1_class=(cls or {}).get(int(rxn), "none"),
               host=os.uname().nodename, job=B.jobtag(), time=C.now(), tight=tight)
    lab = Path(lab) if lab is not None else C.label_ts(rxn)
    sp = Path(sp) if sp is not None else start_geometry(rxn, start)
    if not lab.is_file():
        rec.update(status="no_label")
        return rec
    if not sp.is_file():
        rec.update(status="no_start", start_file=str(sp))
        return rec
    ls, lp = C.read_xyz(lab)
    ss, spos = C.read_xyz(sp)
    if list(ls) != list(ss):
        rec.update(status="no_start", reason="start atom order differs from the label TS")
        return rec
    if not B.covers(spec, ls):
        rec.update(status="not_covered", reason="model has no parameters for an element (Si)")
        return rec
    pairs = C.formed_pairs(row["formed_pairs_ts"])
    d_lab = B.dists(lp, pairs)
    d_st = B.dists(spos, pairs)
    rec.update(pairs=[list(p) for p in pairs], d_label=d_lab, d_start=d_st,
               start_dd_max=float(np.max(np.abs(np.subtract(d_st, d_lab)))))
    if start != "label":
        s0, r0, f0 = B.same_ts((ls, lp), (ss, spos), d_lab, d_st)
        rec.update(start_same_ts=s0, start_rmsd=r0, start_dform=f0)
    work = Path(work_root) / str(int(rxn))
    shutil.rmtree(work, ignore_errors=True)
    try:
        r = B.optimize_ts(spec, ls, spos, pairs, work, sock, tight=tight)
        po = np.asarray(r["pos_A"], float)
        d_opt = B.dists(po, pairs)
        dd = list(np.subtract(d_opt, d_lab))
        same, rm, df = B.same_ts((ls, lp), (ls, po), d_lab, d_opt)
        rec.update({k: r.get(k) for k in ("status", "converged", "n_cycles", "n_segments", "n_ext_calls", "gpu_s",
                                          "n_hess_calls", "n_imag", "n_imag_10", "imag_cm", "freqs_low", "recalc",
                                          "start_hess", "inhess_read", "e_final_eh", "gmax_final_eh_bohr",
                                          "client_overhead_ms", "max_iter_hit", "hessian_source", "rc", "orca_wall_s",
                                          "orca_side_ms_per_call")})
        rec.update(d_opt=d_opt, dd=[float(x) for x in dd], dd_max=float(np.max(np.abs(dd))), same_ts=same,
                   rmsd_heavy=rm if np.isfinite(rm) else None, dform_sorted=df,
                   rmsd_direct=B.heavy_rmsd_direct(ls, lp, po), opt_wall_s=r.get("wall_s"),
                   pos_A=np.round(po, 6).tolist())
        keep = os.environ.get("MLT_KEEP_OUT") == "1" or rec["status"] != "ok"
        if keep:
            outs = sorted(work.glob("seg*/ts.out")) or sorted(work.glob("ts.out"))
            if outs:
                dst = C.assert_write_ok(Path(out_dir) / "fail" / f"{int(rxn)}.out.gz")
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}_{threading.get_ident()}")
                tmp.write_bytes(gzip.compress(outs[-1].read_bytes(), 6))
                os.replace(tmp, dst)
                rec["out_gz"] = str(dst)
            if r.get("out_tail") and rec["status"] != "ok":
                rec["out_tail"] = r["out_tail"][-1500:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rec["wall_s"] = time.time() - t0
    return rec


class Ctx:
    def __init__(self):
        self.mu = threading.Lock()
        self.halt = ""
        self.timed_out = False
        self.n_err = 0
        self.counts = {}


def run_list(spec, rxns, start, out_dir, workers=None, sock=None, rows=None, log=print, tight=False):
    """evaluate every reaction in rxns that has no record yet; returns run info. Starts (and stops) a server for
    MLIP specs unless sock is given. tight: optimize_ts convergence (the `run` CLI passes TIGHT_RUN)."""
    spec = B.resolve_spec(spec)
    out_dir = C.assert_write_ok(Path(out_dir))
    (out_dir / "recs").mkdir(parents=True, exist_ok=True)
    rows = rows or target_rows(rxns)
    cls = g1_classes()
    todo = [int(r) for r in rxns if not rec_path(out_dir, r).is_file()]
    info = dict(model=B.spec_name(spec), spec=spec, start=start, n_list=len(rxns), n_todo=len(todo), host=os.uname().nodename,
                job=B.jobtag(), started=C.now(), recalc=B.RECALC_DEFAULT, tight=tight)
    if not todo:
        info["note"] = "all records exist"
        return info
    xtb = spec["backend"] == "xtb"
    nw = int(workers or os.environ.get("MLT_WORKERS") or (cpus() if xtb else max(1, cpus() - 2)))
    nw = max(1, min(nw, len(todo)))
    info["workers"] = nw
    h = None
    if not xtb and not sock:
        h = B.start_server(spec)
        sock = h.socket
        info["server_start"] = {k: v for k, v in (h.stats() or {}).items() if k != "spec"}
    work_root = B.local_tmp() / f"mlts_tsg_{B.jobtag()}_{os.getpid()}"
    work_root.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    it = iter(todo)

    def claim():
        with ctx.mu:
            if ctx.halt:
                return None
            if time_left_h() < RESERVE_H:
                ctx.timed_out = True
                return None
            return next(it, None)

    def loop():
        while True:
            rx = claim()
            if rx is None:
                return
            try:
                if rec_path(out_dir, rx).is_file():
                    continue
                rec = evaluate_one(rx, rows[rx], spec, start, sock, out_dir, work_root, cls, tight=tight)
                C.write_text(rec_path(out_dir, rx), json.dumps(rec, default=C._json_default) + "\n")   # compact
                with ctx.mu:
                    ctx.counts[rec["status"]] = ctx.counts.get(rec["status"], 0) + 1
                log(f"[ts_geom] {rx} {rec['status']} dd={rec.get('dd_max')} same={rec.get('same_ts')} "
                    f"nimag={rec.get('n_imag')} cyc={rec.get('n_cycles')} wall={rec.get('wall_s', 0):.1f}s")
            except Exception as exc:                             # noqa: BLE001  (no record: retried on resubmission)
                print(f"[ts_geom] pipeline error {rx}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
                      file=sys.stderr, flush=True)
                with ctx.mu:
                    ctx.n_err += 1
                    ctx.counts["pipeline_error"] = ctx.counts.get("pipeline_error", 0) + 1
                    if ctx.n_err > MAX_PIPELINE_ERR:
                        ctx.halt = f"more than {MAX_PIPELINE_ERR} pipeline errors"
    t0 = time.time()
    try:
        with ThreadPoolExecutor(nw) as ex:
            for f in [ex.submit(loop) for _ in range(nw)]:
                f.result()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        if h is not None:
            info["server_end"] = {k: v for k, v in (h.stats() or {}).items() if k != "spec"}
            h.shutdown()
    info.update(counts=ctx.counts, wall_s=time.time() - t0, halt=ctx.halt, timed_out=ctx.timed_out, finished=C.now())
    return info


# ================================================================ summary
def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def table(df):
    """FT-2 metrics of a set of records (DataFrame from load_records). Fractions over attempted reactions
    (status not in no_start / no_label / not_covered); failures count as not same and not n_imag == 1."""
    att = df[~df.status.isin(["no_start", "no_label", "not_covered"])]
    opt = att[att.dd_max.notna()]
    n = len(att)
    q = lambda s, p: _f(np.percentile(s.dropna(), p)) if s.notna().any() else None  # noqa: E731
    out = dict(n_attempted=int(n), n_optimised=int(len(opt)),
               n_converged=int((att.converged == True).sum()),                     # noqa: E712
               conv_frac=_f((att.converged == True).mean()) if n else None,       # noqa: E712
               dd_median_A=q(opt.dd_max, 50), dd_p90_A=q(opt.dd_max, 90), dd_mean_A=_f(opt.dd_max.mean()),
               rmsd_median_A=q(opt.rmsd_heavy, 50), rmsd_direct_median_A=q(opt.rmsd_direct, 50),
               same_ts_frac=_f((att.same_ts == True).sum() / n) if n else None,   # noqa: E712
               nimag1_frac=_f((att.n_imag == 1).sum() / n) if n else None,
               n_imag_counts={str(k): int(v) for k, v in att.n_imag.fillna(-1).astype(int).value_counts().items()},
               cycles_median=q(opt.n_cycles, 50), cycles_p90=q(opt.n_cycles, 90),
               wall_median_s=q(att.wall_s, 50), wall_p90_s=q(att.wall_s, 90),
               gpu_s_median=q(opt.gpu_s, 50) if "gpu_s" in opt else None,
               ext_calls_median=q(opt.n_ext_calls, 50) if "n_ext_calls" in opt else None,
               status_counts={str(k): int(v) for k, v in df.status.value_counts().items()})
    if "dd_max" in opt and len(opt):
        conv = opt[opt.converged == True]                                         # noqa: E712
        out["dd_median_converged_A"] = q(conv.dd_max, 50) if len(conv) else None
    if "start_dd_max" in att and att.start_dd_max.notna().any():
        out["start_dd_median_A"] = q(att.start_dd_max, 50)
    if "start_same_ts" in att and att.start_same_ts.notna().any():
        out["start_same_ts_frac"] = _f((att.start_same_ts == True).sum() / n)   # noqa: E712
    out["ft2_a_pass"] = {k: (out.get(k) is not None and (out[k] <= v if k.startswith(("dd", "rmsd")) else out[k] >= v))
                         for k, v in FT2.items()}
    return out


def load_records(out_dir):
    import pandas as pd
    recs = []
    for p in sorted((Path(out_dir) / "recs").glob("*.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    for c in ("dd_max", "rmsd_heavy", "rmsd_direct", "n_imag", "converged", "same_ts", "n_cycles", "gpu_s",
              "n_ext_calls", "wall_s", "start_dd_max", "start_same_ts"):
        if c not in df:
            df[c] = None
    for c in ("dd_max", "rmsd_heavy", "rmsd_direct", "n_cycles", "gpu_s", "n_ext_calls", "wall_s", "start_dd_max"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_imag"] = pd.to_numeric(df["n_imag"], errors="coerce")
    return df


def summarize(out_dir):
    df = load_records(out_dir)
    if df.empty:
        C.not_ready(f"no records in {out_dir}/recs")
    flat = df.drop(columns=[c for c in ("pos_A", "out_tail", "freqs_low", "imag_cm", "inhess_read", "pairs")
                            if c in df], errors="ignore")
    for c in ("d_label", "d_opt", "d_start", "dd"):
        if c in flat:
            flat[c] = flat[c].map(lambda v: " ".join(f"{x:.5f}" for x in v) if isinstance(v, list) else v)
    C.write_csv(Path(out_dir) / "geom.csv", flat)
    S = dict(out_dir=str(out_dir), models=sorted(df.model.dropna().unique().tolist()),
             starts=sorted(df.start.dropna().unique().tolist()), time=C.now(), thresholds_ft2_a=FT2,
             tight=sorted({str(x) for x in df.tight}) if "tight" in df else ["False (pre-TSG-tight records)"],
             all=table(df), by_dipole={d: table(g) for d, g in df.groupby("dipole")})
    if (df.start == "g2").any():
        S["by_g1_class"] = {c: table(g) for c, g in df.groupby("g1_class")}
    C.write_json(Path(out_dir) / "summary.json", S)
    return S


# ================================================================ Phase X report (spec 12, report only)
FT2_A_KEYS = ("dd_median_A", "dd_p90_A", "rmsd_median_A", "same_ts_frac")      # FT-2 (a); n_imag reported apart
X_PARTIAL = 0.995                                     # = pipeline GEOM_PARTIAL (coverage accepted for a folder)
X_KEEP = ("n_attempted", "n_optimised", "n_converged", "conv_frac", "dd_median_A", "dd_p90_A", "dd_mean_A",
          "rmsd_median_A", "rmsd_direct_median_A", "same_ts_frac", "nimag1_frac", "n_imag_counts", "cycles_median",
          "wall_median_s", "gpu_s_median", "status_counts", "start_dd_median_A", "start_same_ts_frac")


def _x_table(df):
    t = table(df)
    return {k: t.get(k) for k in X_KEEP if k in t}


def _x_records(out_dir, dipole):
    """records of one ts_geom out_dir restricted to one dipole (read only: nothing is written into out_dir)."""
    df = load_records(out_dir)
    if df.empty or "dipole" not in df:
        return df
    return df[df.dipole == dipole]


def x_report(root=None, run_tmpl="FT_LODO_{d}_al", dir_tmpl="{root}/{run}/{start}", ref_root=None,
             ref_tmpl="FT_{k}_al", lists_json=None, out=None):
    """spec 12 Phase X: per left-out dipole d, the FT-2 table of FT_LODO_<d> on the reactions of dipole d for (a) the
    label-TS start and (b) the G2-TS start (by g1 shape class), pass_a = the FT-2 (a) criteria on (a); for reference
    the same dipole's FT-2 numbers of the in-domain fold models (FT_<k>_al on their fold-k tests, pooled). Tables are
    computed in memory from the per-reaction records (recs/*.json); no file under root / ref_root is written.
    complete = both starts present for every dipole with >= 99.5 % of the dipole list recorded (lists.json; without
    a list: >= 1 record). -> results/X_report.json."""
    root = Path(root or C.MLT / "x")
    ref_root = Path(ref_root or C.MLT / "ft2")
    lj = Path(lists_json or C.MLT / "pipeline" / "lists" / "lists.json")
    L = (json.loads(lj.read_text()) if lj.is_file() else {}).get("files") or {}
    per, missing = {}, []
    for d, dn in enumerate(C.DIPOLES):
        run = run_tmpl.format(d=d)
        nl = int((L.get(f"dip{d}") or {}).get("n") or 0) or None
        ent = dict(d=d, model=run, n_list=nl)
        for s in ("label", "g2"):
            od = Path(dir_tmpl.format(root=root, run=run, start=s, d=d))
            df = _x_records(od, dn)
            if df.empty:
                ent[s] = dict(present=False, out_dir=str(od))
                missing.append(f"{dn}/{s}")
                continue
            cov = len(df) / nl if nl else None
            e = dict(present=True, out_dir=str(od), n_records=int(len(df)), coverage=cov, **_x_table(df))
            if s == "g2" and "g1_class" in df:
                e["by_g1_class"] = {c: _x_table(g) for c, g in df.groupby("g1_class")}
            ent[s] = e
            if (nl and cov < X_PARTIAL) or not len(df):
                missing.append(f"{dn}/{s} (coverage {len(df)}/{nl})")
        a = ent["label"]
        if a.get("present") and a.get("n_attempted"):
            chk = {k: (a.get(k) is not None and (a[k] >= FT2[k] if k.startswith("same") else a[k] <= FT2[k]))
                   for k in FT2_A_KEYS}
            ent["pass_a"], ent["pass_a_detail"] = bool(all(chk.values())), chk
            ent["nimag1_pass"] = a.get("nimag1_frac") is not None and a["nimag1_frac"] >= FT2["nimag1_frac"]
        else:
            ent["pass_a"], ent["pass_a_detail"], ent["nimag1_pass"] = None, None, None
        ref = {}
        for s in ("label", "g2"):
            frames = [_x_records(ref_root / ref_tmpl.format(k=k) / s, dn) for k in range(5)]
            frames = [f for f in frames if not f.empty]
            if frames:
                import pandas as pd
                ref[s] = dict(n_records=int(sum(len(f) for f in frames)), **_x_table(pd.concat(frames, ignore_index=True)))
        ent["in_domain"] = ref or None
        per[dn] = ent
    fails = [dn for dn, e in per.items() if e["pass_a"] is False]
    passes = [dn for dn, e in per.items() if e["pass_a"] is True]
    complete = not missing
    if fails:
        note = (f"FT-2 (a) fails for {', '.join(fails)}: 새 dipole 계열에는 그 계열 DFT 데이터를 조금 넣어 미세조정해야 함 "
                f"(a new dipole family needs some DFT data of its own in the fine-tuning; spec 12, report only).")
    elif passes and len(passes) == len(per):
        note = ("FT-2 (a) holds for every left-out dipole: the MLTS reached label-level TS geometry on a dipole family "
                "without that family's DFT data (spec 12, report only).")
    else:
        note = f"FT-2 (a) passes for {', '.join(passes) or 'none'}; not judged: " \
               f"{', '.join(dn for dn, e in per.items() if e['pass_a'] is None) or 'none'} (spec 12, report only)."
    if missing:
        note += f" Incomplete: {'; '.join(missing)}."
    rep = dict(phase="X", spec_section="12 Phase X (LODO, report only)", time=C.now(), complete=complete,
               n_todo=len(missing), stop=None, est=dict(units_total=1, unit_wall_h=0.05, workers=1),
               root=str(root), run_tmpl=run_tmpl, ref_root=str(ref_root), lists_json=str(lj) if lj.is_file() else None,
               thresholds_ft2_a={k: FT2[k] for k in FT2_A_KEYS}, nimag1_min=FT2["nimag1_frac"],
               per_dipole=per, n_pass_a=len(passes), n_fail_a=len(fails), missing=missing, note=note,
               rule="(a) label-TS start, (b) G2-TS start, reactions of the left-out dipole only; pass_a = median |dd| "
                    "<= 0.010 A, p90 <= 0.030 A, median heavy RMSD <= 0.02 A, same TS >= 0.90 (spec 5 FT-2 (a)); "
                    "nimag1_pass (n_imag == 1 >= 0.95) reported apart; autodE is not run (spec 12)")
    C.write_json(Path(out) if out else C.RES / "X_report.json", rep)
    return rep


# ================================================================ D0 supplementary geometry evaluation (D0-plus)
D0E_ROOT = C.MLT / "d0eval"
D0E_REF = C.MLT / "dft" / "plan" / "d0_holdout_ref.json"      # gen_d0.write_holdout_ref (decision D0-hold-ref)
D0E_MODELS = ("FT_all_s0_al", "FT_all_s1_al", "FT_all_s2_al")    # the deployment committee (FT-fin)
D0E_STARTS = dict(label="reference", g2="xtb_g2c")             # FT-2 (a): the Coley TS itself; (b): the G2c xTB TS
GAUSS_FLOOR_A = 0.009                                          # Coley G16 TS vs the ORCA label-level stationary point


def d0e_load(ref=None):
    """(holdout ref json, its evaluation units); not ready (exit 2) while gen_d0 plan_d0 --n-holdout has not run."""
    p = Path(ref or D0E_REF)
    if not p.is_file():
        C.not_ready(f"{p} missing: `gen_d0.py plan_d0 --n-holdout 100` has not written the D0 evaluation list yet")
    J = C.read_json(p)
    return J, list(J.get("units") or [])


def d0e_row(u):
    """evaluate_one row of a D0 evaluation unit (label pairs as formed_pairs_ts text, label atom order)."""
    return dict(job_id=f"D0_{int(u['d0_id'])}", dipole=str(u["d0_class"]), fold_T1=-9, n_atoms=int(u["n_atoms"]),
                formed_pairs_ts=" ".join(f"{int(a)}-{int(b)}" for a, b in u["pairs"]), skeleton="")


def d0e_dir(root, model, start):
    """<root>/<model name>/<start> (a model given as a file path uses its file name: never a path outside root)."""
    import re
    return Path(root) / re.sub(r"[^A-Za-z0-9_.-]", "_", Path(str(model)).name) / start


def d0e_items(models, starts, units):
    """(model, start, unit), model-major: a contiguous slice of an array task needs one server at a time."""
    return [(m, s, u) for m in models for s in starts for u in units]


def _d0e_skip_rec(u, model, start, status, reason):
    return dict(rxn_id=int(u["d0_id"]), job_id=f"D0_{int(u['d0_id'])}", dipole=str(u["d0_class"]), fold_T1=-9,
                n_atoms=int(u["n_atoms"]), start=start, model=str(model), g1_class="none", status=status, reason=reason,
                set="d0", d0_id=int(u["d0_id"]), d0_class=str(u["d0_class"]), charge=int(u.get("charge") or 0),
                host=os.uname().nodename, job=B.jobtag(), time=C.now(), tight=TIGHT_RUN)


def d0_eval_run(i, n, models=D0E_MODELS, starts=tuple(D0E_STARTS), root=None, ref=None, workers=None, log=print):
    """array task i of n: its contiguous slice of (model, start, D0 evaluation reaction); per model one MLTS server and
    a worker pool running evaluate_one (TightOpt, label = the Coley TS, start = the Coley TS or the G2c xTB TS).
    Charged reactions get a not_covered record (the bridge evaluates every structure at charge 0). A reaction that
    raises twice gets a pipeline_error record while the server lives. -> exit 0 slice recorded, 2 incomplete (wall /
    model not ready), 1 server died, 3 > 20 pipeline errors."""
    root = Path(root or D0E_ROOT)
    J, units = d0e_load(ref)
    items = d0e_items(models, starts, units)
    lo, hi = (len(items) * i) // n, (len(items) * (i + 1)) // n
    mine = [x for x in items[lo:hi] if not rec_path(d0e_dir(root, x[0], x[1]), x[2]["d0_id"]).is_file()]
    print(f"[d0_eval {i}/{n}] {hi - lo} items, {len(mine)} to do ({len(units)} D0 evaluation reactions, models "
          f"{list(models)}, starts {list(starts)}, tight={TIGHT_RUN}), {time_left_h():.1f} h left", flush=True)
    rc = 0
    for m in dict.fromkeys(x[0] for x in mine):
        todo = [x for x in mine if x[0] == m]
        for s in starts:
            (C.assert_write_ok(d0e_dir(root, m, s)) / "recs").mkdir(parents=True, exist_ok=True)
        for x in [x for x in todo if int(x[2].get("charge") or 0) != 0]:
            C.write_text(rec_path(d0e_dir(root, m, x[1]), x[2]["d0_id"]), json.dumps(_d0e_skip_rec(
                x[2], m, x[1], "not_covered", "total charge != 0: the MLTS bridge evaluates at charge 0")) + "\n")
        todo = [x for x in todo if int(x[2].get("charge") or 0) == 0]
        if not todo:
            continue
        try:
            spec = B.resolve_spec(m)
        except ValueError as e:
            C.not_ready(f"model {m}: {e}")
        if spec.get("weights") and not Path(spec["weights"]).is_file():
            C.not_ready(f"model {m}: weights {spec['weights']} missing (FT-fin not finished?)")
        h = B.start_server(spec)
        work_root = B.local_tmp() / f"mlts_d0e_{B.jobtag()}_{os.getpid()}"
        work_root.mkdir(parents=True, exist_ok=True)
        ctx, it = Ctx(), iter(todo)
        errs = {}

        def alive():
            return h.proc is None or h.proc.poll() is None

        def claim():
            with ctx.mu:
                if ctx.halt:
                    return None
                if time_left_h() < RESERVE_H:
                    ctx.timed_out = True
                    return None
                if not alive():
                    ctx.halt = ctx.halt or "MLTS server process died"
                    return None
                return next(it, None)

        def loop():
            while True:
                x = claim()
                if x is None:
                    return
                _, s, u = x
                od = d0e_dir(root, m, s)
                if rec_path(od, u["d0_id"]).is_file():
                    continue
                st = (u.get("starts") or {}).get(D0E_STARTS[s]) or (u.get("reference") or {}).get("xyz")
                rec, err = None, None
                for _ in (1, 2):
                    try:
                        rec = evaluate_one(int(u["d0_id"]), d0e_row(u), spec, s, h.socket, od, work_root / s, None,
                                           tight=TIGHT_RUN, lab=(u.get("reference") or {}).get("xyz"), sp=st)
                        break
                    except Exception as e:                                 # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"[:500]
                        print(f"[d0_eval] pipeline error {m}/{s}/{u['d0_id']}: {err}\n{traceback.format_exc(limit=5)}",
                              file=sys.stderr, flush=True)
                        if not alive():
                            break
                if rec is None:
                    with ctx.mu:
                        ctx.n_err += 1
                        errs[(s, int(u["d0_id"]))] = (u, err)
                        if ctx.n_err > MAX_PIPELINE_ERR:
                            ctx.halt = f"more than {MAX_PIPELINE_ERR} pipeline errors"
                    continue
                rec.update(set="d0", d0_id=int(u["d0_id"]), d0_class=str(u["d0_class"]), charge=0,
                           reference="coley_g16_ts", start_kind=D0E_STARTS[s])
                C.write_text(rec_path(od, u["d0_id"]), json.dumps(rec, default=C._json_default) + "\n")
                log(f"[d0_eval] {m}/{s}/{u['d0_id']} {rec['status']} dd={rec.get('dd_max')} same={rec.get('same_ts')} "
                    f"nimag={rec.get('n_imag')} wall={rec.get('wall_s', 0):.1f}s")
        try:
            nw = max(1, min(int(workers or os.environ.get("MLT_WORKERS") or max(1, cpus() - 2)), len(todo)))
            with ThreadPoolExecutor(nw) as ex:
                for f in [ex.submit(loop) for _ in range(nw)]:
                    f.result()
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
            up = alive()
            h.shutdown()
        if not ctx.halt and up:                                   # isolated failures closed as records
            for (s, rid), (u, err) in errs.items():
                C.write_text(rec_path(d0e_dir(root, m, s), rid), json.dumps(dict(
                    _d0e_skip_rec(u, m, s, "pipeline_error", "raised twice"), error=err)) + "\n")
        if ctx.halt:
            rc = 1 if not up else 3
            break
        if ctx.timed_out:
            rc = 2
            break
    for m in models:
        for s in starts:
            if (d0e_dir(root, m, s) / "recs").is_dir() and any((d0e_dir(root, m, s) / "recs").glob("*.json")):
                summarize(d0e_dir(root, m, s))
    left = [x for x in items[lo:hi] if not rec_path(d0e_dir(root, x[0], x[1]), x[2]["d0_id"]).is_file()]
    return rc or (2 if left else 0)


def d0_eval_report(models=D0E_MODELS, starts=tuple(D0E_STARTS), root=None, ref=None, out=None):
    """results/D0_eval_report.json (report only, user decisions D0-plus / D0-gauss): per model and start the FT-2
    table on the D0 evaluation reactions (reference = the Coley G16 TS), pooled over the models and by D0 class, with
    the FT-2 (a) thresholds as flags and the Gaussian-ORCA floor (~0.009 A RMSD). complete = every (model, start, unit)
    recorded; the evaluation never enters a grade (spec: D1 only)."""
    import pandas as pd
    root = Path(root or D0E_ROOT)
    J, units = d0e_load(ref)
    ids = [int(u["d0_id"]) for u in units]
    per, missing, frames = {}, [], {s: [] for s in starts}
    for m in models:
        for s in starts:
            df = load_records(d0e_dir(root, m, s))
            if not df.empty:
                df = df[df.rxn_id.astype(int).isin(set(ids))]
            have = set(int(x) for x in df.rxn_id) if not df.empty else set()
            miss = [x for x in ids if x not in have]
            if miss:
                missing.append(f"{m}/{s}: {len(miss)} of {len(ids)}")
            e = dict(n_records=len(have), out_dir=str(d0e_dir(root, m, s)))
            if not df.empty:
                e.update(table(df))
                e["by_class"] = {c: _x_table(g) for c, g in df.groupby("dipole")}
                frames[s].append(df)
            per.setdefault(m, {})[s] = e
    pooled = {}
    for s in starts:
        if frames[s]:
            P_ = pd.concat(frames[s], ignore_index=True)
            pooled[s] = dict(table(P_), by_class={c: _x_table(g) for c, g in P_.groupby("dipole")})
    n_charged = sum(int(u.get("charge") or 0) != 0 for u in units)
    rep = dict(phase="D0-eval", spec_section="DECISIONS D0-plus / D0-gauss (supplementary, report only; grades stay D1)",
               time=C.now(), complete=not missing, n_todo=len(missing), stop=None,
               est=dict(units_total=len(models) * len(starts) * len(units), unit_wall_h=0.02, workers=14),
               n_units=len(units), n_charged_not_covered=n_charged, models=list(models), starts=dict(D0E_STARTS),
               holdout_sha256=J.get("holdout_sha256"), ref=str(Path(ref or D0E_REF)), root=str(root),
               thresholds_ft2_a={k: FT2[k] for k in FT2_A_KEYS}, nimag1_min=FT2["nimag1_frac"],
               floor_rmsd_A=GAUSS_FLOOR_A, per_model=per, pooled=pooled, missing=missing,
               rule="reference TS = the Coley G16 TS (D0-gauss, D0-hold-ref); label start = that TS (FT-2 (a) analogue), "
                    "g2 start = the G2c xTB TS (FT-2 (b)); TightOpt; MLTS = FT_all_s0-2_al (trained without these "
                    "reactions' TS-derived structures, finetune FT-d0hold); charged reactions not covered (bridge at "
                    "charge 0); the Coley TS is not an ORCA label-level stationary point (~0.009 A RMSD floor), so "
                    "|Δd| / RMSD below about that floor are not meaningful",
               note=("no D0 evaluation reactions in the holdout list" if not units else
                     "report only: supplementary D0 geometry evaluation, not a grade"))
    C.write_json(Path(out) if out else C.RES / "D0_eval_report.json", rep)
    return rep


# ================================================================ CLI
def read_list(path):
    """CSV with a rxn_id column, or (tests) a comma-separated list of rxn_ids."""
    import pandas as pd
    if not Path(path).is_file() and all(t.strip().isdigit() for t in str(path).split(",")):
        return [int(t) for t in str(path).split(",")]
    L = pd.read_csv(path)
    if "rxn_id" not in L:
        C.stop(f"{path}: no rxn_id column")
    return [int(x) for x in L.rxn_id]


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] not in ("run", "summarize", "x_report", "d0_eval"):
        print(__doc__)
        return 2
    if a[0] == "d0_eval":
        import argparse
        ap = argparse.ArgumentParser(prog="ts_geom.py d0_eval")
        ap.add_argument("what", choices=["run", "report"])
        ap.add_argument("i", nargs="?", default="0")
        ap.add_argument("n", nargs="?", type=int, default=1)
        ap.add_argument("--models", default=",".join(D0E_MODELS))
        ap.add_argument("--starts", default=",".join(D0E_STARTS))
        ap.add_argument("--root", default=None)
        ap.add_argument("--ref", default=None)
        ap.add_argument("--out", default=None)
        ap.add_argument("--workers", type=int, default=None)
        o = ap.parse_args(a[1:])
        models = [x for x in o.models.split(",") if x]
        starts = [x for x in o.starts.split(",") if x]
        if any(s not in D0E_STARTS for s in starts):
            C.stop(f"--starts must be among {list(D0E_STARTS)}")
        if o.what == "run":
            i = C.array_task(o.i)
            if not 0 <= i < o.n:
                print(f"d0_eval: task {i} outside 0..{o.n - 1}", file=sys.stderr)
                return 2
            return d0_eval_run(i, o.n, models, starts, o.root, o.ref, o.workers)
        R = d0_eval_report(models, starts, o.root, o.ref, o.out)
        print(json.dumps(dict(complete=R["complete"], n_units=R["n_units"], missing=R["missing"][:10],
                              pooled={s: {k: v.get(k) for k in FT2_A_KEYS + ("nimag1_frac", "n_attempted")}
                                      for s, v in R["pooled"].items()}), indent=1, default=C._json_default))
        return 0 if R["complete"] else 2
    if a[0] == "x_report":
        import argparse
        ap = argparse.ArgumentParser(prog="ts_geom.py x_report")
        ap.add_argument("--root", default=None)
        ap.add_argument("--run-tmpl", default="FT_LODO_{d}_al")
        ap.add_argument("--dir-tmpl", default="{root}/{run}/{start}")
        ap.add_argument("--ref-root", default=None)
        ap.add_argument("--ref-tmpl", default="FT_{k}_al")
        ap.add_argument("--lists", default=None)
        ap.add_argument("--out", default=None)
        o = ap.parse_args(a[1:])
        R = x_report(o.root, o.run_tmpl, o.dir_tmpl, o.ref_root, o.ref_tmpl, o.lists, o.out)
        print(json.dumps(dict(complete=R["complete"], note=R["note"],
                              pass_a={k: v["pass_a"] for k, v in R["per_dipole"].items()}), indent=1,
                         default=C._json_default))
        return 0 if R["complete"] else 2
    if a[0] == "summarize":
        S = summarize(a[1])
        print(json.dumps(dict(all=S["all"], by_g1_class=S.get("by_g1_class")), indent=1, default=C._json_default))
        return 0
    if len(a) not in (5, 7):
        print(__doc__)
        return 2
    spec, lst, start, out_dir = a[1], a[2], a[3], a[4]
    if start not in ("label", "g2"):
        C.stop(f"start must be label or g2, not {start!r}")
    rx = read_list(lst)
    if len(a) == 7:
        i, n = C.array_task(a[5]), int(a[6])
        rx = [r for k, r in enumerate(rx) if k % n == i]
    info = run_list(spec, rx, start, out_dir, tight=TIGHT_RUN)
    C.write_json(Path(out_dir) / f"run_{B.jobtag()}.json", info)
    S = summarize(out_dir)
    print(json.dumps(dict(run={k: v for k, v in info.items() if k not in ("spec",)}, all=S["all"]), indent=1,
                     default=C._json_default))
    return 3 if info.get("halt") else 0


if __name__ == "__main__":
    sys.exit(main())
