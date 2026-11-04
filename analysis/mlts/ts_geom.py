#!/usr/bin/env python3
"""ts_geom.py — shared TS-optimisation geometry evaluator (DEPLOY-V2 Phase Z now, FT-2 (spec 5) later).

  python ts_geom.py run <spec> <list.csv> <label|g2> <out_dir> [i n]
  python ts_geom.py summarize <out_dir>

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
Metrics per reaction (label-order atoms): |Δd| = max over the two label forming pairs (labels formed_pairs_ts) of
  |d_opt - d_label|; RMSD = D1 rmsd.same_ts mapped heavy-atom RMSD (also the same-order Kabsch RMSD); same_ts =
  rmsd.same_ts(label, opt) with 0.10 A / 0.02 A; n_imag = MLIP-Hessian modes below -50 cm-1 (xtb: ORCA Freq)."""
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


def evaluate_one(rxn, row, spec, start, sock, out_dir, work_root, cls=None):
    """optimise one reaction's TS and compare with the label TS -> record dict (also written to recs/)."""
    t0 = time.time()
    rec = dict(rxn_id=int(rxn), job_id=row["job_id"], dipole=row["dipole"], fold_T1=row["fold_T1"],
               n_atoms=row["n_atoms"], start=start, model=B.spec_name(spec), g1_class=(cls or {}).get(int(rxn), "none"),
               host=os.uname().nodename, job=B.jobtag(), time=C.now())
    lab = C.label_ts(rxn)
    sp = start_geometry(rxn, start)
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
        r = B.optimize_ts(spec, ls, spos, pairs, work, sock)
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


def run_list(spec, rxns, start, out_dir, workers=None, sock=None, rows=None, log=print):
    """evaluate every reaction in rxns that has no record yet; returns run info. Starts (and stops) a server for
    MLIP specs unless sock is given."""
    spec = B.resolve_spec(spec)
    out_dir = C.assert_write_ok(Path(out_dir))
    (out_dir / "recs").mkdir(parents=True, exist_ok=True)
    rows = rows or target_rows(rxns)
    cls = g1_classes()
    todo = [int(r) for r in rxns if not rec_path(out_dir, r).is_file()]
    info = dict(model=B.spec_name(spec), spec=spec, start=start, n_list=len(rxns), n_todo=len(todo), host=os.uname().nodename,
                job=B.jobtag(), started=C.now(), recalc=B.RECALC_DEFAULT)
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
                rec = evaluate_one(rx, rows[rx], spec, start, sock, out_dir, work_root, cls)
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
             all=table(df), by_dipole={d: table(g) for d, g in df.groupby("dipole")})
    if (df.start == "g2").any():
        S["by_g1_class"] = {c: table(g) for c, g in df.groupby("g1_class")}
    C.write_json(Path(out_dir) / "summary.json", S)
    return S


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
    if not a or a[0] not in ("run", "summarize"):
        print(__doc__)
        return 2
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
    info = run_list(spec, rx, start, out_dir)
    C.write_json(Path(out_dir) / f"run_{B.jobtag()}.json", info)
    S = summarize(out_dir)
    print(json.dumps(dict(run={k: v for k, v in info.items() if k not in ("spec",)}, all=S["all"]), indent=1,
                     default=C._json_default))
    return 3 if info.get("halt") else 0


if __name__ == "__main__":
    sys.exit(main())
