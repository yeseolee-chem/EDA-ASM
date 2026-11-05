#!/usr/bin/env python3
r"""app_pilot.py — A'' pilot (user instruction 11-05 18:0x (2) / (3), results/DECISIONS.md): a label-level ORCA OptTS
started from the MLIP TS, with the MLIP Hessian as the first Hessian. ONE manual sbatch job outside the frozen launcher
(pipeline.py, tests/, PIPELINE.md and the launcher stages are not touched and do not import this file). sbatch only:
python and ORCA never run on the login node.

What is run
  sample   list $MLT/pipeline/lists/f4_test.csv (fold-4 test); dipole family = mlt_common.targets() (the FT-2 / ts_geom
           source). Eligible = the reaction has a FIRST-PASS FT-2 label record $MLT/ft2/FTpilot_AIMNet2_4/label/recs/
           <rxn>.json (not label_recalc5), attempted (status not no_start / no_label / not_covered), >= 1 MLIP OptTS
           cycle and a final MLIP geometry (pos_A; d_opt only -> regenerated). No filter on FT-2 pass / fail.
           numpy.random.default_rng(42); for each dipole in mlt_common.DIPOLES order: rng.choice(<sorted eligible
           rxn_ids of that dipole>, size=10, replace=False) -> 50 reactions (pick order kept). The records' model must
           be FTpilot_AIMNet2_4 and start 'label' (else STOP).
  start    MLIP TS = the record's pos_A (label atom order, ts_geom.evaluate_one, rounded to 1e-6 A). A record without
           pos_A: the same MLIP OptTS is re-run here (ts_geom.evaluate_one through a CPU MLTS server: same model, label
           start, the record's tight / recalc settings). Both cases report max |d(MLIP TS) - record d_opt| over the two
           label forming bonds (a consistency check for pos_A, the regeneration check otherwise).
  Hessian  mlts_bridge.hessian(spec, label symbols, MLIP TS): the code behind the FT-2 ExtOpt start Hessian (AIMNet2Dense:
           central differences of the dense AIMNet2 forces, h = 2e-3 A, charge 0 as the bridge), here in-process on the
           CPU; mlts_bridge.write_orca_hess -> runs/<rxn>/mlip.hess (label atom order = ORCA input order). Its
           frequencies (mlts_bridge.hessian_summary) are compared with the record's MLIP n_imag.
  input    d1_g3 opt.partb_text through gen_dft.d1_g3 (the D1 autodE L0 OptTS + Freq input that opt's check proved equal
           to the D1 run input: '! OptTS Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J  CPCM(Water)', %cpcm smd true SMDsolvent
           "water", the config_run.yaml orca_optts_block, forming-bond modify_internal blocks, %output xyzfile, %maxcore
           3000, %pal nprocs 8) with ONE change: 'Calc_Hess true' -> 'InHess Read' + 'InHessName "mlip.hess"'
           (Recalc_Hess 30, Trust -0.1, MaxIter 100 and Freq kept). Checked for every input: the %geom block lines, the
           '!' tokens, the SMD lines, %pal 8 / %maxcore 3000, and equality with g3l.g3l_text (the G-2 / G3L input) after
           its 'InHessName "g2.hess"' -> 'InHessName "mlip.hess"' + 'Recalc_Hess 30'. Charge = label charge1 +
           charge2, multiplicity 1, coordinates = the MLIP TS. Any mismatch = STOP ($MLT/app_pilot/STOP).
  run      ORCA 6.1.1 with opt.d1()'s orca and environment (the MLTENV=d1 job environment: D1 venv, own OpenMPI 4.1.8f
           without XPMEM / UCX, OMPI_MCA_pml=ob1, OMPI_MCA_coll_hcoll_enable=0, UCX_TLS=tcp,self,sm, binding none =
           gen_dft run_g2) in a node-local $TMPDIR folder; SLURM_CPUS_PER_TASK // 8 runs at a time, largest n_atoms first;
           an infra death (no SCF iteration and no cycle) is rerun once; a reaction started twice without a result is
           closed as wall_clipped; no new reaction when < APP_RESERVE_H (8) h of the 48 h wall are left (MLT_JOB_T0).
           Kept in runs/<rxn>/: prep.json, mlip.hess, mlip_ts.xyz, ts.inp, ts.out, ts.err (if not empty), ts.xyz,
           ts.hess, ts_trj.xyz, starts.json, result.json (written last = done; existing -> skipped). The work folder
           (.gbw, densities, tmp files) is deleted after parsing. ORCA's output is watched every 5 s for the wall time
           of each optimisation cycle, each Hessian and the Freq phase (approximate, +- 5 s).
  parse    optimisation cycles (= DFT gradients; 'CARTESIAN GRADIENT' count kept as a cross-check), DFT Hessians
           (lines 'SCF HESSIAN' before 'THE OPTIMIZATION HAS CONVERGED' = Recalc_Hess, after it = the final Freq,
           with the cycle each one belongs to; ORCA 6.1.1 does not echo Recalc_Hess in its settings block, so the
           unchanged D1 block's behaviour is read once from d1_g3 Part B B.1002979 (100 cycles) -> REF_PARTB.txt),
           ORCA converged / terminated / MaxIter, final geometry (ts.xyz), Freq frequencies (ts.hess
           $vibrational_frequencies): n_imag = modes below -50 cm-1 (ts_geom / mlts_bridge.IMAG_TOL_CM convention),
           n_neg = all negative modes; InHess evidence; level checks (orca_direct.level_checks / level_ok L0); ORCA
           module timings; wall per ORCA run.
  metrics  label TS = mlt_common.label_ts(rxn); forming pairs = targets formed_pairs_ts; same_ts = mlts_bridge.same_ts
           (D1 rmsd.same_ts: mapped heavy-atom RMSD <= 0.10 A AND sorted forming-distance difference <= 0.02 A,
           mlt_common.SAME_TS); |dd| = max over the two forming bonds |d(final) - d(label)|; converged_to_label = ORCA
           converged AND same_ts; core-h = sum over the reaction's ORCA runs (incl. an infra rerun) of wall x 8 / 3600.
           Fractions in the final summary are over the 50 planned reactions (no result / no Freq = not passed).

CLI (exit 0 = done, 2 = incomplete / not ready, 3 = STOP; submit jobs/app_pilot.slurm from analysis/mlts)
  select              MLTENV=mlts: $MLT/app_pilot/selection.{csv,json}, SELECT.txt, results/app_pilot/selection.csv
                      (an existing selection is kept; a recomputed one that differs = STOP)
  prep [--smoke] [--only a,b]
                      MLTENV=mlts: runs/<rxn>/{prep.json, mlip.hess, mlip_ts.xyz} + PREP.txt (--smoke: only the smallest
                      selected reaction, the smoke reaction)
  run [--only a,b]    MLTENV=d1: the ORCA OptTS + Freq of every selected reaction without result.json; SUMMARY.txt after
                      every finished reaction; then `report`
  smoke               MLTENV=d1: the smoke reaction with MaxIter 2 and no Freq -> $MLT/app_pilot/smoke/<rxn>/ (smoke.json)
                      + SMOKE.txt (input accepted, InHess read, SMD, SCF, parser output)
  report              MLTENV=d1 (numpy only): $MLT/app_pilot/{app_pilot_summary.json, SUMMARY.txt} and
                      results/app_pilot/{app_pilot_summary.json, SUMMARY.txt, results.csv} (exit 2 while incomplete)

Pre-registered use (user 11-05 (3)): A'' is acceptable as the deploy engine only if cost <= 1.5 core-h per reaction AND
convergence to the label TS >= 0.90; the summary gives both cost statistics (mean, median) against 1.5."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as C  # noqa: E402
import mlts_bridge as B  # noqa: E402

CODE_VERSION = "mlts-app-pilot-1"
MODEL = "FTpilot_AIMNet2_4"
FT2_DIR = C.MLT / "ft2" / MODEL / "label"                 # first pass (not label_recalc5)
FT2_RECS = FT2_DIR / "recs"
LIST = C.MLT / "pipeline" / "lists" / "f4_test.csv"
AP = C.MLT / "app_pilot"
RUNS, SMOKE_DIR, REGEN = AP / "runs", AP / "smoke", AP / "regen"
SEL_CSV, SEL_JSON = AP / "selection.csv", AP / "selection.json"
SUMMARY_TXT, SUMMARY_JSON, STOP_FILE = AP / "SUMMARY.txt", AP / "app_pilot_summary.json", AP / "STOP"
SELECT_TXT, PREP_TXT, SMOKE_TXT = AP / "SELECT.txt", AP / "PREP.txt", AP / "SMOKE.txt"
RES_DIR = C.RES / "app_pilot"
SEED, N_PER = 42, 10
HESS = "mlip.hess"
NPROCS = 8                                                 # D1 config_run nc / opt.PARTB_NPROCS (checked)
IMAG_TOL_CM = B.IMAG_TOL_CM                                # 50: modes below -50 cm-1 count (ts_geom convention)
MAX_STARTS = 2
RESERVE_H = float(os.environ.get("APP_RESERVE_H", "8"))
WALL_H = float(os.environ.get("MLT_WALL_H", "48"))
T_START = time.time()
MAX_ERR = 10
WATCH_S = 5.0
COST_MAX_CORE_H, CONV_MIN = 1.5, 0.90                      # user 11-05 (3), pre-registered
NOT_ATTEMPTED = ("no_start", "no_label", "not_covered")    # = ts_geom.NOT_ATTEMPTED
INHESS_LINES = ("InHess Read", f'InHessName "{HESS}"')
GEOM_WANT = ["%geom", "InHess Read", f'InHessName "{HESS}"', "Recalc_Hess 30", "Trust -0.1", "MaxIter 100", "end"]
KW_WANT = ["!", "OptTS", "Freq", "B3LYP", "RIJCOSX", "D3BJ", "def2-SVP", "def2/J", "CPCM(Water)"]
SEL_COLS = ["pick", "dipole", "pick_in_dipole", "rxn_id", "job_id", "n_atoms", "fold_T1", "charge", "formed_pairs_ts",
            "rec_status", "rec_converged", "rec_same_ts", "rec_dd_max", "rec_rmsd_heavy", "rec_n_imag", "rec_n_cycles",
            "rec_fail_class", "rec_has_pos", "rec_tight", "rec_recalc"]

RE_CYC = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")
RE_HESS = re.compile(r"^SCF HESSIAN\s*$", re.M)             # = g3l.RE_SCF_HESS (1 per Freq in the G3L pilot)
RE_HESS_LOOSE = re.compile(r"^\s*SCF HESSIAN\s*$", re.M)
RE_SCF_OK = re.compile(r"SCF CONVERGED AFTER\s+\d+\s+CYCLES")
RE_SCF_BAD = re.compile(r"SCF NOT CONVERGED|NOT CONVERGED AFTER")
RE_OUTFREQ = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", re.M)
RE_TIMING = re.compile(r"^(.{3,60}?)\s+\.\.\.\s+(\d+\.\d+)\s+sec", re.M)
RE_FSPE = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", re.M)
RE_MAXITER = re.compile(r"reached\s+the\s+maximum\s+number\s+of\s+(?:optimization\s+)?(?:cycles|iterations)", re.I)
RE_RECALC_LN = re.compile(r"recalc|exact hessian|hessian.*(?:recomput|re-?calcul)", re.I)
PARTB_REF = C.G3 / "O" / "units" / "B.1002979" / "optts" / "ts.out.gz"   # D1 block unchanged, 100 cycles (I_sources)
REF_TXT = AP / "REF_PARTB.txt"
RE_INHESS = re.compile(r"InHessName:\s*(\S+)\s+The file <([^>]+)> is opened as a \.hess file")   # = g3l.RE_INHESS
RE_RELAX = re.compile(r"ORCA GEOMETRY RELAXATION STEP")
RE_INHESS_SET = re.compile(r"^Initial Hessian\s+InHess\s+\.+\s*(.*?)\s*$", re.M)
RE_VER = re.compile(r"Program Version (\S+)")
CONV_MARK, FREQ_MARK = "THE OPTIMIZATION HAS CONVERGED", "VIBRATIONAL FREQUENCIES"
WATCH_EVENTS = (("cycle", re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")),
                ("scf_hessian", re.compile(r"^\s*SCF HESSIAN\s*$")),
                ("converged", re.compile(CONV_MARK)),
                ("final_energy", re.compile(r"FINAL ENERGY EVALUATION AT THE STATIONARY POINT")),
                ("freq", re.compile(FREQ_MARK)))


class StopAll(Exception):
    """a systematic problem (input builder, InHess not read): $MLT/app_pilot/STOP, no new reaction."""


# ================================================================ small helpers
def _rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _flt(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _stats(v):
    a = np.asarray([x for x in (_flt(y) for y in v) if x is not None], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median=float(np.median(a)), p90=float(np.percentile(a, 90)), mean=float(a.mean()),
                min=float(a.min()), max=float(a.max()), sum=float(a.sum()))


def _wbytes(path, data):
    return C._atomic(path, lambda t: Path(t).write_bytes(data))


def time_left_h():
    t0 = float(os.environ.get("MLT_JOB_T0") or T_START)
    return WALL_H - (time.time() - t0) / 3600.0


def write_stop(msg, smoke=False):
    """the pilot's STOP file (blocks new reactions); a smoke test only reports (its failure is read from SMOKE.txt)."""
    if not smoke and not STOP_FILE.is_file():
        C.write_text(STOP_FILE, f"{C.now()} {B.jobtag()} {msg}\n")
    print(f"[STOP] {msg}", file=sys.stderr, flush=True)


def xyz_text(syms, pos, title):
    return f"{len(syms)}\n{title}\n" + "".join(f"{s:<3s}{x:16.10f}{y:16.10f}{z:16.10f}\n"
                                              for s, (x, y, z) in zip(syms, np.asarray(pos, float)))


def csv_text(rows, cols):
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else (json.dumps(r[c]) if isinstance(r[c], (list, dict)) else r[c]))
                    for c in cols})
    return f.getvalue()


def read_selection():
    if not SEL_CSV.is_file():
        C.not_ready(f"{SEL_CSV} missing: run 'app_pilot.py select' first")
    with open(SEL_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("pick", "pick_in_dipole", "rxn_id", "n_atoms", "fold_T1", "charge"):
            r[k] = int(r[k])
    return rows


def smoke_row(sel):
    """the smoke reaction: the smallest selected reaction (n_atoms, then rxn_id)."""
    return min(sel, key=lambda r: (r["n_atoms"], r["rxn_id"]))


def parse_only(s):
    return {int(x) for x in str(s).split(",") if x.strip()} if s else None


# ================================================================ select (MLTENV=mlts)
def cmd_select():
    import pandas as pd
    import ts_geom as TG
    if not LIST.is_file():
        C.not_ready(f"{LIST} missing")
    if not FT2_RECS.is_dir():
        C.not_ready(f"{FT2_RECS} missing (FT-2 first pass of {MODEL} not run)")
    L = pd.read_csv(LIST)
    if "rxn_id" not in L:
        C.stop(f"{LIST}: no rxn_id column")
    ids = sorted({int(x) for x in L.rxn_id})
    T = C.targets()
    T = T[T.rxn_id.isin(set(ids))]
    miss = sorted(set(ids) - {int(x) for x in T.rxn_id})
    labs = {int(d["rxn_id"]): d for d in C.read_json(C.LABELS)}
    rows, models, starts, why_n = [], Counter(), Counter(), Counter()
    for r in T.itertuples():
        rid = int(r.rxn_id)
        d = labs.get(rid) or {}
        row = dict(rxn_id=rid, job_id=str(r.job_id), dipole=str(r.dipole), n_atoms=int(r.n_atoms),
                   fold_T1=int(r.fold_T1), charge=int(d.get("charge1") or 0) + int(d.get("charge2") or 0),
                   formed_pairs_ts=str(r.formed_pairs_ts))
        rec = _rj(FT2_RECS / f"{rid}.json")
        if rec is None:
            why_n["no_record"] += 1
            rows.append(dict(row, eligible=False, why="no_record"))
            continue
        models[str(rec.get("model"))] += 1
        starts[str(rec.get("start"))] += 1
        st = str(rec.get("status"))
        pos = rec.get("pos_A")
        has_pos = isinstance(pos, list) and len(pos) == int(r.n_atoms)
        ncyc = _int(rec.get("n_cycles")) or 0
        has_d = isinstance(rec.get("d_opt"), list)
        why = ("not_attempted" if st in NOT_ATTEMPTED else "no_mlip_cycle" if ncyc < 1 else
               "no_final_geometry" if not (has_pos or has_d) else "")
        why_n[why or "eligible"] += 1
        rows.append(dict(row, eligible=not why, why=why, rec_status=st, rec_converged=rec.get("converged"),
                         rec_same_ts=rec.get("same_ts"), rec_dd_max=rec.get("dd_max"), rec_rmsd_heavy=rec.get("rmsd_heavy"),
                         rec_n_imag=rec.get("n_imag"), rec_n_cycles=ncyc, rec_fail_class=TG.fail_class(rec) or "pass",
                         rec_has_pos=has_pos, rec_tight=rec.get("tight"), rec_recalc=rec.get("recalc")))
    if set(models) != {MODEL}:
        C.stop(f"FT-2 records in {FT2_RECS}: models {dict(models)}, expected only {MODEL}")
    if set(starts) != {"label"}:
        C.stop(f"FT-2 records in {FT2_RECS}: starts {dict(starts)}, expected only 'label'")
    rng = np.random.default_rng(SEED)
    picks, pools = [], {}
    for dn in C.DIPOLES:
        pool = sorted(x["rxn_id"] for x in rows if x["eligible"] and x["dipole"] == dn)
        pools[dn] = len(pool)
        k = min(N_PER, len(pool))
        got = [int(v) for v in rng.choice(np.asarray(pool, dtype=np.int64), size=k, replace=False)] if k else []
        picks += [(dn, j + 1, v) for j, v in enumerate(got)]
    byid = {x["rxn_id"]: x for x in rows}
    sel = [dict(byid[v], pick=i + 1, pick_in_dipole=j) for i, (dn, j, v) in enumerate(picks)]
    if not sel:
        C.stop("no eligible reaction")
    new_ids = [x["rxn_id"] for x in sel]
    if SEL_CSV.is_file():
        old = [r["rxn_id"] for r in read_selection()]
        if old != new_ids:
            C.stop(f"{SEL_CSV} exists with another list ({len(old)} reactions; recomputed {len(new_ids)}, "
                   f"{len(set(old) ^ set(new_ids))} differ): kept, nothing re-planned")
        print(f"[select] {SEL_CSV} up to date ({len(old)} reactions)", flush=True)
    elig = [x for x in rows if x["eligible"]]
    meta = dict(phase="A'' pilot selection", code=CODE_VERSION, written=C.now(), job=B.jobtag(), list=str(LIST),
                list_sha256=C.sha256(LIST), ft2_recs=str(FT2_RECS), model=MODEL, seed=SEED, n_per_dipole=N_PER,
                rule=f"eligible = first-pass FT-2 label record, status not in {list(NOT_ATTEMPTED)}, >= 1 MLIP OptTS "
                     f"cycle, final MLIP geometry (pos_A or d_opt); no FT-2 pass/fail filter. numpy.random.default_rng("
                     f"{SEED}); per dipole in mlt_common.DIPOLES order rng.choice(sorted eligible rxn_ids, size="
                     f"{N_PER}, replace=False)",
                n_list=len(ids), n_list_not_in_targets=len(miss), n_targets=int(len(T)), eligibility=dict(why_n),
                models=dict(models), starts=dict(starts), pool_per_dipole=pools,
                n_selected=len(sel), per_dipole={dn: [x["rxn_id"] for x in sel if x["dipole"] == dn] for dn in C.DIPOLES},
                pool_fail_class=dict(Counter(x["rec_fail_class"] for x in elig)),
                selected_fail_class=dict(Counter(x["rec_fail_class"] for x in sel)),
                pool_same_ts_frac=float(np.mean([bool(x["rec_same_ts"]) for x in elig])) if elig else None,
                selected_same_ts_frac=float(np.mean([bool(x["rec_same_ts"]) for x in sel])),
                n_charged_selected=sum(x["charge"] != 0 for x in sel),
                n_selected_without_pos=sum(not x["rec_has_pos"] for x in sel),
                selected_n_atoms=_stats([x["n_atoms"] for x in sel]))
    text = csv_text(sel, SEL_COLS)
    if not SEL_CSV.is_file():
        C.write_text(SEL_CSV, text)
    C.write_text(RES_DIR / "selection.csv", SEL_CSV.read_text())
    C.write_json(SEL_JSON, meta)
    C.write_json(RES_DIR / "selection.json", meta)
    lines = [f"A'' pilot selection ({C.now()}, job {B.jobtag()})", f"list {LIST} n={len(ids)}, in targets {len(T)}",
             f"eligibility {dict(why_n)}", f"models {dict(models)} starts {dict(starts)}",
             f"pool per dipole {pools}", f"selected {len(sel)}; without pos_A {meta['n_selected_without_pos']}; "
             f"charged {meta['n_charged_selected']}; n_atoms {meta['selected_n_atoms']}",
             f"fail class pool {meta['pool_fail_class']}", f"fail class selected {meta['selected_fail_class']}",
             f"MLIP same_ts pool {meta['pool_same_ts_frac']} selected {meta['selected_same_ts_frac']}",
             f"recalc/tight of picks {dict(Counter((str(x['rec_recalc']), str(x['rec_tight'])) for x in sel))}"]
    lines += [f"  {dn}: {meta['per_dipole'][dn]}" for dn in C.DIPOLES]
    lines.append(f"smoke reaction {smoke_row(sel)['rxn_id']} (n_atoms {smoke_row(sel)['n_atoms']})")
    C.write_text(SELECT_TXT, "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


# ================================================================ prep (MLTENV=mlts): MLIP TS + MLIP Hessian
def ft2_run_specs():
    out = []
    for p in sorted(FT2_DIR.glob("run_*.json")):
        s = (_rj(p) or {}).get("spec")
        if isinstance(s, dict):
            out.append(s)
    return out


def regen_one(rid, rec, spec, state):
    """the record has no pos_A: the same MLIP OptTS (ts_geom.evaluate_one, label start, record tight / recalc)."""
    import ts_geom as TG
    if state.get("server") is None:
        state["server"] = B.start_server(spec)
    rows = TG.target_rows([rid])
    wr = B.local_tmp() / f"app_regen_{B.jobtag()}_{os.getpid()}"
    wr.mkdir(parents=True, exist_ok=True)
    try:
        r = TG.evaluate_one(rid, rows[rid], spec, "label", state["server"].socket, REGEN, wr, None,
                            tight=rec.get("tight") or False, recalc=rec.get("recalc"))
    finally:
        shutil.rmtree(wr, ignore_errors=True)
    C.write_json(REGEN / "recs" / f"{rid}.json", r)
    if not r.get("pos_A"):
        raise RuntimeError(f"regeneration of {rid} gave no geometry (status {r.get('status')})")
    return r


def prep_one(r, spec, state):
    rid = r["rxn_id"]
    d = RUNS / str(rid)
    pj = d / "prep.json"
    old = _rj(pj)
    if old and (not old.get("hess_ok") or ((d / HESS).is_file() and (d / "mlip_ts.xyz").is_file())):
        return old
    rec = _rj(FT2_RECS / f"{rid}.json")
    if rec is None:
        C.stop(f"{rid}: FT-2 record vanished")
    ls, lp = C.read_xyz(C.label_ts(rid))
    pairs = C.formed_pairs(r["formed_pairs_ts"])
    regen = None
    pos = rec.get("pos_A")
    if isinstance(pos, list) and len(pos) == len(ls):
        pos, src = np.asarray(pos, float), "ft2_record_pos_A"
    else:
        regen = regen_one(rid, rec, spec, state)
        pos, src = np.asarray(regen["pos_A"], float), "regenerated_mlip_optts"
    if pos.shape != (len(ls), 3):
        raise RuntimeError(f"{rid}: MLIP geometry shape {pos.shape}")
    d_now = B.dists(pos, pairs)
    d_rec = rec.get("d_opt")
    dev = (float(np.max(np.abs(np.subtract(d_now, d_rec)))) if isinstance(d_rec, list) and len(d_rec) == len(d_now)
           else None)
    t0 = time.time()
    H = B.hessian(spec, ls, pos)                                  # Eh/bohr^2, symmetrised, charge 0 (as the bridge)
    th = time.time() - t0
    ok = bool(np.isfinite(H).all())
    hs = B.hessian_summary(H, ls, pos)
    d.mkdir(parents=True, exist_ok=True)
    if ok:
        B.write_orca_hess(C.assert_write_ok(d / HESS), ls, pos, H)
    C.write_text(d / "mlip_ts.xyz", xyz_text(ls, pos, f"MLIP TS {MODEL} label start rxn {rid} ({src})"))
    ev = B.evaluator(spec)
    p = dict(rxn_id=rid, job_id=r["job_id"], dipole=r["dipole"], n_atoms=r["n_atoms"], charge=r["charge"],
             pairs=[list(x) for x in pairs], source=src, regen=None if regen is None else {
                 k: regen.get(k) for k in ("status", "converged", "n_cycles", "d_opt", "dd_max", "same_ts", "wall_s")},
             d_opt_record=d_rec, d_opt_now=d_now, d_opt_max_dev_A=dev, hess_ok=ok, hess_s=th,
             hess_charge=0, hessian_method=getattr(ev, "hessian_method", type(ev).__name__), device=B.device_info(),
             mlip_n_imag_now=hs.get("n_imag"), mlip_imag_cm_now=hs.get("imag_cm"), mlip_freqs_low_now=hs.get("freqs_low"),
             rec_n_imag=rec.get("n_imag"), rec_imag_cm=rec.get("imag_cm"),
             hess_sha256=C.sha256(d / HESS) if ok else None, spec={k: v for k, v in spec.items()},
             mlip_stage=dict(status=rec.get("status"), converged=rec.get("converged"), n_cycles=rec.get("n_cycles"),
                             n_ext_calls=rec.get("n_ext_calls"), n_hess_calls=rec.get("n_hess_calls"),
                             wall_s=rec.get("wall_s"), opt_wall_s=rec.get("opt_wall_s"), gpu_s=rec.get("gpu_s"),
                             tight=rec.get("tight"), recalc=rec.get("recalc"), same_ts=rec.get("same_ts"),
                             dd_max=rec.get("dd_max"), rmsd_heavy=rec.get("rmsd_heavy"), host=rec.get("host"),
                             job=rec.get("job")),
             host=socket.gethostname(), job=B.jobtag(), written=C.now(), code=CODE_VERSION)
    C.write_json(pj, p)
    print(f"[prep] {rid} {src} dev={dev} hess {th:.1f}s ok={ok} nimag(mlip now/rec)={hs.get('n_imag')}/"
          f"{rec.get('n_imag')}", flush=True)
    return p


def prep_line(p):
    dv = p.get("d_opt_max_dev_A")
    return (f"{p['rxn_id']} {p['dipole'][:16]:16s} N={p['n_atoms']:3d} {p['source']:24s} dev={'-' if dv is None else f'{dv:.2e}'} "
            f"hess={p.get('hess_s', 0):.1f}s ok={p.get('hess_ok')} nimag now/rec={p.get('mlip_n_imag_now')}/{p.get('rec_n_imag')}")


def cmd_prep(only=None, smoke=False):
    sel = read_selection()
    todo = [smoke_row(sel)] if smoke else [r for r in sel if not only or r["rxn_id"] in only]
    spec = B.resolve_spec(MODEL)
    w = spec.get("weights")
    if not w or not Path(os.path.expandvars(str(w))).is_file():
        C.stop(f"{MODEL}: weights {w} missing")
    ws = {str(s.get("weights")) for s in ft2_run_specs() if s.get("weights")}
    if ws and ws != {str(w)}:
        C.stop(f"FT-2 first pass ran with weights {sorted(ws)}, the model card now gives {w}")
    try:
        import torch
        torch.set_num_threads(max(1, min(16, _cpus())))
    except Exception:                                          # noqa: BLE001
        pass
    state = {}
    t0 = time.time()
    try:
        for r in todo:
            prep_one(r, spec, state)
    finally:
        if state.get("server") is not None:
            state["server"].shutdown()
    ps = [p for p in (_rj(RUNS / str(r["rxn_id"]) / "prep.json") for r in sel) if p]
    devs = [p.get("d_opt_max_dev_A") for p in ps]
    agree = [p.get("mlip_n_imag_now") == p.get("rec_n_imag") for p in ps]
    head = [f"A'' pilot prep ({C.now()}, job {B.jobtag()}): {len(ps)}/{len(sel)} prepared, this step {time.time() - t0:.0f} s",
            f"model {MODEL} weights {w} (FT-2 run specs {sorted(ws) or 'none recorded'})",
            f"sources {dict(Counter(p['source'] for p in ps))}; max |d - d_opt(rec)| {_stats(devs).get('max')} A; "
            f"MLIP n_imag now == record {sum(agree)}/{len(ps)}; hess s {_stats([p.get('hess_s') for p in ps])}",
            f"device {ps[0].get('device') if ps else None}; method {ps[0].get('hessian_method') if ps else None}"]
    C.write_text(PREP_TXT, "\n".join(head + [prep_line(p) for p in ps]) + "\n")
    print("\n".join(head), flush=True)
    bad = [p["rxn_id"] for p in ps if not p.get("hess_ok")]
    if bad:
        print(f"[prep] non-finite MLIP Hessian: {bad} (closed as no_mlip_hessian in run)", flush=True)
    return 0


# ================================================================ ORCA input (MLTENV=d1)
def d1_modules():
    import gen_dft as GD
    D, O, G3L = GD.d1_g3()                                     # D1G3_SCRATCH shadow under $MLT/dft/g2 (gen_dft G-2run)
    if int(O.PARTB_NPROCS) != NPROCS:
        raise StopAll(f"opt.PARTB_NPROCS {O.PARTB_NPROCS} != {NPROCS}")
    return GD, D, O, G3L


def app_text(D, O, G3L, syms, xyz, q, pairs, smoke=False):
    """the D1 OptTS + Freq input (opt.partb_text) with 'Calc_Hess true' -> InHess Read / InHessName "mlip.hess";
    smoke: no Freq, MaxIter 2."""
    syms, xyz = list(syms), np.asarray(xyz, float)
    base = O.partb_text(D, syms, xyz, int(q), 1, [tuple(int(i) for i in p) for p in pairs], O.PARTB_NPROCS)
    lines = base.split("\n")
    ic = [k for k, ln in enumerate(lines) if ln.strip().lower() == "calc_hess true"]
    if len(ic) != 1 or lines[ic[0] - 1].strip().lower() != "%geom":
        raise StopAll("D1 OptTS input: no single '%geom / Calc_Hess true' block")
    k = ic[0]
    new = lines[:k] + list(INHESS_LINES) + lines[k + 1:]
    blk = [ln.strip() for ln in new[k - 1:k - 1 + len(GEOM_WANT)]]
    if blk != GEOM_WANT:
        raise StopAll(f"A'' %geom block {blk} != {GEOM_WANT}")
    if new[0].split() != KW_WANT:
        raise StopAll(f"A'' keyword line {new[0]!r} != {' '.join(KW_WANT)}")
    low = [ln.strip().lower() for ln in new]
    if "%cpcm" not in low or "smd true" not in low or 'smdsolvent "water"' not in low:
        raise StopAll("A'' input: %cpcm smd true / SMDsolvent \"water\" missing")
    txt = "\n".join(new)
    if not re.search(rf"%pal\s+nprocs\s+{NPROCS}\b", txt) or not re.search(r"%\s*maxcore\s+3000\b", txt):
        raise StopAll("A'' input: %pal nprocs 8 / %maxcore 3000 missing")
    if any(ln.startswith("calc_hess") for ln in low) or sum(ln == "inhess read" for ln in low) != 1:
        raise StopAll("A'' input: Calc_Hess left or InHess Read not exactly once")
    try:
        g = G3L.g3l_text(D, syms, xyz, int(q), [list(p) for p in pairs])[0]
    except ValueError as e:
        raise StopAll(f"g3l.g3l_text: {e}")
    want = g.replace(f'InHessName "{G3L.HESS_NAME}"', f'InHessName "{HESS}"\n{lines[k + 1]}', 1)
    if txt != want:
        raise StopAll("A'' input differs from the G3L (G-2) input by more than InHessName + Recalc_Hess 30")
    if smoke:
        new[0] = new[0].replace(" Freq", "", 1)
        new = ["MaxIter 2" if ln.strip() == "MaxIter 100" else ln for ln in new]
        txt = "\n".join(new)
    return txt


# ================================================================ output watcher / parser
class Watch:
    """polls the growing ORCA output every WATCH_S s: (event, seconds since start, detail) for cycles, Hessians,
    convergence, final energy, frequencies."""

    def __init__(self, path, t0):
        self.path, self.t0, self.off, self.buf, self.ev = Path(path), t0, 0, "", []
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.th.start()
        return self

    def _scan(self, final=False):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.off)
                data = f.read()
        except OSError:
            return
        self.off += len(data)
        self.buf += data.decode("utf-8", "replace")
        parts = self.buf.split("\n")
        self.buf = "" if final else parts.pop()
        t = round(time.time() - self.t0, 1)
        for ln in parts:
            for name, rx in WATCH_EVENTS:
                m = rx.search(ln)
                if m:
                    self.ev.append([name, t, m.group(1) if m.groups() else None])

    def _loop(self):
        while not self.stop.wait(WATCH_S):
            try:
                self._scan()
            except Exception:                                  # noqa: BLE001  (timing only)
                pass

    def finish(self):
        self.stop.set()
        self.th.join(timeout=60)
        try:
            self._scan(final=True)
        except Exception:                                      # noqa: BLE001
            pass
        return self.ev


def step_walls(ev, total):
    """approximate walls (+- WATCH_S) from the watcher events of one ORCA run."""
    cyc = [t for n, t, _ in ev if n == "cycle"]
    t_conv = next((t for n, t, _ in ev if n == "converged"), None)
    end_opt = t_conv if t_conv is not None else total
    per = [round(b - a, 1) for a, b in zip(cyc, cyc[1:] + [end_opt])] if cyc else []
    allt = sorted(t for _, t, _ in ev)
    hess = []
    for n, t, _ in ev:
        if n == "scf_hessian":
            nxt = next((u for u in allt if u > t), total)
            hess.append(dict(t_s=t, wall_s=round(nxt - t, 1),
                             phase="freq" if (t_conv is not None and t >= t_conv) else "recalc"))
    return dict(poll_s=WATCH_S, setup_s=cyc[0] if cyc else None, per_cycle_s=per, opt_phase_s=end_opt,
                freq_phase_s=round(total - t_conv, 1) if t_conv is not None else None, hessians=hess, total_s=round(total, 1))


def hess_cycles(txt, hpos):
    """the optimisation cycle each SCF Hessian section belongs to (0 = before cycle 1)."""
    cp = [(m.start(), int(m.group(1))) for m in RE_CYC.finditer(txt)]
    return [max((c for q, c in cp if q < p), default=0) for p in hpos]


def recalc_lines(txt, n=8):
    out = []
    for ln in txt.splitlines():
        s = ln.strip()
        if RE_RECALC_LN.search(s) and s not in out:
            out.append(s[:160])
            if len(out) >= n:
                break
    return out


def ref_partb():
    """read only: how ORCA 6.1.1 ran the UNCHANGED D1 block (Calc_Hess true, Recalc_Hess 30) in the d1_g3 Part B OptTS
    B.1002979 (100 cycles, not converged): its optimisation settings and the cycles with an SCF Hessian ->
    $MLT/app_pilot/REF_PARTB.txt + dict (reference for reading this pilot's Recalc_Hess counts)."""
    if not PARTB_REF.is_file():
        return dict(path=str(PARTB_REF), present=False)
    import gzip
    with gzip.open(PARTB_REF, "rt", errors="replace") as f:
        txt = f.read()
    hpos = [m.start() for m in RE_HESS.finditer(txt)] or [m.start() for m in RE_HESS_LOOSE.finditer(txt)]
    k = txt.find("Geometry optimization settings")
    blk = [ln.rstrip() for ln in (txt[k:k + 3000].splitlines()[:8] if k >= 0 else [])]
    R = dict(path=str(PARTB_REF), present=True, n_cycles=len(RE_CYC.findall(txt)), converged=CONV_MARK in txt,
             n_scf_hessian=len(hpos), hessian_at_cycles=hess_cycles(txt, hpos), settings=blk,
             recalc_lines=recalc_lines(txt), max_iter_lines=[m.group(0) for m in RE_MAXITER.finditer(txt)][:2])
    L = [f"reference (read only): {PARTB_REF}", f"cycles {R['n_cycles']} converged {R['converged']}",
         f"SCF HESSIAN sections {R['n_scf_hessian']} at cycles {R['hessian_at_cycles']} (0 = before cycle 1)",
         f"max-iter lines {R['max_iter_lines']}", "settings:"] + blk + ["lines with recalc / exact hessian:"] + R["recalc_lines"]
    C.write_text(REF_TXT, "\n".join(L) + "\n")
    return R


def parse_out(txt, work, syms, OD=None, maxiter=None):
    """everything the pilot needs from one ORCA OptTS (+ Freq) output and its work folder."""
    work = Path(work)
    term = "ORCA TERMINATED NORMALLY" in txt
    cpos = txt.find(CONV_MARK)
    conv = cpos >= 0
    cyc = [int(x) for x in RE_CYC.findall(txt)]
    hs_s = [m.start() for m in RE_HESS.finditer(txt)]
    hs_l = [m.start() for m in RE_HESS_LOOSE.finditer(txt)]
    hpos = hs_s if (hs_s or not hs_l) else hs_l
    n_rec = sum(p < cpos for p in hpos) if conv else len(hpos)
    n_frq = sum(p > cpos for p in hpos) if conv else 0
    tail = txt[cpos:] if conv else ""
    freq_done = FREQ_MARK in tail
    freqs, fsrc = [], None
    hp = work / "ts.hess"
    if freq_done and hp.is_file():
        try:
            freqs, fsrc = [float(f) for f in B.hess_freqs(hp)], "ts.hess"
        except Exception:                                      # noqa: BLE001
            freqs = []
    f_out = [float(v) for v in RE_OUTFREQ.findall(tail.rsplit(FREQ_MARK, 1)[-1])] if freq_done else []
    if freq_done and not freqs and f_out:
        freqs, fsrc = f_out, "output"
    fin, gsrc = None, None
    xp = work / "ts.xyz"
    if xp.is_file():
        s2, p2 = C.read_xyz(xp)
        if list(s2) != list(syms):
            raise RuntimeError("ORCA final geometry has a different atom order")
        fin, gsrc = np.asarray(p2, float), "ts.xyz"
    elif cyc:
        lc = B.last_coords(txt)
        if lc is not None and len(lc) == len(syms):
            fin, gsrc = np.asarray(lc, float), "output"
    names = [m.group(2) for m in RE_INHESS.finditer(txt)]
    relax = len(RE_RELAX.findall(txt))
    read = bool(names) and names[0] == HESS
    k = txt.find("Geometry optimization settings")
    blk = txt[k:k + 8000].splitlines()[:45] if k >= 0 else []
    lc_ = OD.level_checks(txt) if OD is not None else {}
    lok = OD.level_ok("L0", lc_) if (OD is not None and hasattr(OD, "level_ok")) else {}
    tm = ({m.group(1).strip(): float(m.group(2)) for m in RE_TIMING.finditer(txt.rsplit("Timings for individual modules", 1)[-1])}
          if "Timings for individual modules" in txt else {})
    e = [float(x) for x in RE_FSPE.findall(txt)]
    vib = sorted(f for f in freqs if abs(f) > 1e-6)
    ver = RE_VER.search(txt)
    return dict(
        terminated=term, converged=conv,
        max_iter_hit=bool(RE_MAXITER.search(txt)) or bool(not conv and maxiter and len(cyc) >= int(maxiter)),
        n_cycles=len(cyc), last_cycle=max(cyc) if cyc else 0, n_gradients=txt.count("CARTESIAN GRADIENT"),
        n_scf_converged=len(RE_SCF_OK.findall(txt)), n_scf_not_converged=len(RE_SCF_BAD.findall(txt)),
        n_hess_total=len(hpos), n_hess_recalc=int(n_rec), n_hess_freq=int(n_frq), hess_at_cycles=hess_cycles(txt, hpos),
        recalc_lines=recalc_lines(txt), n_hess_strict=len(hs_s),
        n_hess_loose=len(hs_l), freq_done=freq_done, freqs_source=fsrc, n_freqs=len(freqs),
        n_freqs_output=len(f_out), n_imag=int(sum(f < -IMAG_TOL_CM for f in freqs)) if freqs else None,
        n_neg=int(sum(f < 0 for f in freqs)) if freqs else None, imag_cm=sorted(f for f in freqs if f < 0),
        freqs_low=[round(f, 2) for f in vib[:8]], pos_final=fin, geometry_source=gsrc,
        inhess=dict(read=read, names=names[:4], n_reads=len(names), n_relax_steps=relax,
                    initial_hessian_setting=RE_INHESS_SET.findall(txt)[:1], not_read=relax >= 1 and not read),
        hess_settings=[ln.strip() for ln in blk if re.search(r"hess", ln, re.I)][:12], geom_settings_block=blk,
        level=lc_, level_ok=lok, module_timings_s=tm, orca_hours=OD.run_hours(txt) if OD else None,
        orca_version=ver.group(1) if ver else None, n_energies=len(e), e_last_eh=e[-1] if e else None,
        errors=[ln.strip() for ln in txt.splitlines() if re.search(r"ERROR|aborting|Abort", ln)][:6],
        out_tail="\n".join(txt.splitlines()[-15:]))


# ================================================================ one reaction (MLTENV=d1)
def result_path(rid, smoke):
    return (SMOKE_DIR / str(rid) / "smoke.json") if smoke else (RUNS / str(rid) / "result.json")


def run_one(r, GD, D, O, G3L, local, smoke=False):
    rid = int(r["rxn_id"])
    dest = (SMOKE_DIR if smoke else RUNS) / str(rid)
    rp = result_path(rid, smoke)
    if rp.is_file():
        return "skip", 0.0
    pdir = RUNS / str(rid)
    prep = _rj(pdir / "prep.json")
    base = dict(rxn_id=rid, job_id=r["job_id"], dipole=r["dipole"], n_atoms=r["n_atoms"], fold_T1=r["fold_T1"],
                charge=r["charge"], mult=1, pick=r["pick"], model=MODEL, start="FT-2 first-pass label-start MLIP TS",
                smoke=bool(smoke), nprocs=NPROCS, code=CODE_VERSION, slurm_job=B.jobtag(), host=socket.gethostname(),
                rec=dict(status=r.get("rec_status"), same_ts=r.get("rec_same_ts"), dd_max=_flt(r.get("rec_dd_max")),
                         n_imag=_int(r.get("rec_n_imag")), fail_class=r.get("rec_fail_class")))
    if prep is None:
        raise RuntimeError(f"{pdir}/prep.json missing: run 'app_pilot.py prep' first")
    base["prep"] = {k: prep.get(k) for k in ("source", "d_opt_max_dev_A", "hess_s", "mlip_n_imag_now", "rec_n_imag",
                                             "hess_sha256", "mlip_stage")}
    if not prep.get("hess_ok"):
        if not smoke:
            C.write_json(rp, dict(base, status="error", reason="no_mlip_hessian", core_h=0.0, finished=C.now()))
        return "no_mlip_hessian", 0.0
    hp = pdir / HESS
    if C.sha256(hp) != prep.get("hess_sha256"):
        raise RuntimeError(f"{hp} changed since prep")
    syms, pos = C.read_xyz(pdir / "mlip_ts.xyz")
    pos = np.asarray(pos, float)
    ls, lp = C.read_xyz(C.label_ts(rid))
    lp = np.asarray(lp, float)
    if list(syms) != list(ls):
        raise RuntimeError("MLIP TS atom order differs from the label TS")
    hh = B.OI.read_hess(hp)
    hdev = float(np.abs(np.asarray(hh["pos_bohr"], float) * C.BOHR2A - pos).max())
    if list(hh["symbols"]) != list(syms) or hdev > 1e-5:
        raise RuntimeError(f"{hp} does not belong to mlip_ts.xyz (max |dx| {hdev:.2e} A)")
    pairs_s = GD.label_pairs(r["formed_pairs_ts"])             # sorted (G3L / Part B order)
    text = app_text(D, O, G3L, syms, pos, r["charge"], pairs_s, smoke)
    if not smoke:
        sp = dest / "starts.json"
        starts = _rj(sp) or []
        if len(starts) >= MAX_STARTS:
            C.write_json(rp, dict(base, status="wall_clipped", reason=f"started {len(starts)} times without finishing",
                                  starts=starts, core_h=0.0, finished=C.now()))
            return "wall_clipped", 0.0
        C.write_json(sp, starts + [dict(job=B.jobtag(), host=socket.gethostname(), started=C.now())])
    work = local / str(rid)
    runs, txt, ev = [], "", []
    t_all = time.time()
    try:
        for k in (1, 2):
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            (work / "ts.inp").write_text(text)
            shutil.copyfile(hp, work / HESS)
            t0 = time.time()
            with open(work / "ts.out", "w") as fo, open(work / "ts.err", "w") as fe:
                pr = subprocess.Popen([str(D["orca"]), "ts.inp"], cwd=str(work), stdout=fo, stderr=fe, env=D["env"])
                W = Watch(work / "ts.out", t0).start()
                rc = pr.wait()
            wall = time.time() - t0
            ev = W.finish()
            txt = (work / "ts.out").read_text(errors="replace")
            term = "ORCA TERMINATED NORMALLY" in txt
            infra = (not term) and (rc < 0 or not txt.strip()
                                    or ("SCF ITERATIONS" not in txt and "GEOMETRY OPTIMIZATION CYCLE" not in txt))
            runs.append(dict(run=k, rc=rc, wall_s=wall, core_h=wall * NPROCS / 3600, infra=infra,
                             host=socket.gethostname(), started=C.now()))
            if not infra:
                break
            C.write_text(dest / f"ts.infra{k}.out", "\n".join(txt.splitlines()[-200:]) + "\n")
        dest.mkdir(parents=True, exist_ok=True)
        for fn in ("ts.inp", "ts.out", "ts.err", "ts.xyz", "ts.hess", "ts_trj.xyz"):      # kept before parsing
            f = work / fn
            if f.is_file() and not (fn == "ts.err" and f.stat().st_size == 0):
                _wbytes(dest / fn, f.read_bytes())
        core = float(sum(x["core_h"] for x in runs))
        try:
            res, P, status = post_one(r, base, text, txt, work, runs, ev, syms, pos, ls, lp, D, smoke, t_all, core)
        except StopAll:
            raise
        except Exception as e:                                 # noqa: BLE001  (the ORCA run and its cost are kept)
            print(f"[app_pilot] post-processing error {rid}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}",
                  file=sys.stderr, flush=True)
            P, status = dict(inhess=dict(not_read=False)), "error"
            res = dict(base, status="error", reason="post_error", error=f"{type(e).__name__}: {e}"[:500],
                       runs=runs, core_h=core, wall_s=float(sum(x["wall_s"] for x in runs)),
                       header=text.split("*xyz")[0], events=ev, finished=C.now())
        if not smoke:
            res.pop("geom_settings_block", None)
        C.write_json(rp, res)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if P["inhess"]["not_read"]:
        raise StopAll(f"{rid}: ORCA did not read {HESS} (InHess)")
    return status, core


def post_one(r, base, text, txt, work, runs, ev, syms, pos, ls, lp, D, smoke, t_all, core):
    """parse one finished ORCA run and compare it with the label TS -> (result dict, parse dict, status)."""
    mi = re.search(r"^\s*MaxIter\s+(\d+)\s*$", text, re.M)
    P = parse_out(txt, work, syms, D["OD"], maxiter=int(mi.group(1)) if mi else None)
    pairs_t = C.formed_pairs(r["formed_pairs_ts"])             # targets order (ts_geom metrics)
    d_lab = B.dists(lp, pairs_t)
    d_st = B.dists(pos, pairs_t)
    s0, r0, f0 = B.same_ts((ls, lp), (ls, pos), d_lab, d_st)
    m = dict(pairs=[list(p) for p in pairs_t], d_label=d_lab, d_start=d_st, start_same_ts=bool(s0),
             start_rmsd_heavy=_flt(r0), start_dform_sorted=f0,
             start_dd_max=float(np.max(np.abs(np.subtract(d_st, d_lab)))))
    fin = P.pop("pos_final")
    if fin is not None:
        d_fin = B.dists(fin, pairs_t)
        dd = [float(a - b) for a, b in zip(d_fin, d_lab)]
        same, rm, df = B.same_ts((ls, lp), (ls, fin), d_lab, d_fin)
        m.update(d_final=d_fin, dd=dd, dd_max=float(np.max(np.abs(dd))), same_ts=bool(same),
                 rmsd_heavy=_flt(rm), dform_sorted=df, rmsd_direct=B.heavy_rmsd_direct(ls, lp, fin),
                 moved_rmsd_direct=B.heavy_rmsd_direct(ls, pos, fin), pos_final_A=np.round(fin, 6).tolist())
    else:
        m.update(d_final=None, dd=None, dd_max=None, same_ts=False, rmsd_heavy=None, dform_sorted=None)
    ctl = bool(P["converged"] and m["same_ts"])
    if runs[-1]["infra"]:
        status, reason = "error", "pipeline_infra"
    elif P["inhess"]["not_read"]:
        status, reason = "error", "inhess_not_read"
    elif smoke:
        status, reason = "smoke", ""
    elif not P["converged"]:
        status = "not_converged"
        reason = "max_iter" if P["max_iter_hit"] else ("optts_not_converged" if P["terminated"] else "orca_not_terminated")
    elif not (P["terminated"] and P["freq_done"]):
        status, reason = "no_freq", "orca_not_terminated" if not P["terminated"] else "freq_missing"
    else:
        status, reason = "ok", "converged_to_label" if ctl else "other_saddle"
    res = dict(base, status=status, reason=reason, converged_to_label=ctl, nimag1=P.get("n_imag") == 1,
               runs=runs, core_h=core, wall_s=float(sum(x["wall_s"] for x in runs)),
               wall_all_s=time.time() - t_all, header=text.split("*xyz")[0],
               walls=step_walls(ev, runs[-1]["wall_s"]), events=ev, **P, **m, finished=C.now())
    return res, P, status


# ================================================================ run / smoke (MLTENV=d1)
def cmd_run(only=None, smoke=False):
    if STOP_FILE.is_file():
        print(f"[STOP] {STOP_FILE} exists: {STOP_FILE.read_text().strip()}", file=sys.stderr, flush=True)
        return 3
    try:
        GD, D, O, G3L = d1_modules()
    except Exception as e:                                     # noqa: BLE001  (gen_dft.StopAll, our StopAll, imports)
        print(f"[STOP] D1 modules: {type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}", file=sys.stderr, flush=True)
        if smoke:
            C.write_text(SMOKE_TXT, f"SMOKE FAIL: D1 modules: {type(e).__name__}: {e}\n")
        return 3
    sel = read_selection()
    rows = [smoke_row(sel)] if smoke else [r for r in sel if not only or r["rxn_id"] in only]
    todo = sorted([r for r in rows if not result_path(r["rxn_id"], smoke).is_file()],
                  key=lambda r: (-r["n_atoms"], r["rxn_id"]))
    slots = 1 if smoke else max(1, min(len(todo) or 1, _cpus() // NPROCS))
    local = Path(os.environ.get("TMPDIR") or "/tmp") / f"app_pilot_{B.jobtag()}_{os.getpid()}"
    ctx = dict(mu=threading.Lock(), halt="", n_err=0, timed_out=False, it=iter(todo), running=set(), stats=Counter())
    print(f"[app_pilot {'smoke' if smoke else 'run'}] {len(rows)} reactions, {len(todo)} to do, {slots} slots x {NPROCS} "
          f"ranks, ORCA {D['orca']}, job {B.jobtag()} on {socket.gethostname()}, {time_left_h():.1f} h left", flush=True)

    def claim():
        with ctx["mu"]:
            if ctx["halt"]:
                return None
            if not smoke and time_left_h() < RESERVE_H:
                ctx["timed_out"] = True
                return None
            r = next(ctx["it"], None)
            if r is not None:
                ctx["running"].add(r["rxn_id"])
            return r

    def loop():
        while True:
            r = claim()
            if r is None:
                return
            try:
                st, ch = run_one(r, GD, D, O, G3L, local, smoke)
                with ctx["mu"]:
                    ctx["stats"][st] += 1
                print(f"[app_pilot] {r['rxn_id']} {r['dipole']} N={r['n_atoms']} {st} core-h {ch:.2f} {C.now()}", flush=True)
            except StopAll as e:
                with ctx["mu"]:
                    ctx["halt"] = ctx["halt"] or str(e)
                write_stop(str(e), smoke)
                if smoke:
                    C.write_text(SMOKE_TXT, f"SMOKE FAIL (StopAll): {e}\n")
            except Exception as e:                             # noqa: BLE001  (no record: retried on resubmission)
                print(f"[app_pilot] pipeline error {r['rxn_id']}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}",
                      file=sys.stderr, flush=True)
                with ctx["mu"]:
                    ctx["n_err"] += 1
                    ctx["stats"]["pipeline_error"] += 1
                    if ctx["n_err"] > MAX_ERR and not ctx["halt"]:
                        ctx["halt"] = f"more than {MAX_ERR} pipeline errors"
            finally:
                with ctx["mu"]:
                    ctx["running"].discard(r["rxn_id"])
                    if not smoke:
                        try:
                            write_summary(sel, running=sorted(ctx["running"]))
                        except Exception as e:                 # noqa: BLE001
                            print(f"[app_pilot] summary error: {e}", file=sys.stderr, flush=True)

    if not smoke and not (AP / "ref_partb.json").is_file():
        try:
            C.write_json(AP / "ref_partb.json", ref_partb())
        except Exception as e:                                 # noqa: BLE001  (reference only)
            print(f"[app_pilot] Part B reference not read: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    if not smoke and todo:
        with ctx["mu"]:
            write_summary(sel, running=[])
    if todo:
        local.mkdir(parents=True, exist_ok=True)
        try:
            with ThreadPoolExecutor(slots) as ex:
                for f in [ex.submit(loop) for _ in range(slots)]:
                    f.result()
        finally:
            shutil.rmtree(local, ignore_errors=True)
    print(f"[app_pilot] stats {dict(ctx['stats'])} halt {ctx['halt'] or '-'} timed_out {ctx['timed_out']}", flush=True)
    if smoke:
        return smoke_report(sel, D)
    rc = cmd_report()
    if ctx["halt"]:
        return 3
    return rc


def smoke_report(sel, D):
    r = smoke_row(sel)
    x = _rj(result_path(r["rxn_id"], True))
    if x is None:
        prev = SMOKE_TXT.read_text() if SMOKE_TXT.is_file() else ""
        C.write_text(SMOKE_TXT, prev + f"SMOKE FAIL: no smoke.json for {r['rxn_id']} (see the job log)\n")
        return 3
    lv, lok = x.get("level") or {}, x.get("level_ok") or {}
    chk = dict(inhess_read=bool((x.get("inhess") or {}).get("read")), smd_cds=bool(lv.get("smd_cds")),
               eps_78_355=lv.get("eps") == "78.3550", scf_converged=x.get("n_scf_converged", 0) >= 1 and
               x.get("n_scf_not_converged", 0) == 0, cycles=x.get("n_cycles", 0) >= 1,
               gradient=x.get("n_gradients", 0) >= 1,
               keyword_line=(lv.get("keyword_line") or "").split() == [t for t in KW_WANT if t != "Freq"],
               level_ok=all(v for v in lok.values() if v is not None), not_infra=x.get("status") != "error")
    ok = all(chk.values())
    p = _rj(RUNS / str(r["rxn_id"]) / "prep.json") or {}
    L = [f"SMOKE {'PASS' if ok else 'FAIL'}  rxn {r['rxn_id']} {r['dipole']} N={r['n_atoms']} charge {r['charge']} "
         f"({C.now()}, job {B.jobtag()})", f"checks {chk}",
         f"status {x.get('status')} terminated {x.get('terminated')} converged {x.get('converged')} max_iter_hit "
         f"{x.get('max_iter_hit')} cycles {x.get('n_cycles')} gradients {x.get('n_gradients')} scf ok/bad "
         f"{x.get('n_scf_converged')}/{x.get('n_scf_not_converged')} hess recalc/freq {x.get('n_hess_recalc')}/"
         f"{x.get('n_hess_freq')}", f"inhess {x.get('inhess')}",
         f"level {dict((k, lv.get(k)) for k in ('keyword_line', 'eps', 'smd_cds', 'rijcosx', 'vwn', 'd3'))}",
         f"level_ok {lok}", f"wall {x.get('wall_s', 0):.1f} s, core-h {x.get('core_h', 0):.3f}, runs "
         f"{[(y['run'], y['rc'], round(y['wall_s'], 1), y['infra']) for y in x.get('runs', [])]}",
         f"walls {dict((k, v) for k, v in (x.get('walls') or {}).items() if k != 'hessians')}",
         f"start: same_ts {x.get('start_same_ts')} rmsd {x.get('start_rmsd_heavy')} dd {x.get('start_dd_max')}; after 2 "
         f"cycles dd {x.get('dd_max')} moved {x.get('moved_rmsd_direct')}",
         f"prep: {prep_line(p) if p else 'none'}", f"errors {x.get('errors')}", "--- input header ---"]
    L += (x.get("header") or "").rstrip("\n").splitlines()
    L += ["--- ORCA geometry optimization settings (first lines) ---"] + list(x.get("geom_settings_block") or [])[:30]
    L += ["--- out tail ---"] + (x.get("out_tail") or "").splitlines()[-10:]
    C.write_text(SMOKE_TXT, "\n".join(L) + "\n")
    print("\n".join(L[:12]), flush=True)
    return 0 if ok else 3


# ================================================================ summary / report
def load_results(sel):
    out = {}
    for r in sel:
        x = _rj(result_path(r["rxn_id"], False))
        if x is not None:
            out[r["rxn_id"]] = x
    return out


def _frac(n, d):
    return (n / d) if d else None


def summarize(sel, res, running=()):
    N = len(sel)
    fin = [res[r["rxn_id"]] for r in sel if r["rxn_id"] in res]
    nf = len(fin)
    g = lambda k, xs=fin: [x.get(k) for x in xs]                                     # noqa: E731
    n_ctl = sum(bool(x.get("converged_to_label")) for x in fin)
    n_conv = sum(bool(x.get("converged")) for x in fin)
    n_same = sum(bool(x.get("same_ts")) for x in fin)
    n_im1 = sum(x.get("n_imag") == 1 for x in fin)
    conv = [x for x in fin if x.get("converged")]
    hw = [h for x in fin for h in ((x.get("walls") or {}).get("hessians") or [])]
    tkeys = sorted({k for x in fin for k in (x.get("module_timings_s") or {})})
    preps = [x.get("prep") or {} for x in fin]
    S = dict(
        phase="A'' pilot", decision="user 11-05 18:0x (2) / (3), results/DECISIONS.md", code=CODE_VERSION,
        written=C.now(), job=B.jobtag(), model=MODEL, start="FT-2 first-pass label-start MLIP TS ($MLT/ft2/"
        f"{MODEL}/label/recs)", list=str(LIST), seed=SEED, n_planned=N, n_finished=nf, n_running=len(running),
        running=list(running), complete=nf == N,
        input=dict(geom_block=GEOM_WANT, keyword_line=" ".join(KW_WANT[1:]), nprocs=NPROCS, maxcore_mb=3000,
                   change="D1 OptTS input (opt.partb_text) with 'Calc_Hess true' -> 'InHess Read' + 'InHessName "
                          "\"mlip.hess\"'; nothing else changed",
                   header_example=fin[0].get("header") if fin else None),
        status_counts=dict(Counter(x.get("status") for x in fin)),
        reason_counts=dict(Counter(f"{x.get('status')}:{x.get('reason')}" for x in fin)),
        dft_gradients=dict(_stats(g("n_cycles")), definition="ORCA optimisation cycles (one SCF + gradient each)"),
        dft_gradients_cartesian_blocks=_stats(g("n_gradients")),
        dft_hessians=dict(recalc=_stats(g("n_hess_recalc")), freq=_stats(g("n_hess_freq")),
                          recalc_total=int(sum(_int(v) or 0 for v in g("n_hess_recalc"))),
                          freq_total=int(sum(_int(v) or 0 for v in g("n_hess_freq"))),
                          recalc_counts=dict(Counter(str(v) for v in g("n_hess_recalc"))),
                          freq_counts=dict(Counter(str(v) for v in g("n_hess_freq"))),
                          recalc_at_cycles=dict(Counter(str(c) for x in fin for c in
                                                        (x.get("hess_at_cycles") or [])[:_int(x.get("n_hess_recalc")) or 0])),
                          definition="'SCF HESSIAN' sections before 'THE OPTIMIZATION HAS CONVERGED' = Recalc_Hess "
                                     "(every 30 cycles), after it = the final Freq"),
        core_h=dict(_stats(g("core_h")), definition=f"sum over the reaction's ORCA runs of wall x {NPROCS} cores / 3600 "
                                                    "(OptTS + Freq, incl. an infra rerun); MLIP stage not included"),
        converged_to_label=dict(n=n_ctl, frac_planned=_frac(n_ctl, N), frac_finished=_frac(n_ctl, nf),
                                rule="ORCA converged AND mlts_bridge.same_ts(final, label TS): mapped heavy RMSD <= "
                                     "0.10 A and sorted forming-bond difference <= 0.02 A"),
        same_ts_any=dict(n=n_same, frac_planned=_frac(n_same, N), frac_finished=_frac(n_same, nf),
                         rule="same_ts of the last geometry, converged or not"),
        orca_converged=dict(n=n_conv, frac_planned=_frac(n_conv, N), frac_finished=_frac(n_conv, nf)),
        nimag1=dict(n=n_im1, frac_planned=_frac(n_im1, N), frac_finished=_frac(n_im1, nf),
                    rule=f"DFT Freq: exactly one mode below -{IMAG_TOL_CM:g} cm-1 (no Freq = not 1)"),
        n_imag_counts=dict(Counter(str(v) for v in g("n_imag"))),
        n_neg_counts=dict(Counter(str(v) for v in g("n_neg"))),
        dd_max_A=dict(_stats(g("dd_max")), definition="max over the 2 label forming bonds |d(final) - d(label TS)|"),
        dd_max_converged_A=_stats(g("dd_max", conv)), rmsd_heavy_A=_stats(g("rmsd_heavy")),
        wall=dict(orca_wall_s=_stats(g("wall_s")),
                  per_cycle_s=_stats([v for x in fin for v in ((x.get("walls") or {}).get("per_cycle_s") or [])]),
                  freq_phase_s=_stats([(x.get("walls") or {}).get("freq_phase_s") for x in fin]),
                  recalc_hessian_s=_stats([h["wall_s"] for h in hw if h.get("phase") == "recalc"]),
                  freq_hessian_s=_stats([h["wall_s"] for h in hw if h.get("phase") == "freq"]),
                  note=f"per-step walls from polling the output every {WATCH_S:g} s (approximate)"),
        module_timings_s_median={k: _stats([(x.get("module_timings_s") or {}).get(k) for x in fin]).get("median")
                                 for k in tkeys},
        mlip_start=dict(start_same_ts_frac=_frac(sum(bool(x.get("start_same_ts")) for x in fin), nf),
                        start_dd_max_A=_stats(g("start_dd_max")),
                        note="the MLIP TS itself vs the label TS (= the FT-2 record numbers)"),
        mlip_stage_cost=dict(ft2_wall_s=_stats([(p.get("mlip_stage") or {}).get("wall_s") for p in preps]),
                             ft2_gpu_s=_stats([(p.get("mlip_stage") or {}).get("gpu_s") for p in preps]),
                             prep_hessian_cpu_s=_stats([p.get("hess_s") for p in preps]),
                             note="context only, not in core_h: FT-2 MLIP OptTS (1 ORCA core + GPU) and this pilot's "
                                  "CPU MLIP Hessian"),
        prep=dict(source_counts=dict(Counter(str(p.get("source")) for p in preps)),
                  d_opt_max_dev_A=_stats([p.get("d_opt_max_dev_A") for p in preps]).get("max"),
                  mlip_n_imag_agree=sum(p.get("mlip_n_imag_now") == p.get("rec_n_imag") for p in preps), n=len(preps)),
        by_dipole={})
    for dn in C.DIPOLES:
        xs = [res[r["rxn_id"]] for r in sel if r["dipole"] == dn and r["rxn_id"] in res]
        npl = sum(r["dipole"] == dn for r in sel)
        S["by_dipole"][dn] = dict(n_planned=npl, n_finished=len(xs),
                                  converged_to_label=sum(bool(x.get("converged_to_label")) for x in xs),
                                  nimag1=sum(x.get("n_imag") == 1 for x in xs),
                                  orca_converged=sum(bool(x.get("converged")) for x in xs),
                                  cycles_median=_stats(g("n_cycles", xs)).get("median"),
                                  core_h_median=_stats(g("core_h", xs)).get("median"),
                                  core_h_mean=_stats(g("core_h", xs)).get("mean"),
                                  dd_median_A=_stats(g("dd_max", xs)).get("median"))
    cm, cmed, ctl = S["core_h"].get("mean"), S["core_h"].get("median"), S["converged_to_label"]["frac_planned"]
    S["rule_check"] = dict(
        rule="user 11-05 18:0x (3): A'' is the deploy engine only if cost <= 1.5 core-h per reaction AND convergence "
             "to the label TS >= 0.90 (applies when FT2-r2 misses the MLIP-only rule)",
        final=S["complete"], cost_mean_le_1p5=None if cm is None else bool(cm <= COST_MAX_CORE_H),
        cost_median_le_1p5=None if cmed is None else bool(cmed <= COST_MAX_CORE_H),
        conv_to_label_ge_0p90=None if ctl is None else bool(ctl >= CONV_MIN),
        pass_with_cost_mean=None if (cm is None or ctl is None) else bool(cm <= COST_MAX_CORE_H and ctl >= CONV_MIN),
        pass_with_cost_median=None if (cmed is None or ctl is None) else bool(cmed <= COST_MAX_CORE_H and ctl >= CONV_MIN),
        note="the rule does not say which cost statistic; both are given (interim while final is false)")
    return S


def _f(v, d=3):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def summary_text(S, sel, res):
    gr, hr, hf, ch = S["dft_gradients"], S["dft_hessians"]["recalc"], S["dft_hessians"]["freq"], S["core_h"]
    ct, sa, oc, ni, dd = S["converged_to_label"], S["same_ts_any"], S["orca_converged"], S["nimag1"], S["dd_max_A"]
    w, rc = S["wall"], S["rule_check"]
    L = [f"A'' pilot: {MODEL} label-start MLIP TS -> label-level ORCA OptTS Freq (D1 opt_ts block, Calc_Hess -> InHess "
         f"Read mlip.hess, Recalc_Hess 30), {NPROCS} cores/rxn",
         f"updated {S['written']}  job {S['job']}  finished {S['n_finished']}/{S['n_planned']}  running {S['n_running']}"
         f"{' ' + str(S['running']) if S['running'] else ''}  complete {S['complete']}",
         f"status {S['status_counts']}",
         f"reasons {S['reason_counts']}",
         f"DFT gradients (opt cycles): median {_f(gr.get('median'), 1)}  p90 {_f(gr.get('p90'), 1)}  mean "
         f"{_f(gr.get('mean'), 1)}  max {_f(gr.get('max'), 0)}",
         f"DFT Hessians Recalc_Hess: total {S['dft_hessians']['recalc_total']}  per rxn {S['dft_hessians']['recalc_counts']}"
         f"  median {_f(hr.get('median'), 1)} p90 {_f(hr.get('p90'), 1)}  at cycles {S['dft_hessians']['recalc_at_cycles']}",
         f"DFT Hessians final Freq: total {S['dft_hessians']['freq_total']}  per rxn {S['dft_hessians']['freq_counts']}",
         f"core-h per rxn: median {_f(ch.get('median'), 3)}  p90 {_f(ch.get('p90'), 3)}  mean {_f(ch.get('mean'), 3)}  "
         f"max {_f(ch.get('max'), 3)}  sum {_f(ch.get('sum'), 2)}",
         f"converged to label TS (ORCA conv & same_ts): {ct['n']}  of planned {_f(ct['frac_planned'])}  of finished "
         f"{_f(ct['frac_finished'])}",
         f"same_ts (last geometry): {sa['n']}  of planned {_f(sa['frac_planned'])}   ORCA converged: {oc['n']}  of "
         f"planned {_f(oc['frac_planned'])}  of finished {_f(oc['frac_finished'])}",
         f"exactly 1 imag (DFT Freq < -50 cm-1): {ni['n']}  of planned {_f(ni['frac_planned'])}  of finished "
         f"{_f(ni['frac_finished'])}",
         f"n_imag counts {S['n_imag_counts']}   raw negative-mode counts {S['n_neg_counts']}",
         f"|dd| vs label: median {_f(dd.get('median'), 4)}  p90 {_f(dd.get('p90'), 4)} A (converged only: median "
         f"{_f(S['dd_max_converged_A'].get('median'), 4)}); heavy RMSD median {_f(S['rmsd_heavy_A'].get('median'), 4)} A",
         f"MLIP start vs label: same_ts {_f(S['mlip_start']['start_same_ts_frac'])}  |dd| median "
         f"{_f(S['mlip_start']['start_dd_max_A'].get('median'), 4)} A",
         f"wall: ORCA run median {_f(w['orca_wall_s'].get('median'), 0)} s; per cycle median "
         f"{_f(w['per_cycle_s'].get('median'), 0)} s; Freq phase median {_f(w['freq_phase_s'].get('median'), 0)} s; "
         f"recalc Hessian median {_f(w['recalc_hessian_s'].get('median'), 0)} s (approx.)",
         f"prep: {S['prep']['source_counts']} max |d - d_opt(rec)| {_f(S['prep']['d_opt_max_dev_A'], 7)} A; MLIP n_imag "
         f"agree {S['prep']['mlip_n_imag_agree']}/{S['prep']['n']}",
         f"rule (cost <= 1.5 core-h AND conv_to_label >= 0.90; final={rc['final']}): cost mean ok {rc['cost_mean_le_1p5']}, "
         f"median ok {rc['cost_median_le_1p5']}, conv ok {rc['conv_to_label_ge_0p90']} -> pass(mean) "
         f"{rc['pass_with_cost_mean']} pass(median) {rc['pass_with_cost_median']}",
         "by dipole: planned/finished conv_to_label nimag1 orca_conv | cycles med | core-h med/mean | |dd| med"]
    for dn, b in S["by_dipole"].items():
        L.append(f"  {dn:17s} {b['n_planned']:2d}/{b['n_finished']:2d}  {b['converged_to_label']:2d} {b['nimag1']:2d} "
                 f"{b['orca_converged']:2d} | {_f(b['cycles_median'], 1)} | {_f(b['core_h_median'], 2)}/"
                 f"{_f(b['core_h_mean'], 2)} | {_f(b['dd_median_A'], 4)}")
    L.append("finished (latest first): rxn dipole N status:reason cyc hess(r+f) core-h same dd nimag/neg")
    fin = sorted(res.values(), key=lambda x: str(x.get("finished")), reverse=True)
    room = max(0, 60 - len(L))
    for x in fin[:room]:
        L.append(f"  {x['rxn_id']} {str(x['dipole'])[:14]:14s} {x['n_atoms']:3d} {x.get('status')}:{x.get('reason')} "
                 f"{x.get('n_cycles')} {x.get('n_hess_recalc')}+{x.get('n_hess_freq')} {_f(x.get('core_h'), 2)} "
                 f"{x.get('same_ts')} {_f(x.get('dd_max'), 4)} {x.get('n_imag')}/{x.get('n_neg')}")
    return "\n".join(L[:60]) + "\n"


def write_summary(sel, running=()):
    res = load_results(sel)
    S = summarize(sel, res, running)
    C.write_text(SUMMARY_TXT, summary_text(S, sel, res))
    return S, res


RES_COLS = ["pick", "rxn_id", "job_id", "dipole", "n_atoms", "charge", "status", "reason", "converged",
            "converged_to_label", "same_ts", "rmsd_heavy", "dform_sorted", "dd_max", "n_imag", "n_neg", "imag_cm",
            "n_cycles", "n_gradients", "n_hess_recalc", "n_hess_freq", "core_h", "wall_s", "start_same_ts",
            "start_dd_max", "rec_same_ts", "rec_dd_max", "prep_source", "d_opt_max_dev_A", "finished"]


def cmd_report():
    sel = read_selection()
    S, res = write_summary(sel)
    ref = _rj(AP / "ref_partb.json")
    if ref:
        S["reference_d1_block"] = dict(ref, note="read only: the unchanged D1 block (Calc_Hess true, Recalc_Hess 30) in "
                                                  "d1_g3 Part B B.1002979 (100 cycles): where ORCA computed SCF Hessians")
    C.write_json(SUMMARY_JSON, S)
    C.write_json(RES_DIR / "app_pilot_summary.json", S)
    C.write_text(RES_DIR / "SUMMARY.txt", SUMMARY_TXT.read_text())
    rows = []
    for r in sel:
        x = res.get(r["rxn_id"]) or {}
        p = x.get("prep") or {}
        rows.append(dict({k: x.get(k) for k in RES_COLS}, pick=r["pick"], rxn_id=r["rxn_id"], job_id=r["job_id"],
                         dipole=r["dipole"], n_atoms=r["n_atoms"], charge=r["charge"], status=x.get("status") or "missing",
                         rec_same_ts=r.get("rec_same_ts"), rec_dd_max=r.get("rec_dd_max"), prep_source=p.get("source"),
                         d_opt_max_dev_A=p.get("d_opt_max_dev_A")))
    C.write_text(RES_DIR / "results.csv", csv_text(rows, RES_COLS))
    print(SUMMARY_TXT.read_text(), flush=True)
    return 0 if S["complete"] else 2


# ================================================================ CLI
def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] not in ("select", "prep", "run", "smoke", "report"):
        print(__doc__)
        return 2
    only = None
    if "--only" in a:
        k = a.index("--only")
        only = parse_only(a[k + 1] if k + 1 < len(a) else "")
    if a[0] == "select":
        return cmd_select()
    if a[0] == "prep":
        return cmd_prep(only=only, smoke="--smoke" in a)
    if a[0] == "run":
        return cmd_run(only=only)
    if a[0] == "smoke":
        return cmd_run(smoke=True)
    return cmd_report()


if __name__ == "__main__":
    sys.exit(main())
