#!/usr/bin/env python3
r"""eval_v2.py — DEPLOY-V2 (MLTS) Phase E evaluation (docs/specs/DEPLOY_V2_MLTS.md 11; grade = 0-4 as registered in
results/PREREG_V2.md / prereg_v2.json). Compute nodes only (jobs/mlt.slurm or jobs/sbx.sh); each command re-executes
itself in the environment it needs, so any MLTENV can start it.

  python eval_v2.py run [--no-figures]
        readiness (stdlib, any env) -> the per-reaction structure tables (`geom`, D1 env, as a child, when missing or
        stale) -> env_espley (= LADDER evaluate.py E1-1) -> results/E_metrics.csv, E_compare.csv, E_bends.csv,
        E_bend_rates.json, E_metrics_ladder.csv, E_goal.csv, E_report.json {complete, grade, prereg_change_needed,
        prereg_change_reason, ...}, figures results/E_fig_para_mae.png, E_fig_dd_error.png (+ .csv of the plotted
        values); scratch $MLT/eval/{E_pairs.csv, E_pair_errors.parquet} (read by final_w.py / predict_v2.py).
        Exit 0 written (also with prereg_change_needed: the launcher's decide_e HALTs on it), 2 not ready, 3 STOP.
  python eval_v2.py geom [--root R] [--tag T] [--engine A|Ap|App]
        D1 env: per D1 target with an A-2 folder R/ade/jobs/<job_id> (default R = $MLT, tag main) the A-3 record of
        mlts_autode.arm_record (rmsd.same_ts 0.10 / 0.02 A vs the label TS, |dd| = max over the label forming pairs
        after atom mapping, mapped heavy RMSD, n_imag < -50 cm-1, cost) -> $MLT/eval/E_geom_<T>.csv + .json.
  python eval_v2.py selftest
        the pipeline on LADDER / DEPLOY files only (no MLTS output): dft REP8 single frames on the full dft test set
        must reproduce results E1_metrics.csv / E1_bend_rates.json (para MAE, CI, sens / FPR; 1e-9), lf on its own set
        the DEPLOY E_metrics.csv para MAE, C1 := dft relabelled gives MLTS = dft (d = 0), the label dd_disp = disp(a) -
        disp(b), grade rule unit cases -> $MLT/eval/selftest/E_selftest.json (exit 3 when a check fails).

Inputs (read only)
  results/PREREG_V2.md + prereg_v2.json (md_sha256 = sha256(md); the registered grade rules = this file's constants,
  imported from prereg_v2.py), results/K_report.json {complete, primary, configs, reps} (K), $MLT/preds/<config>
  _predictions.parquet (L_predictions layout; C1 also $MLT/preds/C1_representatives.json), $MLT/pipeline/arms.json
  {engine, extra_arm_App, app_root, app_usable}, the H tables $MLT/inputs/mlts[_min].parquet (D3_int), LADDER
  analysis/d1_ladder/results/L_predictions.parquet (rung dft; L_report.json sha256), DEPLOY analysis/deploy/results/
  K_predictions.parquet (lf), D1PAIR S_folds.csv, $LAD/F_pairs_cr2.csv, analysis/d1_g2/results/C_compare.csv.
Reused (imported, unchanged): analysis/geo2/deploy/eval_deploy.py context / fold_check / rung_part (the LADDER E1
  pipeline per rung: metrics_e1, bends_e1, goal_e1, goal_rep8), analysis/d1_ladder/evaluate.py (metrics_e1, met_index,
  bends_e1, _clean), analysis/geo2/eval_geo2.py (reps, load_units, boot_diff), analysis/geo2/eval_d0d1.py boot_mae_ci,
  analysis/deploy/eval_tf.py (hidden, hidden_rates, auc, boot_auc, boot_index, pct_ci), analysis/d1_pair/d1_figures.py
  (style, palette slots), analysis/mlts/mlts_autode.py arm_record (geom), prereg_v2.py constants.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)
import prereg_v2 as PR  # noqa: E402            (registered constants; stdlib only at import)

CODE_VERSION = "mlts-e-1"
REPO = M.REPO
EVAL = M.MLT / "eval"
PREDS = M.MLT / "preds"
K_REPORT = M.RES / "K_report.json"
ARMS_JSON = M.MLT / "pipeline" / "arms.json"
PREREG_MD, PREREG_JS = M.RES / "PREREG_V2.md", M.RES / "prereg_v2.json"
C1_REPS = PREDS / "C1_representatives.json"
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
L_PRED, L_REPORT = LAD_RES / "L_predictions.parquet", LAD_RES / "L_report.json"
LF_PRED = REPO / "analysis" / "deploy" / "results" / "K_predictions.parquet"
DEP_E_METRICS = REPO / "analysis" / "deploy" / "results" / "E_metrics.csv"
GEO2_DEPLOY = REPO / "analysis" / "geo2" / "deploy"
DEPLOY_DIR = REPO / "analysis" / "deploy"
D1PAIR_DIR = REPO / "analysis" / "d1_pair"
OUT = dict(metrics="E_metrics.csv", pos="E_metrics_pos.csv", compare="E_compare.csv", bends="E_bends.csv",
           rates="E_bend_rates.json", ladder="E_metrics_ladder.csv", goal="E_goal.csv", report="E_report.json",
           fig1="E_fig_para_mae.png", fig2="E_fig_dd_error.png")
TARGETS = list(M.TARGETS)
PRIMARY9 = list(PR.PRIMARY9)
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]
ARMS8, MODELS4 = ("S", "SH"), ("AW", "GNN", "TF", "XGB")
SPLITS = ("all", "same_ts", "dipole", "g1")
CLASSES = ("same", "mixed", "different", "unclassified")
GRADE_KO = {"goal": "목표 달성", "partial": "부분 개선", "fail": "실패"}
COVERAGE_FLAG = 0.95
CROSS_TOL = 1e-9
T0 = time.time()

DECISIONS = {
    "E-sets": "evaluation pairs = D1 T1 test (fold, pair) predicted by the LADDER dft rung AND lf AND the MLTS primary "
              "config (prereg 1: common set); grade, MAE and bends on its P_main reference para pairs; descriptive "
              "configs (C1/C2/C3/App not primary) on common ∩ their own pairs (n and a flag when smaller); coverage = "
              "MLTS primary P_main para pairs / LADDER dft P_main para test pairs",
    "E-judged": "judged MLTS model = the K primary config's REP8 per target with disp = D3_int(a) - D3_int(b) of its H "
                "table (prereg 1 / 7, H-0); a pair without finite D3_int keeps the model disp (counted); the model-disp "
                "variant is reported next to it (disp both ways); dft = REP8 of the LADDER dft units (model disp), "
                "lf = DEPLOY_MODEL lf S·TF 5-fold predictions (prereg 3)",
    "E-frames": "every case is one 'single frame' (pair, target -> y, yhat) scored with the LADDER E1 functions "
                "(evaluate.metrics_e1 on FU2 metrics_tab + d1_eval.boot_diff CI, 2,000 core resamples, seed 20261019; "
                "evaluate.bends_e1 = FU2 bends_tab on P_main reference para families >= 12 points incl. H, d1_eval "
                "bend_rates 'all'); the 8-unit rungs (dft, C1/C2/C3) also through eval_deploy.rung_part (LADDER tables, "
                "goal_e1 + goal_rep8, descriptive) and their REP8 MAE must equal the single frame (1e-9, else STOP)",
    "E-hidden": "hidden-bend family / continuous g AUC = deploy eval_tf.hidden / hidden_rates / auc on the same bends_e1 "
                "rows (P_main families; DEPLOY's own numbers used the 89 P_sens families), family bootstrap 2,000 "
                "(eval_tf.boot_index, seed 20261019); the cases share the families, so MLTS - lf and MLTS - dft AUC "
                "differences use the same resamples",
    "E-grade": "prereg 1: goal = oi and cpcm para MAE <= 0.13 AND sens >= 0.7, FPR <= 0.1 for d1, elst, pauli, oi "
               "(point estimates; NaN = not met); partial = all 9 targets MAE_MLTS <= 0.5 x MAE_lf; else fail; "
               "'surely' = the CI bound also meets it; the grade uses disp = D3_int, the model-disp grade is reported",
    "E-split": "breakdowns on the P_main para pairs: (i) pair class from the MLTS TS vs the label TS (mlts_autode A-3 "
               "rule rmsd.same_ts; same = both ends same, mixed = one, different = neither, unclassified = an end "
               "without an ok record), (ii) dipole, (iii) g1 shape class (C_compare.csv same_ts_g1, LADDER SAME_TS_RULE "
               "pair classes); MAE + core bootstrap CI (eval_d0d1.boot_mae_ci); descriptive",
    "E-prereg": "prereg_change_needed (launcher HALT) when: the registered grade rules != the code constants, the C1 "
                "primary was predicted with refs != the registered refs.primary (when K records refs), the common "
                "P_main para set is empty, or the primary REP8 lacks a target; coverage < 0.95, smaller descriptive "
                "sets, D3 fallbacks and a missing declared App arm are flags only",
    "E-pos": "secondary positional metrics (spec 11 '위치별(o, m, direct)은 부 지표'): the LADDER metrics_e1 rows of "
             "every frame (sets P_main / test_all as metrics_e1 gives them, positions all / p / m / o / direct) -> "
             "results/E_metrics_pos.csv (case, disp_source); the grade uses P_main para only",
    "E-hidden-families": "the hidden-bend families are the bends_e1 P_main reference para families of the common pairs "
                         "(dft / lf / MLTS share them); DEPLOY's published lf hidden PPV used its 89 P_sens families, so "
                         "it is not the comparator (flag + n_fam in the report; prereg 1)",
    "E-figs": "figures: para MAE bars (dft / lf / MLTS, 9 targets, log axis, CI whiskers, d1_figures palette slots 1-3) "
              "and per-pair |dd| (max of the two ends) vs |error| of the judged MLTS for oi, cpcm, pauli, d1 (log-log, "
              "Spearman rho in the panel title); the plotted values next to each png as csv",
}


def log(msg):
    print(f"[E {time.time() - T0:7.0f}s] {msg}", flush=True)


def rj(p):
    try:
        return M.read_json(p)
    except Exception:                                               # noqa: BLE001
        return None


def fsha(p):
    return M.sha256(p) if Path(p).is_file() else None


# ================================================================ environments
def run_in_env(env, args):
    """`python eval_v2.py <args>` in MLTENV=env (jobs/env.sh: reactot | mlts | d1) as a child; returns its exit code."""
    cmd = ["bash", "-c", f'cd "$0" && export MLTENV={env} && source jobs/env.sh && exec "$PY" "$@"', str(HERE),
           str(THIS)] + [str(a) for a in args]
    log(f"child ({env}): {' '.join(str(a) for a in args)}")
    return subprocess.run(cmd, env=dict(os.environ)).returncode


def ensure_espley():
    import inputs_mlts as IM                                       # stdlib only at import
    IM.ensure_env("espley")


# ================================================================ readiness (stdlib)
def _path_of(p):
    """a predictions entry: a path string, or a dict {path, sha256, ...} (older K_report layout)."""
    if isinstance(p, dict):
        p = p.get("path")
    return str(p) if p else None


def _cfg_list(K):
    """K_report configs -> {name: dict(predictions, refs, app, table, units_dir, rung)} (list of names or dict
    accepted). predictions may be a path or a {path, ...} dict; refs fall back to the K plan's refs (K_report.refs)
    for the trained configs and to the C1 report for C1 (train_k.py c1pred); units_dir to $MLT/units/<config>."""
    raw = K.get("configs") or []
    if isinstance(raw, dict):
        items = [(str(k), dict(v) if isinstance(v, dict) else {}) for k, v in raw.items()]
    else:
        items = [(str(x), {}) if not isinstance(x, dict) else (str(x.get("name")), dict(x)) for x in raw]
    out = {}
    for name, c in items:
        if name in ("C1v",) or name.startswith("C2_s"):          # inner-validation units / seed sets: not evaluated
            continue
        p = _path_of(c.get("predictions")) or _path_of(c.get("path")) or str(PREDS / f"{name}_predictions.parquet")
        refs = c.get("refs")
        if refs is None and not name.startswith("C1"):
            refs = K.get("refs")
        if refs is None and name == "C1":
            refs = (rj(M.RES / "C1_report.json") or {}).get("refs")
        out[name] = dict(name=name, predictions=p, refs=refs, table=c.get("table"),
                         units_dir=c.get("units_dir") or (None if name.startswith("C1") else str(M.MLT / "units" / name)),
                         app=bool(c.get("app") or c.get("arm") == "App" or "App" in name), rung=c.get("rung"),
                         descriptive=bool(c.get("descriptive")))
    return out


def readiness():
    """everything E needs before any channel number is read: (info dict). Exit 2 / 3 on missing / broken inputs."""
    if not (PREREG_MD.is_file() and PREREG_JS.is_file()):
        M.not_ready(f"{PREREG_MD} / {PREREG_JS} missing (Phase R first)")
    pj = M.read_json(PREREG_JS)
    if pj.get("md_sha256") != M.sha256(PREREG_MD):
        M.stop(f"{PREREG_MD.name} changed after {PREREG_JS.name} (md_sha256)")
    K = rj(K_REPORT)
    if not K or K.get("complete") is not True:
        M.not_ready(f"{K_REPORT} missing or incomplete (Phase K)")
    cfgs = _cfg_list(K)
    if "C1" not in cfgs and (PREDS / "C1_predictions.parquet").is_file():
        cfgs["C1"] = dict(name="C1", predictions=str(PREDS / "C1_predictions.parquet"),
                          refs=(rj(M.RES / "C1_report.json") or {}).get("refs"), table=None, units_dir=None, app=False,
                          rung=None)
    prim = K.get("primary")
    if prim not in cfgs:
        M.stop(f"K_report primary {prim!r} not among the configs {sorted(cfgs)}")
    arms = rj(ARMS_JSON)
    if arms is None:
        M.not_ready(f"{ARMS_JSON} missing (written by the launcher after A-3)")
    app_ok = bool(arms.get("extra_arm_App") and arms.get("app_usable") and arms.get("app_root"))
    flags = []
    for name, c in list(cfgs.items()):
        if c["app"] and not app_ok:
            flags.append(f"config {name} is an App arm but arms.json does not declare a usable App arm: left out")
            del cfgs[name]
            continue
        if not Path(c["predictions"]).is_file():
            if name == prim:
                M.not_ready(f"{c['predictions']} (primary {name}) missing")
            flags.append(f"config {name}: predictions {c['predictions']} missing: left out")
            del cfgs[name]
    if app_ok and not any(c["app"] for c in cfgs.values()):
        flags.append("arms.json declares a usable App arm but K has no App config predictions")
    for p in (L_PRED, L_REPORT, LF_PRED):
        if not p.is_file():
            M.stop(f"{p} missing (comparator)")
    return dict(K=K, prereg=pj, configs=cfgs, primary=prim, arms=arms, app_ok=app_ok, flags=flags)


def count_folders(jobs):
    try:
        return sum(1 for e in os.scandir(jobs) if e.is_dir() and not e.name.startswith("."))
    except FileNotFoundError:
        return 0


def geom_paths(tag):
    return EVAL / f"E_geom_{tag}.csv", EVAL / f"E_geom_{tag}.json"


def geom_roots(info):
    """[(tag, root, engine, A4 report path)] the structure tables E needs (main engine; App arm when usable)."""
    arms = info["arms"]
    eng = arms.get("engine") or "A"
    out = [("main", M.MLT, eng, M.RES / "A4_report.json")]
    if info["app_ok"]:
        r = Path(arms["app_root"])
        out.append(("App", r, "App", r / "reports" / "A4_report.json"))
    return out


def geom_fresh(tag, root, a4):
    c, j = geom_paths(tag)
    meta = rj(j)
    if not (c.is_file() and meta):
        return False
    return meta.get("n_folders") == count_folders(Path(root) / "ade" / "jobs") and meta.get("a4_sha256") == fsha(a4) \
        and meta.get("csv_sha256") == M.sha256(c)


# ================================================================ geom (D1 env)
def _geom_one(args):
    row, arm, folder, fp, al = args
    try:
        import mlts_autode as MA
        rec = MA.arm_record(row, arm, Path(folder), {row["rxn_id"]: fp}, al)
    except Exception as e:                                          # noqa: BLE001  (record the failure, keep going)
        rec = dict(row, arm=arm, status="error", reason=f"{type(e).__name__}: {e}"[:200])
    return rec


GEOM_COLS = ("rxn_id", "job_id", "fold_T1", "dipole", "n_atoms", "arm", "status", "reason", "compare", "same_ts",
             "rmsd_heavy", "dform_sorted", "dd_max", "dd_how", "n_imag", "core_h", "core_h_own", "gpu_min", "wall_s",
             "n_ext_calls", "disk_gb", "n_dft_cycles", "dft_core_h", "core_h_alloc", "gpu_min_alloc", "pre_start_fail")


def cmd_geom(root=None, tag="main", engine="A"):
    if os.environ.get("MLTENV") != "d1":
        args = ["geom", "--tag", tag, "--engine", engine] + (["--root", str(root)] if root else [])
        return run_in_env("d1", args)
    from concurrent.futures import ProcessPoolExecutor
    R = Path(root) if root else M.MLT
    jobs = R / "ade" / "jobs"
    T = M.targets()
    import mlts_autode as MA                                       # A2-cost allocated (the A-4 GPU jobs' alloc logs)
    alloc = {a: {} for a in ("A", "Ap")}
    for k in sorted({int(x) for x in T.fold_T1}):
        for a in alloc:
            alloc[a].update(MA.read_alloc(f"d1:{k}", a))
    rows = []
    for r in T.itertuples():
        d = jobs / str(r.job_id)
        if d.is_dir():
            al = {a: {str(r.job_id): v[str(r.job_id)]} for a, v in alloc.items() if str(r.job_id) in v}
            rows.append((dict(rxn_id=int(r.rxn_id), job_id=str(r.job_id), fold_T1=int(r.fold_T1), dipole=str(r.dipole),
                              n_atoms=int(r.n_atoms)), engine, str(d), str(r.formed_pairs_ts), al))
    if not rows:
        M.not_ready(f"no A-2 folder under {jobs}")
    n = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1))
    log(f"geom {tag}: {len(rows)} folders under {jobs}, {n} processes")
    with ProcessPoolExecutor(n) as ex:
        recs = list(ex.map(_geom_one, rows, chunksize=16))
    c, j = geom_paths(tag)
    M.assert_write_ok(c)
    c.parent.mkdir(parents=True, exist_ok=True)
    tmp = c.with_name(f".{c.name}.tmp{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(GEOM_COLS), extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in GEOM_COLS})
    os.replace(tmp, c)
    st = {}
    for r in recs:
        st[str(r.get("status"))] = st.get(str(r.get("status")), 0) + 1
    a4 = (M.RES / "A4_report.json") if R == M.MLT else (R / "reports" / "A4_report.json")
    M.write_json(j, dict(tag=tag, root=str(R), engine=engine, n_folders=count_folders(jobs), n_records=len(recs),
                         status=st, n_same_ts=sum(1 for r in recs if r.get("same_ts") in (True, "True")),
                         a4_sha256=fsha(a4), csv_sha256=M.sha256(c), written=M.now(), code=CODE_VERSION,
                         rule="mlts_autode.arm_record (A3-metrics)"))
    log(f"geom {tag}: {st} -> {c}")
    return 0


# ================================================================ espley-side imports
G = {}


def mods():
    if G:
        return G
    for p in (str(GEO2_DEPLOY),):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import pandas as pd
    import eval_deploy as ED2                                     # puts geo2, d1_ladder on sys.path
    if str(DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPLOY_DIR))
    import eval_tf as ETF
    if str(D1PAIR_DIR) not in sys.path:
        sys.path.insert(0, str(D1PAIR_DIR))
    G.update(np=np, pd=pd, ED2=ED2, LE=ED2.LE, EG=ED2.EG, ED=ED2.ED, ETF=ETF)
    LE = ED2.LE
    if (LE.N_BOOT, LE.SEED) != (PR.N_BOOT, PR.BOOT_SEED) or tuple(LE.PRIMARY9) != tuple(PRIMARY9):
        M.stop("LADDER evaluate.py bootstrap / target constants differ from the pre-registration")
    return G


def _clean(o):
    return mods()["LE"]._clean(o)


# ================================================================ predictions and selections
def norm_pred(P, name):
    pd = mods()["pd"]
    miss = [c for c in PRED_COLS if c not in P.columns]
    if miss:
        M.stop(f"{name}: prediction columns missing {miss}")
    P = P[PRED_COLS].copy()
    P["pair_id"] = P.pair_id.astype(str)
    P["fold"] = P.fold.astype(int)
    P["ref_kind"] = P.ref_kind.fillna("").astype(str)
    P["pos"] = P.pos.fillna("").astype(str)
    if P.duplicated(["pair_id", "arm", "model", "target"]).any():
        M.stop(f"{name}: duplicated (pair, arm, model, target) rows")
    if not mods()["np"].isfinite(P.yhat.to_numpy(float)).all():
        M.stop(f"{name}: non-finite yhat")
    del pd
    return P.reset_index(drop=True)


def keys(P):
    q = P[P.target == TARGETS[0]][["fold", "pair_id"]].drop_duplicates()
    return set(zip(q.fold.astype(int), q.pair_id.astype(str)))


def dft_reps():
    """REP / REP8 of the LADDER dft units (eval_geo2.reps on L_report + L_xgb_report), REP = E1_report (else STOP)."""
    EG = mods()["EG"]
    rep, rep8, _ = EG.reps(EG.load_units(EG.BASE_UNITS))
    e1 = ((rj(LAD_RES / "E1_report.json") or {}).get("representatives") or {}).get("dft") or {}
    bad = [(a, t) for a in ARMS8 for t in TARGETS if rep.get(("dft", a, t)) != (e1.get(a) or {}).get(t, {}).get("model")]
    if bad:
        M.stop(f"dft REP differs from E1_report.json representatives: {bad[:5]}")
    return ({(a, t): rep[("dft", a, t)] for a in ARMS8 for t in TARGETS},
            {t: tuple(rep8[("dft", t)]) for t in TARGETS})


def _pick(d, rung):
    """a reps block keyed by arm/target directly, or by rung first."""
    if not isinstance(d, dict) or not d:
        return None
    if set(d) & set(ARMS8) or set(d) & set(TARGETS):
        return d
    for k in (rung, "mlts", "dft"):
        if k in d:
            return d[k]
    return next(iter(d.values()))


def parse_reps(src, rung):
    if not isinstance(src, dict):
        return None, None
    r8 = _pick(src.get("REP8") or src.get("rep8"), rung)
    rp = _pick(src.get("REP") or src.get("rep"), rung)
    rep8 = {t: tuple(r8[t]) for t in TARGETS} if isinstance(r8, dict) and all(t in r8 for t in TARGETS) else None
    rep = None
    if isinstance(rp, dict) and all(a in rp for a in ARMS8):
        rep = {(a, t): rp[a][t] for a in ARMS8 for t in TARGETS if t in rp[a]}
        if len(rep) != len(ARMS8) * len(TARGETS):
            rep = None
    return rep, rep8


def unit_reps(units_dir, rung):
    """fallback: LADDER 13-1 / REP8 on the K unit jsons L__<rung>__<arm>__<model>__f<k>.json (val_kcal_mae)."""
    pd, EG = mods()["pd"], mods()["EG"]
    rows = []
    for a in ARMS8:
        for m in MODELS4:
            for k in range(5):
                p = Path(units_dir) / f"L__{rung}__{a}__{m}__f{k}.json"
                J = rj(p)
                if not J or not isinstance(J.get("val_kcal_mae"), dict):
                    return None, None
                rows += [dict(rung=rung, arm=a, model=m, fold=k, target=t, val=float(v))
                         for t, v in J["val_kcal_mae"].items()]
    rep, rep8, _ = EG.reps(pd.DataFrame(rows))
    return ({(a, t): rep[(rung, a, t)] for a in ARMS8 for t in TARGETS}, {t: tuple(rep8[(rung, t)]) for t in TARGETS})


def config_reps(c, K, rung, dft_rep, dft_rep8):
    src = (K.get("reps") or {}).get(c["name"]) if isinstance(K.get("reps"), dict) else None
    rep, rep8 = parse_reps(src, rung)
    how = "K_report reps"
    if rep8 is None and c["name"].startswith("C1") and C1_REPS.is_file():
        rep, rep8 = parse_reps(M.read_json(C1_REPS), "mlts")
        how = "C1_representatives.json"
    if rep8 is None:
        rep, rep8 = unit_reps(c.get("units_dir") or (M.MLT / "units"), rung)
        how = "unit jsons (LADDER 13-1 on val_kcal_mae)"
    if rep8 is None:
        M.not_ready(f"no REP8 for config {c['name']} (K_report reps, C1_representatives.json, unit jsons)")
    if rep is None:
        rep = dict(dft_rep) if c["name"].startswith("C1") else None
        if rep is None:
            M.not_ready(f"no REP (per arm) for config {c['name']}")
    if c["name"].startswith("C1") and (rep8 != dft_rep8 or rep != dft_rep):
        M.stop(f"C1 representatives differ from the LADDER dft choices (C1 = 'LADDER가 고른 것 그대로')")
    return rep, rep8, how


def sel_frame(P, rep8, label):
    """REP8 per target -> one frame (pair_id, fold, target, y, yhat, source)."""
    pd = mods()["pd"]
    parts = []
    for t in TARGETS:
        a, m = rep8[t]
        q = P[(P.target == t) & (P.arm == a) & (P.model == m)]
        if q.empty:
            M.stop(f"{label}: no rows for the REP8 choice {m}·{a} of {t}")
        parts.append(q[["pair_id", "fold", "target", "y", "yhat"]].assign(source=f"{m}·{a}"))
    return pd.concat(parts, ignore_index=True)


def d3_table(c, info):
    """rxn_id -> D3_int of the config's H table (refs min -> mlts_min.parquet, else mlts.parquet; App: app_root)."""
    pd = mods()["pd"]
    if c.get("table"):
        p = Path(c["table"])
    else:
        root = Path(info["arms"]["app_root"]) if c["app"] else M.MLT
        p = root / "inputs" / ("mlts_min.parquet" if (c.get("refs") or "min") == "min" else "mlts.parquet")
        if not p.is_file():
            p = root / "inputs" / "mlts.parquet"
    if not p.is_file():
        return None, str(p)
    T = pd.read_parquet(p, columns=["rxn_id", "D3_int"])
    return dict(zip(T.rxn_id.astype(int), T.D3_int.astype(float))), str(p)


def with_d3(F, d3, S):
    """disp rows -> D3_int(a) - D3_int(b) (finite on both ends; else the model value is kept and counted)."""
    np = mods()["np"]
    F = F.copy()
    m = (F.target == "disp").to_numpy()
    pid = F.pair_id.to_numpy()[m]
    a = S.a_rxn.reindex(pid).to_numpy()
    b = S.b_rxn.reindex(pid).to_numpy()
    va = np.array([d3.get(int(x), np.nan) for x in a], float)
    vb = np.array([d3.get(int(x), np.nan) for x in b], float)
    v = va - vb
    ok = np.isfinite(v)
    yh = F.yhat.to_numpy(float).copy()
    src = F.source.to_numpy(object).copy()
    idx = np.where(m)[0]
    yh[idx[ok]] = v[ok]
    src[idx] = "D3_int" if ok.all() else "D3_int+model"             # one source per target (metrics_tab groups)
    F["yhat"], F["source"] = yh, src
    return F, int((~ok).sum())


# ================================================================ single-frame LADDER E1 scoring
def frame_scores(EV, S, frames):
    """{key: frame} -> metrics (LADDER metrics_e1), MI, bends (bends_e1), rates; key = the 'rung' label."""
    LE = mods()["LE"]
    A = {}
    for k, f in frames.items():
        A[(k, "J", "sel", "common")] = EV.annotate(f, S)
        A[(k, "J", "null", "common")] = EV.annotate(f.assign(yhat=0.0, source="0"), S)
    Mt = LE.metrics_e1(EV, A)
    Mt = Mt[Mt.arm == "J"].reset_index(drop=True)
    MI = LE.met_index(Mt)
    Bd, rates, fams = LE.bends_e1(EV, A, S, {})
    Bd = Bd[Bd.arm == "J"].reset_index(drop=True)
    srcs = {k: f.groupby("target").source.first().to_dict() for k, f in frames.items()}
    Bd["source"] = [srcs.get(r, {}).get(t, "0") if m == "sel" else "0" for r, m, t in zip(Bd.rung, Bd.model, Bd.target)]
    return Mt, MI, Bd, rates, fams


def auc_hidden(Bd, keys_):
    """continuous g AUC (eval_tf.auc, family bootstrap) + hidden-bend rates (eval_tf.hidden) per key."""
    np, ETF = mods()["np"], mods()["ETF"]
    out, boots, famsets = {}, {}, {}
    for k in keys_:
        B = Bd[(Bd.rung == k) & (Bd.model == "sel")]
        fams = sorted(B.fam.unique())
        famsets[k] = fams
        idx = ETF.boot_index(len(fams))
        rec, bs = {}, {}
        for c in ETF.CH:
            q = B[B.target == c].set_index("fam").reindex(fams)
            if q.bent_dft.isna().any():
                rec[c] = dict(auc_g=None, auc_g_ci=[None, None], n_fam=int(q.bent_dft.notna().sum()))
                continue
            lab, sc = q.bent_dft.to_numpy(bool), q.gain_pred.to_numpy(float)
            a = ETF.auc(lab, sc)
            bb = ETF.boot_auc(lab, sc, idx)
            rec[c] = dict(auc_g=a, auc_g_ci=list(ETF.pct_ci(bb)), n_fam=len(fams))
            bs[c] = bb
        hid = ETF.hidden_rates(ETF.hidden(B)) if len(B) else None
        out[k] = dict(targets=rec, hidden=hid, n_fam=len(fams))
        boots[k] = bs
    return out, boots, famsets


def auc_diff(boots, famsets, out, x, b):
    ETF = mods()["ETF"]
    if famsets.get(x) != famsets.get(b):
        return dict(note="different family sets: no paired difference")
    res = {}
    for c in ETF.CH:
        if c in boots[x] and c in boots[b]:
            res[c] = dict(d=out[x]["targets"][c]["auc_g"] - out[b]["targets"][c]["auc_g"],
                          ci=list(ETF.pct_ci(boots[x][c] - boots[b][c])))
    return res


# ================================================================ breakdowns
def reaction_classes(info):
    """{tag: {rxn_id: 'same' | 'different' | None}} from the geom tables + g1 classes (C_compare same_ts_g1)."""
    pd = mods()["pd"]
    out = {}
    for tag, *_ in geom_roots(info):
        c, _ = geom_paths(tag)
        G_ = pd.read_csv(c)
        st = G_.status.astype(str) == "ok"
        same = G_.same_ts.astype(str).str.lower().map({"true": True, "false": False})
        out[tag] = {int(r): ("same" if s is True else "different" if s is False else None) if ok else None
                    for r, ok, s in zip(G_.rxn_id, st, same)}
        out[tag + "_dd"] = {int(r): float(x) for r, ok, x in zip(G_.rxn_id, st, pd.to_numeric(G_.dd_max, errors="coerce"))
                            if ok}
    Cc = pd.read_csv(M.C_COMPARE, usecols=["rxn_id", "same_ts_g1"], dtype={"same_ts_g1": str}, keep_default_na=False)
    out["g1"] = {int(r): {"true": "same", "false": "different"}.get(str(v).strip().lower()) for r, v in
                 zip(Cc.rxn_id, Cc.same_ts_g1)}
    return out


def pair_class(a, b, cls):
    ca, cb = cls.get(int(a)), cls.get(int(b))
    if ca is None or cb is None:
        return "unclassified"
    n = (ca == "same") + (cb == "same")
    return {2: "same", 1: "mixed", 0: "different"}[n]


def breakdown_rows(frames, roles, S, para_ids, cls, geo_tag, MI):
    """E_metrics rows: case x split x group x target (P_main para pairs of the frame's set)."""
    np, pd, ED = mods()["np"], mods()["pd"], mods()["ED"]
    rows, bad = [], []
    for k, f in frames.items():
        ids = sorted(set(f.pair_id) & set(para_ids))
        Pt = S.loc[ids]
        core = Pt.core.astype(str).to_numpy()
        tag = geo_tag.get(k, "main")
        grp = {"all": np.array(["all"] * len(ids), object), "dipole": Pt.dipole.astype(str).to_numpy(),
               "same_ts": np.array([pair_class(a, b, cls.get(tag, {})) for a, b in zip(Pt.a_rxn, Pt.b_rxn)], object),
               "g1": np.array([pair_class(a, b, cls["g1"]) for a, b in zip(Pt.a_rxn, Pt.b_rxn)], object)}
        W = f[f.pair_id.isin(ids)].pivot_table(index="pair_id", columns="target", values=["y", "yhat"], aggfunc="first")
        W = W.reindex(ids)
        case, disp = k.split("#")
        for t in TARGETS:
            e = (W[("yhat", t)] - W[("y", t)]).to_numpy(float)
            y = W[("y", t)].to_numpy(float)
            for sp in SPLITS:
                g = grp[sp]
                for gv in sorted(set(g.tolist())):
                    m = g == gv
                    if not m.any():
                        continue
                    mae, lo, hi = ED.boot_mae_ci(e[m], core[m])
                    rows.append(dict(case=case, disp_source=disp, role=roles[k], pair_set="common", split=sp, group=gv,
                                     target=t, pos="p", n=int(m.sum()), n_cores=int(len(set(core[m]))), mae=mae,
                                     mae_ci_lo=lo, mae_ci_hi=hi, rmse=float(np.sqrt(np.mean(e[m] ** 2))),
                                     mae_null=float(np.abs(y[m]).mean())))
                    if sp == "all":
                        ref = MI[(k, "J", "sel", "common", t, "P_main", "p")]
                        if int(ref["n"]) != int(m.sum()) or abs(float(ref["mae"]) - mae) > CROSS_TOL \
                                or abs(float(ref["mae_ci_hi"]) - hi) > CROSS_TOL:
                            bad.append(f"{k} {t}")
    if bad:
        M.stop(f"breakdown 'all' rows differ from the LADDER metrics_e1 rows: {bad[:5]}")
    return pd.DataFrame(rows)


# ================================================================ grade
def _fin(x):
    try:
        import math
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def grade_of(MI, rates, key, lf_key):
    g = PR.GOAL
    r = rates[f"{key}|J|sel|common"]["all"]
    mae = {t: MI[(key, "J", "sel", "common", t, "P_main", "p")] for t in PRIMARY9}
    lfm = {t: MI[(lf_key, "J", "sel", "common", t, "P_main", "p")] for t in PRIMARY9}
    gm = {}
    for t in PR.GOAL_MAE:
        v, hi = mae[t]["mae"], mae[t]["mae_ci_hi"]
        gm[t] = dict(para_mae=v, ci=[mae[t]["mae_ci_lo"], hi], n=mae[t]["n"], ok=bool(_fin(v) and v <= g["mae"]),
                     surely=bool(_fin(hi) and hi <= g["mae"]))
    gb = {}
    for t in PR.GOAL_BEND:
        b = r[t]
        s, f = b["sens"], b["fpr"]
        ok = bool(_fin(s) and s >= g["sens"] and _fin(f) and f <= g["fpr"])
        sure = bool(ok and _fin(b["sens_ci"][0]) and b["sens_ci"][0] >= g["sens"] and _fin(b["fpr_ci"][1])
                    and b["fpr_ci"][1] <= g["fpr"])
        gb[t] = dict(sens=s, sens_ci=b["sens_ci"], n_bent_dft=b["n_bent_dft"], fpr=f, fpr_ci=b["fpr_ci"],
                     n_straight_dft=b["n_straight_dft"], n_fam=b["n_fam"], ok=ok, surely=sure)
    goal = all(v["ok"] for v in gm.values()) and all(v["ok"] for v in gb.values())
    goal_sure = goal and all(v["surely"] for v in gm.values()) and all(v["surely"] for v in gb.values())
    part = {}
    for t in PRIMARY9:
        a, b = mae[t]["mae"], lfm[t]["mae"]
        part[t] = dict(mae_mlts=a, mae_lf=b, ratio=(a / b if _fin(a) and _fin(b) and b > 0 else None),
                       ok=bool(_fin(a) and _fin(b) and a <= PR.PARTIAL_FACTOR * b),
                       surely=bool(_fin(mae[t]["mae_ci_hi"]) and _fin(b) and mae[t]["mae_ci_hi"] <= PR.PARTIAL_FACTOR * b))
    partial = all(v["ok"] for v in part.values())
    grade = "goal" if goal else ("partial" if partial else "fail")
    return dict(grade=grade, grade_ko=GRADE_KO[grade], goal=dict(passed=goal, surely=goal_sure, mae=gm, bend=gb),
                partial=dict(passed=partial, factor=PR.PARTIAL_FACTOR, per_target=part,
                             surely=bool(partial and all(v["surely"] for v in part.values()))))


def registered_rules():
    g = PR.GOAL
    return [f"para MAE <= {g['mae']} for {list(PR.GOAL_MAE)} AND bend sensitivity >= {g['sens']} and false-positive "
            f"rate <= {g['fpr']} for {list(PR.GOAL_BEND)}",
            f"para MAE_MLTS <= {PR.PARTIAL_FACTOR} x MAE_lf for all 9 targets {list(PR.PRIMARY9)} (same pairs)"]


# ================================================================ figures
def figures(Mrows, PE, out_paths):
    np, pd = mods()["np"], mods()["pd"]
    import d1_figures as DF
    plt = DF.style()
    rec = {}
    q = Mrows[(Mrows.split == "all")]
    cases = [("dft#model", "dft (label TS, upper bound)", DF.SLOT[0]), ("lf#model", "lf (deployed, xTB TS)", DF.SLOT[1]),
             (PE["primary_key"], f"MLTS ({PE['primary']}, disp = D3_int)", DF.SLOT[2])]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.26
    x = np.arange(len(PRIMARY9))
    vals = []
    floor = 1e-3                                                   # SH baselines make cds (and dft disp) ~exact
    for j, (k, lab, col) in enumerate(cases):
        case, disp = k.split("#")
        r = q[(q.case == case) & (q.disp_source == disp)].set_index("target").reindex(PRIMARY9)
        m, lo, hi = r.mae.to_numpy(float), r.mae_ci_lo.to_numpy(float), r.mae_ci_hi.to_numpy(float)
        mp = np.maximum(m, floor)
        ax.bar(x + (j - 1) * w, mp - floor, w * 0.92, bottom=floor, color=col, label=lab, edgecolor="white",
               linewidth=0.6)
        ax.errorbar(x + (j - 1) * w, mp, yerr=[np.clip(m - lo, 0, mp - floor), np.maximum(hi - m, 0)], fmt="none",
                    ecolor=DF.INK2, elinewidth=0.6, capsize=1.5)
        for xi, v in zip(x + (j - 1) * w, m):
            if np.isfinite(v) and v < floor:
                ax.text(xi, floor * 1.15, "<0.001", rotation=90, ha="center", va="bottom", fontsize=5, color=DF.INK2)
        vals += [dict(case=lab, target=t, mae=a, ci_lo=b, ci_hi=c) for t, a, b, c in zip(PRIMARY9, m, lo, hi)]
    ax.set_yscale("log")
    ax.set_ylim(floor, None)
    ax.axhline(PR.GOAL["mae"], color=DF.MUTED, lw=0.7, ls="--")
    ax.text(-0.45, PR.GOAL["mae"] * 1.1, "0.13 = goal (oi, cpcm)", color=DF.INK2, ha="left", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([DF.TLAB.get(t, t) for t in PRIMARY9])
    ax.set_ylabel("para MAE (kcal/mol, log)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
    ax.set_xlabel(f"target (D1 T1 test, P_main para, common pairs n = {PE['n_para']}; whiskers = core bootstrap 95 % CI)")
    _save_fig(fig, out_paths["fig1"], pd.DataFrame(vals))
    rec["fig1"] = str(out_paths["fig1"])
    # |dd| vs |error| (judged MLTS)
    E = PE["pair_err"]
    tg = ["oi", "cpcm", "pauli", "d1"]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=False)
    pts = []
    from scipy.stats import spearmanr
    for ax, t in zip(axs, tg):
        d = E[(E.target == t) & np.isfinite(E.dd_pair)]
        xe, ye = d.dd_pair.to_numpy(float), np.abs(d.err_mlts.to_numpy(float))
        ok = (xe > 0) & (ye > 0)
        ax.scatter(xe[ok], ye[ok], s=3, color=DF.SLOT[2], alpha=0.55, linewidths=0)
        rho = spearmanr(xe, ye).statistic if len(xe) > 2 else float("nan")
        if ok.sum() >= 2:
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "no |dd| records", transform=ax.transAxes, ha="center", color=DF.INK2)
        ax.set_title(f"{DF.TLAB.get(t, t)}  (rho {rho:.2f}, n {len(xe)})", loc="left")
        ax.set_xlabel("pair max |dd| (A)")
        pts += [dict(target=t, pair_id=p, dd_pair=a, abs_err=b) for p, a, b in zip(d.pair_id, xe, ye)]
    axs[0].set_ylabel("|error| (kcal/mol)")
    _save_fig(fig, out_paths["fig2"], pd.DataFrame(pts))
    rec["fig2"] = str(out_paths["fig2"])
    plt.close("all")
    return rec


def _save_fig(fig, path, df):
    path = M.assert_write_ok(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.png")
    fig.savefig(tmp, bbox_inches="tight", dpi=200)
    os.replace(tmp, path)
    M.write_csv(path.with_suffix(".csv"), df)


# ================================================================ the evaluation
def build(info, restrict_to=None, lf_on_own=False):
    """the full computation (no writes) -> dict of tables + report pieces."""
    np, pd, ED2, LE = mods()["np"], mods()["pd"], mods()["ED2"], mods()["LE"]
    K, cfgs, prim = info["K"], info["configs"], info["primary"]
    flags = list(info["flags"])
    rep_d, rep8_d = dft_reps()                                     # validation records only (before predictions)
    sel = {}
    for name, c in cfgs.items():
        sel[name] = config_reps(c, K, c.get("rung") or "mlts", rep_d, rep8_d)
    ctx = ED2.context()
    EV, S = ctx["EV"], ctx["S"]
    lr = rj(L_REPORT) or {}
    if fsha(L_PRED) != lr.get("predictions_sha256"):
        M.stop(f"{L_PRED} changed after {L_REPORT.name}")
    La = pd.read_parquet(L_PRED, columns=PRED_COLS)
    Pd = norm_pred(La[La.rung == "dft"], "dft")
    del La
    Plf = norm_pred(pd.read_parquet(LF_PRED), "lf")
    if set(Plf.rung) != {"lf"} or set(Plf.arm) != {"S"} or set(Plf.model) != {"TF"}:
        M.stop("lf K_predictions.parquet must hold rung lf, arm S, model TF only")
    Pc = {}
    for name, c in cfgs.items():
        P = norm_pred(pd.read_parquet(c["predictions"]), name)
        if set(zip(P.arm, P.model)) != {(a, m) for a in ARMS8 for m in MODELS4}:
            M.stop(f"{name}: arm x model != S/SH x AW/GNN/TF/XGB")
        if P.rung.nunique() != 1:
            M.stop(f"{name}: more than one rung label")
        c["rung_in_file"] = str(P.rung.iloc[0])
        Pc[name] = P.assign(rung=name)
    for nm, P in [("dft", Pd), ("lf", Plf)] + list(Pc.items()):
        ED2.fold_check(P, S, nm)
    Kd, Kl, Kp = keys(Pd), keys(Plf), keys(Pc[prim])
    common = Kd & Kl & Kp
    if restrict_to is not None:
        common = set(restrict_to)
    ids_common = {p for _, p in common}
    para = S[(S.ref_kind == "reference") & S.in_P_main.astype(bool) & (S.pos == "p")].index
    para_set = set(para.astype(str))

    def npara(Kx):
        return sum(1 for _, p in Kx if p in para_set)
    n = dict(ladder_dft_test=len(Kd), lf_test=len(Kl), mlts_primary_test=len(Kp), common=len(common),
             ladder_dft_para=npara(Kd), lf_para=npara(Kl), mlts_primary_para=npara(Kp), common_para=npara(common))
    n["coverage_mlts_para"] = n["mlts_primary_para"] / n["ladder_dft_para"] if n["ladder_dft_para"] else None
    n["coverage_common_para"] = n["common_para"] / n["ladder_dft_para"] if n["ladder_dft_para"] else None
    if n["coverage_mlts_para"] is not None and n["coverage_mlts_para"] < COVERAGE_FLAG:
        flags.append(f"MLTS primary covers {n['coverage_mlts_para']:.3f} of the LADDER P_main para test pairs")
    # frames
    frames, roles, geo_tag, sets_of, d3info = {}, {}, {}, {}, {}
    frames["dft#model"] = sel_frame(Pd[Pd.pair_id.isin(ids_common)], rep8_d, "dft")
    roles["dft#model"] = "upper_bound"
    lf_ids = {p for _, p in Kl} if lf_on_own else ids_common
    frames["lf#model"] = Plf[Plf.pair_id.isin(lf_ids)][["pair_id", "fold", "target", "y", "yhat"]].assign(source="TF·S")
    roles["lf#model"] = "deployed"
    for name, c in cfgs.items():
        own = keys(Pc[name])
        ids = ids_common & {p for _, p in own} if restrict_to is None or name != prim else ids_common
        if len(ids) < len(ids_common):
            flags.append(f"config {name}: {len(ids_common) - len(ids)} common pairs not predicted (descriptive, smaller "
                         "set)")
        sets_of[name] = ids
        F = sel_frame(Pc[name][Pc[name].pair_id.isin(ids)], sel[name][1], name)
        role = "primary" if name == prim else "descriptive"
        frames[f"{name}#model"] = F
        roles[f"{name}#model"] = role + "_model_disp"
        d3, p3 = (info["d3_override"].get(name), "override") if info.get("d3_override") else d3_table(c, info)
        if d3 is None:
            if name == prim:
                M.not_ready(f"H table with D3_int missing for the primary config: {p3}")
            flags.append(f"config {name}: no D3_int table ({p3}); only the model disp is reported")
            continue
        F3, nfb = with_d3(F, d3, S)
        d3info[name] = dict(table=p3, n_fallback_model_disp=nfb)
        if nfb:
            flags.append(f"config {name}: {nfb} pairs without finite D3_int kept the model disp")
        frames[f"{name}#D3_int"] = F3
        roles[f"{name}#D3_int"] = role
        geo_tag[f"{name}#D3_int"] = geo_tag[f"{name}#model"] = "App" if c["app"] else "main"
    log(f"frames: {sorted(frames)}; common {len(common)} pairs ({n['common_para']} P_main para)")
    Mt, MI, Bd, rates, fams = frame_scores(EV, S, frames)
    log("single frames scored (metrics_e1, bends_e1)")
    prim_key = f"{prim}#D3_int"
    ah, boots, famsets = auc_hidden(Bd, list(frames))
    diffs = {f"{prim_key} - lf#model": auc_diff(boots, famsets, ah, prim_key, "lf#model"),
             f"{prim_key} - dft#model": auc_diff(boots, famsets, ah, prim_key, "dft#model")}
    # LADDER rung pipeline (descriptive) + cross-check of the REP8 MAE
    rung_tabs, rung_goal = [], []
    rep_all = {("dft", a, t): rep_d[(a, t)] for a in ARMS8 for t in TARGETS}
    rep8_all = {("dft", t): rep8_d[t] for t in TARGETS}
    for name in cfgs:
        rep_all.update({(name, a, t): sel[name][0][(a, t)] for a in ARMS8 for t in TARGETS})
        rep8_all.update({(name, t): sel[name][1][t] for t in TARGETS})
    cross = []
    if not info.get("skip_rung_part"):
        for rung, P, ids in [("dft", Pd, ids_common)] + [(nm, Pc[nm], sets_of[nm]) for nm in cfgs]:
            part = ED2.rung_part(P[P.pair_id.isin(ids)].reset_index(drop=True), rung, rep_all, rep8_all, ctx)
            rung_tabs.append(part["Mt"].assign(case=rung))
            rung_goal.append(part["goal"].assign(case=rung))
            for t in TARGETS:
                a, m = rep8_all[(rung, t)]
                x = part["MI"][(rung, a, m, "own", t, "P_main", "p")]
                y = MI[(f"{rung}#model", "J", "sel", "common", t, "P_main", "p")]
                if int(x["n"]) != int(y["n"]) or abs(float(x["mae"]) - float(y["mae"])) > CROSS_TOL:
                    cross.append(f"{rung} {t}: rung_part {x['mae']} (n {x['n']}) vs single frame {y['mae']} (n {y['n']})")
            log(f"rung_part {rung}: {part['seconds']:.0f} s")
        if cross:
            M.stop("LADDER rung_part REP8 MAE != single-frame MAE: " + "; ".join(cross[:5]))
    # breakdowns
    cls = reaction_classes(info) if not info.get("cls_override") else info["cls_override"]
    Mrows = breakdown_rows(frames, roles, S, para_set, cls, geo_tag, MI)
    # paired comparisons on the common para pairs
    Fi = S.loc[sorted(ids_common & para_set)]
    cores = Fi.core.astype(str).to_numpy()
    comp = []

    def errs(k, t):
        f = frames[k]
        q = f[f.target == t].set_index("pair_id").reindex(Fi.index)
        return (q.yhat - q.y).to_numpy(float)
    for x in [k for k in frames if k not in ("dft#model", "lf#model") and frames[k].pair_id.nunique() == len(ids_common)]:
        for b in ("lf#model", "dft#model"):
            for t in TARGETS:
                ex, eb = errs(x, t), errs(b, t)
                if not (np.isfinite(ex).all() and np.isfinite(eb).all()):
                    continue
                d, lo, hi, p = mods()["EG"].boot_diff(ex, eb, cores)
                comp.append(dict(x=x, b=b, target=t, pos="p", set="P_main", n=len(ex), n_cores=len(set(cores)),
                                 mae_x=float(np.abs(ex).mean()), mae_b=float(np.abs(eb).mean()), d=d, ci_lo=lo,
                                 ci_hi=hi, p=p, ratio=float(np.abs(ex).mean() / np.abs(eb).mean())
                                 if np.abs(eb).mean() > 0 else None))
    Cmp = pd.DataFrame(comp)
    # per-pair errors for W / P (all common pairs, every target; dd of the pair from the main geometry)
    ddm = cls.get("main_dd", {})
    Pt = S.loc[sorted(ids_common)]
    PE = []
    for t in TARGETS:
        cols = {}
        for k in ("dft#model", "lf#model", prim_key):
            f = frames[k]
            cols[k] = f[f.target == t].set_index("pair_id").reindex(Pt.index)
        y = cols["dft#model"].y.to_numpy(float)
        PE.append(pd.DataFrame(dict(
            pair_id=Pt.index, fold=cols["dft#model"].fold.to_numpy(), a_rxn=Pt.a_rxn.astype(int).to_numpy(),
            b_rxn=Pt.b_rxn.astype(int).to_numpy(), pos=Pt.pos.to_numpy(), ref_kind=Pt.ref_kind.to_numpy(),
            in_P_main=Pt.in_P_main.astype(bool).to_numpy(), para_main=Pt.index.isin(list(para_set)), target=t, y=y,
            yhat_dft=cols["dft#model"].yhat.to_numpy(float), yhat_lf=cols["lf#model"].yhat.to_numpy(float),
            yhat_mlts=cols[prim_key].yhat.to_numpy(float),
            err_mlts=cols[prim_key].yhat.to_numpy(float) - y,
            dd_a=[ddm.get(int(a), np.nan) for a in Pt.a_rxn], dd_b=[ddm.get(int(b), np.nan) for b in Pt.b_rxn],
            cls_same_ts=[pair_class(a, b, cls.get("main", {})) for a, b in zip(Pt.a_rxn, Pt.b_rxn)],
            cls_g1=[pair_class(a, b, cls["g1"]) for a, b in zip(Pt.a_rxn, Pt.b_rxn)],
            dipole=Pt.dipole.astype(str).to_numpy())))
    PE = pd.concat(PE, ignore_index=True)
    PE["dd_pair"] = np.fmax(PE.dd_a.to_numpy(float), PE.dd_b.to_numpy(float))
    gr = grade_of(MI, rates, prim_key, "lf#model")
    gr_model = grade_of(MI, rates, f"{prim}#model", "lf#model")
    return dict(n=n, flags=flags, frames=frames, roles=roles, Mt=Mt, MI=MI, Bd=Bd, rates=rates, fams=fams, ah=ah,
                auc_diffs=diffs, rung_tabs=rung_tabs, rung_goal=rung_goal, Mrows=Mrows, Cmp=Cmp, PE=PE, grade=gr,
                grade_model_disp=gr_model, prim_key=prim_key, sel=sel, rep8_dft=rep8_d, d3info=d3info,
                ctx_info=dict(fu2_evaluate=ctx["fu2_evaluate"], labels=ctx["labels"]), common=common, cfgs=cfgs,
                para_set=para_set)


def geometry_summary(info):
    pd, np = mods()["pd"], mods()["np"]
    out = {}
    for tag, root, eng, a4 in geom_roots(info):
        c, j = geom_paths(tag)
        G_ = pd.read_csv(c)
        ok = G_[G_.status.astype(str) == "ok"]
        same = ok.same_ts.astype(str).str.lower() == "true"

        def q(s, p):
            s = pd.to_numeric(s, errors="coerce").dropna()
            return float(np.percentile(s, p)) if len(s) else None

        def cost(col):                                             # A2-cost: allocated when logged, else own
            own = pd.to_numeric(G_[col], errors="coerce")
            al = f"{col}_alloc"
            return pd.to_numeric(G_[al], errors="coerce").fillna(own) if al in G_ else own
        out[tag] = dict(root=str(root), engine=eng, n_folders=int(len(G_)), n_ok=int(len(ok)),
                        core_h_own_median=q(G_.core_h, 50), gpu_min_own_median=q(G_.gpu_min, 50),
                        n_targets=9266, same_ts_frac_of_targets=float(same.sum()) / 9266,
                        same_ts_frac_of_ok=float(same.mean()) if len(ok) else None,
                        dd_median_same=q(ok[same].dd_max, 50), dd_median_ok=q(ok.dd_max, 50), dd_p90_ok=q(ok.dd_max, 90),
                        rmsd_median_same=q(ok[same].rmsd_heavy, 50), rmsd_median_ok=q(ok.rmsd_heavy, 50),
                        nimag1_frac_of_ok=float((pd.to_numeric(ok.n_imag, errors="coerce") == 1).mean()) if len(ok)
                        else None, core_h_median=q(cost("core_h"), 50), gpu_min_median=q(cost("gpu_min"), 50),
                        status=G_.status.astype(str).value_counts().to_dict(), table=str(c), table_sha256=fsha(c))
    return out


def prereg_check(info, R):
    reasons = []
    pj = info["prereg"]
    regs = [x.get("rule") for x in ((pj.get("criteria") or {}).get("grades") or [])][:2]
    if regs != registered_rules():
        reasons.append(f"registered grade rules {regs} != the code constants {registered_rules()}")
    prim = info["primary"]
    reg_refs = ((pj.get("criteria") or {}).get("refs") or {}).get("primary")
    used = info["configs"][prim].get("refs")
    if reg_refs and not used:
        reasons.append(f"the primary {prim}'s refs mode is not recorded (K_report configs / C1_report.json): the "
                       f"registered refs.primary {reg_refs} cannot be verified")
    elif used and reg_refs and used != reg_refs:
        reasons.append(f"primary {prim} predicted with refs {used} != registered refs.primary {reg_refs}")
    if R["n"]["common_para"] == 0:
        reasons.append("the common P_main para set is empty")
    return reasons


def cmd_run(no_figures=False, no_rung_part=False):
    info0 = readiness()
    for tag, root, eng, a4 in geom_roots(info0):
        if not geom_fresh(tag, root, a4):
            rc = cmd_geom(root, tag, eng) if os.environ.get("MLTENV") == "d1" else \
                run_in_env("d1", ["geom", "--tag", tag, "--engine", eng, "--root", str(root)])
            if rc != 0:
                return rc
            if not geom_fresh(tag, root, a4):
                M.stop(f"structure table {tag} still stale after geom")
    ensure_espley()
    info = readiness()
    info["skip_rung_part"] = bool(no_rung_part)
    np, pd = mods()["np"], mods()["pd"]
    R = build(info)
    reasons = prereg_check(info, R)
    outs = {}
    LE = mods()["LE"]

    def wcsv(key, df):
        p = M.write_csv(M.RES / OUT[key], df)
        outs[str(p)] = M.sha256(p)
    wcsv("metrics", R["Mrows"])
    Mp = R["Mt"].copy()                                            # E-pos: every position (p, m, o, direct, all) and
    Mp.insert(0, "disp_source", Mp.rung.astype(str).str.split("#").str[1])   # set of the LADDER metrics_e1 rows
    Mp.insert(0, "case", Mp.rung.astype(str).str.split("#").str[0])
    wcsv("pos", Mp)
    wcsv("compare", R["Cmp"])
    Bd = R["Bd"].copy()
    Bd.insert(0, "disp_source", Bd.rung.str.split("#").str[1])
    Bd.insert(0, "case", Bd.rung.str.split("#").str[0])
    wcsv("bends", Bd)
    if R["rung_tabs"]:
        wcsv("ladder", pd.concat(R["rung_tabs"], ignore_index=True))
        wcsv("goal", pd.concat(R["rung_goal"], ignore_index=True))
    rates = {k: dict(R["rates"][f"{k}|J|sel|common"], auc_hidden=R["ah"][k], role=R["roles"][k]) for k in R["frames"]}
    p = M.write_json(M.RES / OUT["rates"], LE._clean(dict(
        key="case#disp_source (single frame on the common pairs)", rates=rates, auc_differences=R["auc_diffs"],
        families=R["fams"], rules=dict(bends=DECISIONS["E-frames"], hidden=DECISIONS["E-hidden"]))))
    outs[str(p)] = M.sha256(p)
    M.write_csv(EVAL / "E_pairs.csv", R["PE"][R["PE"].target == "oi"][["pair_id", "fold", "a_rxn", "b_rxn", "pos",
                                                                       "ref_kind", "para_main", "dipole", "cls_same_ts",
                                                                       "cls_g1", "dd_pair"]])
    M.write_parquet(EVAL / "E_pair_errors.parquet", R["PE"])
    figs = {}
    if not no_figures:
        figs = figures(R["Mrows"], dict(primary=info["primary"], primary_key=R["prim_key"], n_para=R["n"]["common_para"],
                                        pair_err=R["PE"][R["PE"].para_main]),
                       {k: M.RES / OUT[k] for k in ("fig1", "fig2")})
        for v in figs.values():
            outs[v] = M.sha256(v)
    q = R["Mrows"][(R["Mrows"].split == "all")]

    def para(k):
        case, disp = k.split("#")
        r = q[(q.case == case) & (q.disp_source == disp)].set_index("target")
        return {t: dict(mae=float(r.at[t, "mae"]), ci=[float(r.at[t, "mae_ci_lo"]), float(r.at[t, "mae_ci_hi"])],
                        n=int(r.at[t, "n"])) for t in TARGETS if t in r.index}
    pk = R["prim_key"]
    hid = (R["ah"][pk] or {}).get("hidden") or {}
    R["flags"].append(f"hidden bends on {hid.get('n_fam')} bends_e1 P_main para families of the common pairs (dft / lf "
                      f"/ MLTS alike); DEPLOY's published lf value used its 89 P_sens families (not comparable)")
    gm = R["grade"]["goal"]["mae"]
    head = dict(grade=R["grade"]["grade"], grade_ko=R["grade"]["grade_ko"], oi_para_mae=gm["oi"]["para_mae"],
                cpcm_para_mae=gm["cpcm"]["para_mae"], oi_ci=gm["oi"]["ci"], cpcm_ci=gm["cpcm"]["ci"],
                hidden_bend_ppv=(hid.get("ppv") or {}).get("value"), hidden_bend_ppv_ci=(hid.get("ppv") or {}).get("ci"),
                hidden_bend_sens=(hid.get("sens") or {}).get("value"), hidden_n_fam=hid.get("n_fam"))
    by_cls = {}
    for sp in ("same_ts", "g1"):
        z = R["Mrows"][(R["Mrows"].split == sp) & (R["Mrows"].case == info["primary"]) &
                       (R["Mrows"].disp_source == "D3_int") & R["Mrows"].target.isin(["oi", "cpcm"])]
        by_cls[sp] = {f"{r.group}|{r.target}": dict(n=int(r.n), mae=float(r.mae)) for r in z.itertuples()}
    rep = dict(
        phase="DEPLOY-V2 E (spec 11)", complete=True, n_todo=0, stop=None, est=dict(units_total=1, unit_wall_h=0.5,
                                                                                   workers=1),
        grade=R["grade"]["grade"], grade_ko=R["grade"]["grade_ko"], prereg_change_needed=bool(reasons),
        prereg_change_reason="; ".join(reasons) if reasons else None, headline=head, primary=info["primary"],
        primary_key=pk, configs={k: {kk: v for kk, v in c.items()} for k, c in R["cfgs"].items()},
        representatives={k: dict(REP8={t: list(v) for t, v in s[1].items()}, how=s[2]) for k, s in R["sel"].items()},
        dft_REP8={t: list(v) for t, v in R["rep8_dft"].items()}, n=R["n"], flags=R["flags"],
        grade_detail=R["grade"], grade_model_disp=dict(grade=R["grade_model_disp"]["grade"],
                                                       partial=R["grade_model_disp"]["partial"]),
        para_mae={k: para(k) for k in R["frames"]},
        disp_both={k.split("#")[0]: dict(model=para(f"{k.split('#')[0]}#model").get("disp"),
                                         D3_int=para(k).get("disp")) for k in R["frames"] if k.endswith("#D3_int")},
        d3=R["d3info"], hidden={k: (v or {}).get("hidden") for k, v in R["ah"].items()},
        auc_g={k: {c: x.get("auc_g") for c, x in (v or {}).get("targets", {}).items()} for k, v in R["ah"].items()},
        auc_differences=R["auc_diffs"], by_class_oi_cpcm=by_cls, geometry=geometry_summary(info),
        arms=info["arms"], figures=figs, outputs=outs, decisions=DECISIONS, ladder_fu2=R["ctx_info"],
        inputs={str(x): fsha(x) for x in [PREREG_MD, PREREG_JS, K_REPORT, ARMS_JSON, L_PRED, L_REPORT, LF_PRED,
                                          M.C_COMPARE] + [Path(c["predictions"]) for c in R["cfgs"].values()]},
        code=dict(eval_v2=M.sha256(THIS), version=CODE_VERSION), written=M.now(), seconds=time.time() - T0)
    M.write_json(M.RES / OUT["report"], LE._clean(rep))
    log(f"E: grade {rep['grade']} ({rep['grade_ko']}); oi {head['oi_para_mae']}, cpcm {head['cpcm_para_mae']}, "
        f"hidden PPV {head['hidden_bend_ppv']}; prereg change {rep['prereg_change_needed']} -> {M.RES / OUT['report']}")
    return 0


# ================================================================ selftest (LADDER / DEPLOY files only)
def cmd_selftest():
    ensure_espley()
    np, pd = mods()["np"], mods()["pd"]
    LE, ED2 = mods()["LE"], mods()["ED2"]
    out_dir = EVAL / "selftest"
    ck, det = {}, {}
    # (1) dd_disp = disp(a) - disp(b) on the label file
    lab = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
    Fp = pd.read_csv(M.LAD / "F_pairs_cr2.csv", usecols=["pair_id", "a_rxn", "b_rxn", "dd_disp"])
    v = np.array([lab[int(a)]["disp_dft"] - lab[int(b)]["disp_dft"] for a, b in zip(Fp.a_rxn, Fp.b_rxn)], float)
    det["dd_disp_a_minus_b_max_abs"] = float(np.nanmax(np.abs(v - Fp.dd_disp.to_numpy(float))))
    ck["dd_disp = disp(a) - disp(b)"] = det["dd_disp_a_minus_b_max_abs"] <= 1e-9
    # (2) dft REP8 frames on the full dft set (C1 := dft relabelled, D3 := label disp) through build()
    La = pd.read_parquet(L_PRED, columns=PRED_COLS)
    Pd = La[La.rung == "dft"].copy()
    tmp_pred = out_dir / "C1_predictions.parquet"
    M.write_parquet(tmp_pred, Pd.assign(rung="mlts"))
    rep_d, rep8_d = dft_reps()
    Kd = keys(norm_pred(Pd, "dft"))
    info = dict(K=dict(complete=True, primary="C1", reps={"C1": dict(REP8={t: list(v) for t, v in rep8_d.items()},
                                                                    REP={a: {t: rep_d[(a, t)] for t in TARGETS}
                                                                         for a in ARMS8})}),
                prereg=dict(criteria=dict(grades=[dict(rule=r) for r in registered_rules()], refs=dict(primary="min"))),
                configs={"C1": dict(name="C1", predictions=str(tmp_pred), refs="min", table=None, units_dir=None,
                                    app=False, rung="mlts")},
                primary="C1", arms=dict(engine="A"), app_ok=False, flags=[],
                d3_override={"C1": {r: float(d["disp_dft"]) for r, d in lab.items()}},
                cls_override={"main": {}, "main_dd": {}, "g1": {}}, skip_rung_part=False)
    R = build(info, restrict_to=Kd, lf_on_own=True)
    E1 = pd.read_csv(LAD_RES / "E1_metrics.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    E1 = E1[(E1.rung == "dft") & (E1.pair_set == "own") & (E1.set == "P_main") & (E1.pos == "p")]
    BR = M.read_json(LAD_RES / "E1_bend_rates.json")["rates"]
    dm, dr = 0.0, 0.0
    for t in TARGETS:
        a, m = rep8_d[t]
        e = E1[(E1.arm == a) & (E1.model == m) & (E1.target == t)].iloc[0]
        x = R["MI"][("dft#model", "J", "sel", "common", t, "P_main", "p")]
        dm = max(dm, abs(x["mae"] - e.mae), abs(x["mae_ci_lo"] - e.mae_ci_lo), abs(x["mae_ci_hi"] - e.mae_ci_hi),
                 abs(int(x["n"]) - int(e.n)))
        if t in LE.PRIMARY9:
            b1 = BR[f"dft|{a}|{m}|own"]["all"][t]
            b2 = R["rates"]["dft#model|J|sel|common"]["all"][t]
            for kk in ("sens", "fpr", "n_fam", "n_bent_dft", "n_straight_dft"):
                u, w = b1.get(kk), b2.get(kk)
                nu, nw = not _fin(u), not _fin(w)
                dr = max(dr, (0.0 if nu == nw else 1.0) if (nu or nw) else abs(float(u) - float(w)))
    det["dft_rep8_vs_E1_metrics_max_abs"], det["dft_rep8_vs_E1_bend_rates_max_abs"] = dm, dr
    ck["dft REP8 single frame = E1_metrics (MAE, CI, n; 1e-9)"] = dm <= 1e-9
    ck["dft REP8 single frame = E1_bend_rates (sens, FPR, n; 1e-12)"] = dr <= 1e-12
    # (3) C1 = dft -> MLTS(model disp) == dft, MLTS(D3 = label disp) disp MAE == 0
    z = R["Cmp"][(R["Cmp"].x == "C1#model") & (R["Cmp"].b == "dft#model")]
    ck["C1 := dft: d = 0 and CI = [0, 0] vs dft (every target)"] = bool(len(z) == len(TARGETS) and (z.d == 0).all()
                                                                      and (z.ci_lo == 0).all() and (z.ci_hi == 0).all())
    dd3 = R["MI"][("C1#D3_int", "J", "sel", "common", "disp", "P_main", "p")]["mae"]
    det["C1 D3 := label disp: disp para MAE"] = dd3
    ck["D3 disp from the label disp: MAE <= 1e-12"] = dd3 <= 1e-12
    # (4) lf on its own set = DEPLOY E_metrics (lf, own, P_main, p)
    DM = pd.read_csv(DEP_E_METRICS)
    DM = DM[(DM.model == "lf") & (DM.pair_set == "own") & (DM.eval_set == "P_main") & (DM.pos == "p")].set_index("target")
    dl = max(abs(R["MI"][("lf#model", "J", "sel", "common", t, "P_main", "p")]["mae"] - DM.at[t, "mae"]) for t in TARGETS)
    det["lf_own_vs_DEPLOY_E_metrics_max_abs"] = dl
    ck["lf own para MAE = DEPLOY E_metrics (1e-9)"] = dl <= 1e-9
    # (5) grade rule unit cases on synthetic metric dicts
    g = R["grade"]
    det["grade_C1_eq_dft"] = g["grade"]
    det["grade_detail_C1_eq_dft"] = dict(goal_mae={t: v["ok"] for t, v in g["goal"]["mae"].items()},
                                         goal_bend={t: v["ok"] for t, v in g["goal"]["bend"].items()})
    ladder_dftSH = dict(oi=0.102, cpcm=0.060)
    ck["grade(C1 = dft) has oi / cpcm MAE ok (LADDER dft·SH 0.102 / 0.060 <= 0.13)"] = all(
        g["goal"]["mae"][t]["ok"] for t in ladder_dftSH)
    rep_out = dict(checks=ck, detail=det, all_pass=all(ck.values()), n=R["n"], written=M.now(),
                   seconds=time.time() - T0, rung_part_rows=int(sum(len(x) for x in R["rung_tabs"])))
    M.write_json(out_dir / "E_selftest.json", LE._clean(rep_out))
    for k, v in ck.items():
        log(f"selftest {k}: {'ok' if v else 'FAIL'}")
    del ED2
    return 0 if rep_out["all_pass"] else 3


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="eval_v2.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-rung-part", action="store_true", help="tests only: skip the descriptive LADDER rung tables")
    p = sub.add_parser("geom")
    p.add_argument("--root", default=None)
    p.add_argument("--tag", default="main")
    p.add_argument("--engine", default="A")
    sub.add_parser("selftest")
    o = ap.parse_args(argv)
    if o.cmd == "run":
        return cmd_run(o.no_figures, o.no_rung_part)
    if o.cmd == "geom":
        return cmd_geom(o.root, o.tag, o.engine)
    if o.cmd == "selftest":
        return cmd_selftest()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
