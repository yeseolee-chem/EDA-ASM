#!/usr/bin/env python3
"""deploy_common.py — GEO2b DEPLOY (the LADDER dft-rung energy models applied to the GEO2b geo-rung inputs): shared paths,
constants, writers, the deploy pair rule and the regression comparison. numpy / pandas only, so it loads in env_d1fu2
(predict_atoms.py) and env_espley (predict_xgb.py, compat_deploy.py, eval_deploy.py). Rules: analysis/geo2/results/
DECISIONS_deploy.md (DP-1 .. DP-6). sbatch only; nothing in this directory was run when it was written (login node).

Writes only under $GEO2B/deploy (= $GEO2_SCRATCH/deploy) and analysis/geo2/results/DEPLOY_* (assert_out). Every LADDER
($LAD, analysis/d1_ladder) and GEO2b ($GEO2B, analysis/geo2/results) file is read only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent                       # analysis/geo2/deploy
GEO2_DIR = HERE.parent                                       # analysis/geo2
REPO = HERE.parents[2]
LAD_DIR = REPO / "analysis" / "d1_ladder"
RES = GEO2_DIR / "results"                                   # outputs: DEPLOY_* only
LAD_RES = LAD_DIR / "results"
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2b"))       # $GEO2B
LAD = Path(os.environ.get("D1LADDER_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder"))  # $LAD (read only)
OUT = GEO2 / "deploy"
PARTS = OUT / "parts"
PART_ATOMS = PARTS / "atoms.parquet"                          # predict_atoms.py (rung dft_on_geo, AW / GNN / TF)
PART_XGB = PARTS / "xgb.parquet"                              # predict_xgb.py (rung dft_on_geo, XGB)
DEPLOY_PRED = OUT / "L_predictions_deploy.parquet"            # eval_deploy.py collect
COMMON_CSV = OUT / "common_pairs.csv"                         # compat_deploy.py (pair_id, fold, ...)
REG_ATOMS = RES / "DEPLOY_regress_atoms.json"
REG_XGB = RES / "DEPLOY_regress_xgb.json"
COMPAT_JSON = RES / "DEPLOY_compat.json"
COLLECT_JSON = RES / "DEPLOY_collect.json"
DECISIONS_MD = RES / "DECISIONS_deploy.md"
L_PRED = LAD_RES / "L_predictions.parquet"                    # LADDER (a)
L_REPORT = LAD_RES / "L_report.json"
L_XGB_REPORT = LAD_RES / "L_xgb_report.json"
GEO_PRED = LAD_RES / "L_predictions_geo.parquet"              # GEO2b (c)
GEO_REPORT = LAD_RES / "L_report_geo.json"
Z_STATUS = RES / "Z_status.csv"                               # GEO2b Phase Z: rxn_id, src, source (f0-f4 | fm1)

SRC_RUNG, IN_RUNG, RUNG_DEPLOY = "dft", "geo", "dft_on_geo"   # model rung, input rung, label of (b)
CASES = {"a": SRC_RUNG, "b": RUNG_DEPLOY, "c": IN_RUNG}
TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
NT = len(TARGETS)
ARMS = ("S", "SH")
ATOM_MODELS = ("AW", "GNN", "TF")
MODELS = ATOM_MODELS + ("XGB",)                               # LADDER 13-1 tie order
NF = 5
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]                                  # = train_atoms.PRED_COLS (L_predictions.parquet)
TOL_EXACT = 1e-6        # DP-2: |yhat - L_predictions yhat| (kcal/mol), the task gate
TOL_FP32 = 1e-4         # DP-2: documented GPU float32 bound (= fu2_common ANTISYM_TOL, gate M-1'), atom models only
Y_TOL = 1e-12           # y / base identity (= LADDER L-11 tolerance)
CODE_FILES = ("deploy_common.py", "predict_atoms.py", "predict_xgb.py", "compat_deploy.py", "eval_deploy.py")


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def log(t0, msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha():
    return {n: sha256(HERE / n) for n in CODE_FILES if (HERE / n).is_file()}


def read_json(p):
    return json.loads(Path(p).read_text())


def clean(o):
    """JSON-able copy: numpy -> python, NaN / inf -> None, tuples -> lists, Path -> str (= LADDER evaluate._clean)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [clean(v) for v in (sorted(o) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def assert_out(p):
    """DEPLOY writes: under $GEO2B/deploy, or analysis/geo2/results/DEPLOY_*; exit 3 otherwise."""
    q = Path(os.path.abspath(p))
    out = Path(os.path.abspath(OUT))
    if q == out or out in q.parents:
        return q
    if q.parent == Path(os.path.abspath(RES)) and q.name.startswith("DEPLOY_"):
        return q
    stop(f"write path {q} outside {OUT} and {RES}/DEPLOY_*")


def write_atomic(path, data):
    """data = bytes / str, or a callable writing the temporary path (e.g. lambda t: df.to_parquet(t, index=False))."""
    path = assert_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if callable(data):
        data(tmp)
    else:
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    return write_atomic(path, json.dumps(clean(obj), indent=1))


def write_parquet(path, df):
    return write_atomic(path, lambda t: df.to_parquet(t, index=False))


def write_csv(path, df):
    return write_atomic(path, df.to_csv(index=False))


def maxdiff(a, b):
    """max |a - b| with equal NaN patterns required (inf when shapes / NaN patterns differ; 0 for empty)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


def env_info():
    return dict(python=platform.python_version(), host=platform.node(), prefix=sys.prefix,
                slurm_job=os.environ.get("SLURM_JOB_ID"), date=time.strftime("%F %T"))


# ---------------------------------------------------------------- the deploy pair rule (DP-4)
def deploy_rows(Dd, Dg, te_rows):
    """dft unit test rows (pair-table rows of the dft data object Dd: train_atoms.LadData or train_xgb.XData) ->
    (keep [len(te_rows)] bool, geo pair-table rows of the kept pairs, in the dft test order). keep = both ends usable in
    the geo rung (Dg.usable_pair: token + input-table row + finite H 9 + finite 182, the trainers' rule)."""
    te_rows = np.asarray(te_rows, dtype=int)
    pids = np.asarray(Dd.P.index)[te_rows].astype(str)
    miss = [p for p in pids if p not in Dg.pid_row]
    if miss:
        stop(f"{len(miss)} dft test pairs are not in the geo label file, e.g. {miss[:3]}")
    g = np.array([Dg.pid_row[p] for p in pids], dtype=int)
    keep = np.asarray(Dg.usable_pair, dtype=bool)[g] if len(g) else np.zeros(0, dtype=bool)
    return keep, g[keep]


def check_same_pairs(Dd, Dg, te_rows, keep, rows_g):
    """the kept dft test pairs and their geo rows are the same pairs (id, ends, ref_kind, pos) with the same y."""
    td = np.asarray(te_rows, dtype=int)[keep]
    Pd, Pg = Dd.P, Dg.P
    bad = []
    if (np.asarray(Pd.index)[td].astype(str) != np.asarray(Pg.index)[rows_g].astype(str)).any():
        bad.append("pair ids")
    for c in ("a_rxn", "b_rxn", "ref_kind", "pos"):
        if (Pd[c].to_numpy()[td].astype(str) != Pg[c].to_numpy()[rows_g].astype(str)).any():
            bad.append(c)
    dy = maxdiff(Dd.y[td], Dg.y[rows_g])
    if dy > Y_TOL:
        bad.append(f"y (max {dy:.3g})")
    if bad:
        stop(f"dft test pairs and their geo rows differ in {bad}")
    return dy


# ---------------------------------------------------------------- regression comparison (DP-2)
def compare_unit(mine, ref, cols=("y", "yhat", "base", "beta", "yhat_seed_sd")):
    """one unit: recomputed rows `mine` vs reference rows `ref` (PRED_COLS), matched on (target, pair_id); same pair set
    per target and the same order -> dict(n, same_pairs, same_order, max_abs_<col>)."""
    a = mine.assign(pair_id=mine.pair_id.astype(str)).reset_index(drop=True)
    b = ref.assign(pair_id=ref.pair_id.astype(str)).reset_index(drop=True)
    ka, kb = list(zip(a.target, a.pair_id)), list(zip(b.target, b.pair_id))
    out = dict(n=int(len(a)), n_ref=int(len(b)))
    same = len(set(ka)) == len(ka) and len(set(kb)) == len(kb) and set(ka) == set(kb)
    out["same_pairs"] = bool(same)
    out["same_order"] = bool(same and all(a.pair_id[a.target == t].tolist() == b.pair_id[b.target == t].tolist()
                                          for t in TARGETS))
    if not same:
        out.update({f"max_abs_{c}": float("inf") for c in cols})
        return out
    bi = b.set_index(["target", "pair_id"]).loc[ka]
    for c in cols:
        out[f"max_abs_{c}"] = maxdiff(a[c].to_numpy(float), bi[c].to_numpy(float))
    return out


def gate_status(dmax, allow_fp32):
    """DP-2: 'exact' (<= 1e-6), 'within_fp32_bound' (<= 1e-4, atom models only), else 'fail'."""
    if np.isfinite(dmax) and dmax <= TOL_EXACT:
        return "exact"
    if allow_fp32 and np.isfinite(dmax) and dmax <= TOL_FP32:
        return "within_fp32_bound"
    return "fail"


def fingerprint_diff(old, now, allowed_keys, allowed_code):
    """unit json fingerprint vs the current one: keys that differ, code files that differ, and whether the difference is
    only the allowed one (DP-2)."""
    old, now = old or {}, now or {}
    keys = sorted(k for k in set(old) | set(now) if old.get(k) != now.get(k))
    co, cn = old.get("code_sha256") or {}, now.get("code_sha256") or {}
    code = sorted(k for k in set(co) | set(cn) if co.get(k) != cn.get(k))
    ok = not (set(keys) - set(allowed_keys)) and not (set(code) - set(allowed_code))
    return dict(keys=keys, code_files=code, ok=bool(ok))


def pred_frame(D, rows, u, Y, base, beta, rung):
    """unit prediction rows (= train_atoms.run_unit / train_xgb.run_unit output layout, PRED_COLS) for pair-table rows of
    the data object D; Y [seeds, n, 10] kcal/mol (mean = yhat, SD ddof 0 = yhat_seed_sd), base [n, 10], beta [10]."""
    import pandas as pd
    rows = np.asarray(rows, dtype=int)
    Y = np.asarray(Y, dtype=float)
    pid = np.asarray(D.P.index)[rows].astype(str)
    rk, ps = D.P.ref_kind.to_numpy()[rows], D.P.pos.to_numpy()[rows]
    parts = [pd.DataFrame(dict(pair_id=pid, ref_kind=rk, pos=ps, fold=int(u["fold"]), rung=rung, arm=u["arm"],
                               model=u["model"], target=t, y=D.y[rows, j], yhat=Y[:, :, j].mean(0), base=base[:, j],
                               beta=float(beta[j]), yhat_seed_sd=Y[:, :, j].std(0))) for j, t in enumerate(TARGETS)]
    return pd.concat(parts, ignore_index=True)[PRED_COLS]


def ladder_rows(Lp, arm, model, fold):
    """the L_predictions.parquet rows of one dft unit (Lp = the dft rows, pair_id str, fold int)."""
    return Lp[(Lp.arm == arm) & (Lp.model == model) & (Lp.fold == int(fold))]


def load_ladder_dft(models):
    """L_predictions.parquet dft rows of the given models (sha256 = L_report.json predictions_sha256, else STOP)."""
    import pandas as pd
    rep = read_json(L_REPORT)
    if sha256(L_PRED) != rep.get("predictions_sha256"):
        stop(f"{L_PRED} changed after {L_REPORT.name} (predictions_sha256)")
    Lp = pd.read_parquet(L_PRED, columns=PRED_COLS)
    Lp = Lp[(Lp.rung == SRC_RUNG) & Lp.model.isin(list(models))].copy()
    Lp["pair_id"] = Lp.pair_id.astype(str)
    Lp["fold"] = Lp.fold.astype(int)
    return Lp, dict(path=str(L_PRED), sha256=rep.get("predictions_sha256"), n_rows=int(len(Lp)))
