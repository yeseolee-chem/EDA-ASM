#!/usr/bin/env python3
"""select_blocks.py — rev 5 Phase C (docs/specs/REV5_FEATURES_FIGURES.md C-1 .. C-3): lockbox / dev split, dev-CV
evaluation of the feature blocks B1..B6 and forward block selection (EXT_SEL). Every constant, arm and block column
comes from rev5_common; the model pipeline and grid from train_ml_single (_make_pipe, GRIDS['KRR_rbf']).

  python select_blocks.py lockbox           C-1 (r5_lockbox.sh)
  python select_blocks.py select [--check]  C-2 / C-3 (r5_select.sh); --check validates inputs + caches, fits nothing
  python select_blocks.py status            progress: lockbox, cached units, selection path so far, prereg_rev5b.json

lockbox  ids = rxn_id of results_rev5/rows_rev5.csv sorted ascending (must be a subset of rows_rev4.csv);
         lockbox = sorted(np.random.default_rng(SEED).choice(ids, size=round(LOCKBOX_FRAC * len(ids)), replace=False)),
         dev = the other ids, ascending -> results_rev5/lockbox_ids.csv, dev_ids.csv (one column rxn_id) and
         lockbox.json (n, sha256 of rows_rev5.csv and both files, rule). A missing file is written (atomic); an
         existing one is never rewritten: if it differs from the recomputation -> exit 3, nothing written.
select   Checks the lockbox files against lockbox.json (sha256) and the recomputed split (exit 3 on any difference).
         Rows: train_ml_single.load_ml(G1 parquet + rev5_common.FEAT_EXT, rows = dev_ids.csv, the 9 targets) = exactly
         the dev rows in ascending rxn_id order, gated (ok, NaN-free, hygiene; exit on any bad row), then asserted to be
         dev_ids.csv and disjoint from lockbox_ids.csv before any X / y exists. No lockbox row is ever in X / y.
         Outer CV: KFold(N_OUTER, shuffle=True, random_state=SEED) over the dev rows. Unit = (arm, target, outer fold):
         GridSearchCV(_make_pipe(KernelRidge rbf), GRIDS['KRR_rbf'], cv=KFold(N_INNER, shuffle=True,
         random_state=SEED), neg MAE) on the fold's train rows, scored on its test rows (MAE, NMAE = rev5_common.nmae,
         r2) -> $R5_SCRATCH/select/units/<arm>/<target>__fold<k>.json (atomic; reused when present, exit 3 if it was
         made from other inputs: dev rows, feature files, columns, grid, folds, numpy / scipy / sklearn versions).
         A unit also stores best params, n_train / n_test, its test ids (`test_rxn_id`) and predictions (`yhat`);
         $R5_SCRATCH/select/dev_folds.csv holds rxn_id -> outer fold. Every id under $R5_SCRATCH/select is a dev id
         (final_eval.py audits this, D-1).
         Stage 1: BASE (= ESPLEY73), BASE+Bk (k = B1..B6), EXT_ALL on all 9 TARGETS9 (C-2 report).
         Forward selection (C-1 rule): criterion = mean over SEL_TARGETS of the fold NMAE, averaged over the folds;
         each step evaluates current + each remaining block on SEL_TARGETS ('BASE+B2+B5' arm names, blocks in
         BLOCK_ORDER), takes the lowest criterion (ties: BLOCK_ORDER) and accepts it iff the relative decrease is
         >= SEL_MIN_REL AND its fold criterion is lower than the current arm's in >= SEL_MIN_FOLDS folds, else stops.
         No block accepted -> EXT_SEL = BASE (not a STOP). EXT_SEL is then also evaluated on the other 6 targets
         (report rows only; the selection never looks at them).
         Units run in one ProcessPoolExecutor (spawn), R5_SEL_PAR at a time, each GridSearchCV with R5_SEL_NJOBS
         workers; a failing unit stops the run (the running ones finish and are cached, the rest is cancelled).
         Outputs (results_rev5/, atomic):
           C2_block_cv_folds.csv   every unit: arm, alias_of, set, blocks, n_features, target, fold, n_train, n_test,
                                   mae, nmae, r2, best_inner_mae, best_<param>, edge_hits, n_jobs, seconds
           C2_block_cv.csv         per (arm, target): n_folds, mae / nmae mean and sample SD (ddof 1) over the folds,
                                   r2_mean, differences to BASE and the number of folds with NMAE below BASE's
           (set = stage1 | forward | EXT_SEL; the EXT_SEL rows copy the selected arm, alias_of = its name)
           C2_selection_path.csv   step, current, candidate, added_block, n_features, criterion, criterion_current,
                                   rel_decrease, folds_lower, best, accepted, decision, crit_fold<k>, nmae_<target>
           C2_selection_path.json  the same rows + rule, inputs, folds, final EXT_SEL, unit timing, versions
           C2_feature_target_r.csv Pearson r of each of the 109 block features with the SEL_TARGETS, dev rows only
           prereg_rev5b.json       C-3: ext_sel_blocks, n_features, criterion values, rule, inputs. Written last and
                                   only once the path is complete; if it exists and differs -> exit 3 before any
                                   output is written (it is never rewritten).
status   Prints the lockbox state, the stage-1 unit matrix, the selection path replayed from the cached units (keys
         not re-validated) and prereg_rev5b.json. Never fits; exit 0.

env  R5_SCRATCH (rev5_common.SCRATCH), ESPLEY_OUT (feature parquets, rev5_common.FEAT_G1 / FEAT_EXT),
     SLURM_CPUS_PER_TASK, R5_SEL_NJOBS (GridSearchCV n_jobs per unit, default 1),
     R5_SEL_PAR (units in parallel, default SLURM_CPUS_PER_TASK // R5_SEL_NJOBS).
exit 0 ok, 1 a unit failed, 2 missing / invalid input (incl. the train_ml_single.load_ml row / feature gates),
     3 an existing file differs (lockbox, caches, prereg_rev5b.json).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before numpy: one BLAS thread each
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev5_common as R5  # noqa: E402
import train_ml_single as TM  # noqa: E402

# ---------------------------------------------------------------- files
ROWS5 = R5.ROWS5
LOCK_CSV = R5.LOCKBOX
DEV_CSV = R5.RES5 / "dev_ids.csv"
LOCK_JSON = R5.RES5 / "lockbox.json"
PREREG = R5.PREREG5B_JSON
SEL_DIR = R5.SCRATCH / "select"
UNIT_DIR = SEL_DIR / "units"
FOLDS_CSV = SEL_DIR / "dev_folds.csv"
OUT_FOLDS = R5.RES5 / "C2_block_cv_folds.csv"
OUT_CV = R5.RES5 / "C2_block_cv.csv"
OUT_PATH_CSV = R5.RES5 / "C2_selection_path.csv"
OUT_PATH_JSON = R5.RES5 / "C2_selection_path.json"
OUT_R = R5.RES5 / "C2_feature_target_r.csv"

# ---------------------------------------------------------------- selection set-up (constants from rev5_common)
MODEL = "KRR_rbf"
SCORING = "neg_mean_absolute_error"
STAGE1_ARMS = ["BASE"] + [f"BASE+{b}" for b in R5.BLOCK_ORDER] + ["EXT_ALL"]
OTHER_TARGETS = [t for t in R5.TARGETS9 if t not in R5.SEL_TARGETS]
FOLDS = list(range(R5.N_OUTER))
assert set(R5.SEL_TARGETS) <= set(R5.TARGETS9) and len(set(R5.SEL_TARGETS)) == len(R5.SEL_TARGETS)
assert MODEL in TM.GRIDS
EXIT_FAIL, EXIT_BAD, EXIT_DIFF = 1, 2, 3
LOCK_INFO_KEYS = {"numpy_version"}          # recorded, not compared on a rerun
FIT_VERSIONS = dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)  # in every unit key

RULE_LOCKBOX = (f"ids = rxn_id of results_rev5/rows_rev5.csv sorted ascending; lockbox = sorted(np.random.default_rng("
                f"{R5.SEED}).choice(ids, size=round({R5.LOCKBOX_FRAC} * len(ids)), replace=False)); dev = the other ids, "
                f"ascending. The lockbox enters no fit, tuning, selection or feature statistic before Phase D (spec C-1).")
CRIT_TEXT = (f"mean over SEL_TARGETS {list(R5.SEL_TARGETS)} of the outer-fold test NMAE (rev5_common.nmae: MAE / mean "
             f"absolute deviation of the fold's test targets), averaged over the {R5.N_OUTER} outer folds")
RULE_SELECT = (f"Dev rows only, KRR(RBF) only (train_ml_single._make_pipe + GRIDS['{MODEL}']). Outer CV: KFold("
               f"{R5.N_OUTER}, shuffle=True, random_state={R5.SEED}) over the dev rows in ascending rxn_id order; in each "
               f"outer train fold GridSearchCV with cv=KFold({R5.N_INNER}, shuffle=True, random_state={R5.SEED}), "
               f"scoring {SCORING}. Criterion: {CRIT_TEXT}. Forward selection from BASE = {R5.BASE}: at each step every "
               f"remaining block is added to the current arm; the candidate with the lowest criterion (ties: block order "
               f"{'..'.join((R5.BLOCK_ORDER[0], R5.BLOCK_ORDER[-1]))}) is accepted iff (criterion_current - "
               f"criterion_candidate) / criterion_current >= {R5.SEL_MIN_REL} AND its fold criterion is lower than the "
               f"current arm's in >= {R5.SEL_MIN_FOLDS} of the {R5.N_OUTER} outer folds; otherwise the selection stops. "
               f"EXT_SEL = BASE + the accepted blocks (= BASE if none; not a STOP). EXT_ALL = BASE + all blocks enters "
               f"Phase D regardless of the selection.")


# ---------------------------------------------------------------- small helpers
def die(msg, code=EXIT_BAD):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def rel(p):
    """Path relative to this directory (portable across checkouts), else as given."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(R5.HERE))
    except ValueError:
        return str(p)


def _jdefault(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def norm(obj):
    """JSON round trip: the form in which an object is stored (tuples -> lists, numpy -> Python)."""
    return json.loads(json.dumps(obj, default=_jdefault))


def same(a, b, rtol=1e-9):
    """Structural equality of JSON-like objects; floats within rtol (NaN == NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rtol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12)
    return a == b


def diff_keys(old, new):
    return sorted(k for k in set(old) | set(new) if not same(old.get(k), new.get(k)))


def env_int(name, default):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        i = int(v)
    except ValueError:
        die(f"{name}={v!r} is not an integer")
    if i < 1:
        die(f"{name}={v!r} must be >= 1")
    return i


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ids_text(ids):
    return "rxn_id\n" + "".join(f"{int(i)}\n" for i in ids)


def read_ids(path, exact=True):
    """rxn_id column of a CSV -> int64 array (file order). exact: rxn_id must be the only column. Exits on a missing
    file, no rows, a non-integer / NaN id or a duplicate."""
    path = Path(path)
    if not path.exists():
        die(f"{path} missing")
    df = pd.read_csv(path)
    if "rxn_id" not in df.columns or (exact and list(df.columns) != ["rxn_id"]):
        die(f"{path}: expected {'exactly ' if exact else ''}a column rxn_id, got {list(df.columns)}")
    s = df["rxn_id"]
    if len(s) == 0:
        die(f"{path}: no rows")
    if not pd.api.types.is_integer_dtype(s):
        die(f"{path}: rxn_id dtype {s.dtype}, not integer (NaN or non-integer ids?)")
    a = s.to_numpy(dtype=np.int64)
    n_dup = len(a) - len(np.unique(a))
    if n_dup:
        die(f"{path}: {n_dup} duplicate rxn_id")
    return a


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    d = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def short(t):
    return t.replace("dft_", "").replace("_kcal", "").replace("_dft", "")


# ---------------------------------------------------------------- C-1 lockbox
def lockbox_split(rows_ids):
    """-> (ids ascending, lockbox ascending, dev ascending), the pre-registered rule."""
    ids = np.sort(np.asarray(rows_ids, dtype=np.int64))
    k = int(round(R5.LOCKBOX_FRAC * len(ids)))
    if not 0 < k < len(ids):
        die(f"lockbox size {k} of {len(ids)} rows is not usable")
    lock = np.sort(np.random.default_rng(R5.SEED).choice(ids, size=k, replace=False))
    dev = np.setdiff1d(ids, lock)
    if len(np.unique(lock)) != k or len(dev) + k != len(ids) or np.intersect1d(lock, dev).size:
        die("internal: inconsistent lockbox split")
    return ids, lock, dev


def lockbox_record(rows_ids, ids, lock, dev):
    return dict(spec="docs/specs/REV5_FEATURES_FIGURES.md C-1", rule=RULE_LOCKBOX, seed=R5.SEED,
                lockbox_frac=R5.LOCKBOX_FRAC,
                rows_file=rel(ROWS5), rows_sha256=R5.sha256(ROWS5),
                rows_file_ascending=bool(np.all(np.diff(rows_ids) > 0)),
                n_rows=int(len(ids)), n_lockbox=int(len(lock)), n_dev=int(len(dev)),
                lockbox_file=rel(LOCK_CSV), lockbox_sha256=sha_text(ids_text(lock)),
                dev_file=rel(DEV_CSV), dev_sha256=sha_text(ids_text(dev)),
                numpy_version=np.__version__)


def cmd_lockbox(args):
    rows = read_ids(ROWS5, exact=False)
    r4 = read_ids(R5.ROWS4, exact=False)
    extra = np.setdiff1d(rows, r4)
    if extra.size:
        die(f"{ROWS5}: {extra.size} rxn_id not in {R5.ROWS4} (rows_rev5 = rows_rev4 ∩ new features ok): "
            f"{extra[:10].tolist()}")
    print(f"rows_rev5: {len(rows)} rows (rows_rev4 {len(r4)}, {len(r4) - len(rows)} without usable new features); "
          f"sha256 {R5.sha256(ROWS5)[:12]}", flush=True)
    ids, lock, dev = lockbox_split(rows)
    texts = {LOCK_CSV: ids_text(lock), DEV_CSV: ids_text(dev)}
    rec = norm(lockbox_record(rows, ids, lock, dev))
    conflicts = [f"{p} (existing {len(p.read_text().splitlines()) - 1} ids)" for p, t in texts.items()
                 if p.exists() and p.read_text() != t]
    if LOCK_JSON.exists():
        old = json.loads(LOCK_JSON.read_text())
        bad = [k for k in diff_keys(old, rec) if k not in LOCK_INFO_KEYS]
        if bad:
            conflicts.append(f"{LOCK_JSON} fields {bad}")
    if conflicts:
        die("existing lockbox files differ from the recomputed split; they are never rewritten: " + "; ".join(conflicts),
            EXIT_DIFF)
    for p, t in texts.items():
        if p.exists():
            print(f"unchanged {rel(p)}", flush=True)
        else:
            R5.write_atomic(p, t)
            print(f"wrote     {rel(p)}", flush=True)
    if LOCK_JSON.exists():
        print(f"unchanged {rel(LOCK_JSON)}", flush=True)
    else:
        R5.write_json(LOCK_JSON, rec)                        # last: its presence marks a finished split
        print(f"wrote     {rel(LOCK_JSON)}", flush=True)
    print(f"lockbox {len(lock)} ({100 * len(lock) / len(ids):.2f} %), dev {len(dev)}, of {len(ids)}; "
          f"sha256 lockbox {rec['lockbox_sha256'][:12]} dev {rec['dev_sha256'][:12]}", flush=True)


def verify_lockbox():
    """Lockbox files == lockbox.json (sha256) == the recomputed split of rows_rev5.csv; exits otherwise.
    -> dict(ids, lock, dev, rec)."""
    missing = [rel(p) for p in (ROWS5, LOCK_CSV, DEV_CSV, LOCK_JSON) if not p.exists()]
    if missing:
        die(f"lockbox not ready, missing {missing} (select_blocks.py lockbox, r5_lockbox.sh)")
    rec = json.loads(LOCK_JSON.read_text())
    ids, lock, dev = lockbox_split(read_ids(ROWS5, exact=False))
    lock_f, dev_f = read_ids(LOCK_CSV), read_ids(DEV_CSV)
    bad = []
    for path, key in ((ROWS5, "rows_sha256"), (LOCK_CSV, "lockbox_sha256"), (DEV_CSV, "dev_sha256")):
        if R5.sha256(path) != rec.get(key):
            bad.append(f"{rel(path)} sha256 != lockbox.json {key}")
    if not np.array_equal(lock_f, lock):
        bad.append(f"{rel(LOCK_CSV)} != the recomputed lockbox")
    if not np.array_equal(dev_f, dev):
        bad.append(f"{rel(DEV_CSV)} != the recomputed dev set")
    if np.intersect1d(lock_f, dev_f).size:
        bad.append(f"{np.intersect1d(lock_f, dev_f).size} ids in both lockbox and dev")
    if not np.array_equal(np.union1d(lock_f, dev_f), ids):
        bad.append("lockbox ∪ dev != rows_rev5")
    if bad:
        die("lockbox check failed: " + "; ".join(bad), EXIT_DIFF)
    return dict(ids=ids, lock=lock, dev=dev, rec=rec)


def assert_dev_only(rxn_ids, lb, what):
    """No lockbox id in `what`, and `what` is exactly dev_ids.csv in ascending order; exits otherwise."""
    rid = np.asarray(rxn_ids, dtype=np.int64)
    leak = np.intersect1d(rid, lb["lock"])
    if leak.size:
        die(f"LOCKBOX LEAK: {leak.size} lockbox rxn_id in {what}: {leak[:10].tolist()}", EXIT_DIFF)
    if not np.array_equal(rid, lb["dev"]):
        die(f"{what}: rows are not dev_ids.csv in ascending rxn_id order")


# ---------------------------------------------------------------- arms
def arm_name(blocks):
    """Canonical forward-selection arm name: BASE or BASE+<blocks in BLOCK_ORDER>."""
    bs = set(blocks)
    ordered = [b for b in R5.BLOCK_ORDER if b in bs]
    return "BASE+" + "+".join(ordered) if ordered else "BASE"


def arm_blocks(arm):
    if arm == "BASE":
        return []
    if arm == "EXT_ALL":
        return list(R5.BLOCK_ORDER)
    if arm.startswith("BASE+"):
        b = arm.split("+")[1:]
        if b and len(set(b)) == len(b) and all(x in R5.BLOCK_ORDER for x in b) and arm_name(b) == arm:
            return b
    raise ValueError(f"not a selection arm name: {arm!r}")


def arm_cols(arm):
    arm_blocks(arm)                                           # validates the name
    cols = R5.arm_columns(R5.BASE if arm == "BASE" else arm, TM.FEATURE_SETS)
    if len(set(cols)) != len(cols):
        raise ValueError(f"{arm}: duplicate feature columns")
    return cols


def cols_sha(cols):
    return sha_text("\n".join(cols))


def unit_path(u):
    arm, t, f = u
    return UNIT_DIR / arm / f"{t}__fold{int(f)}.json"


# ---------------------------------------------------------------- worker (runs in the spawned pool processes)
_W = {}


def _init_worker(data, folds, njobs):
    _W.update(data=data, folds=folds, njobs=njobs)


def _run_unit(arm, target, fold, cols, key, out):
    t0 = time.time()
    data = _W["data"]
    tr, te = _W["folds"][fold]
    X = data[cols].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    est, grid = TM.GRIDS[MODEL]
    gs = GridSearchCV(TM._make_pipe(clone(est)), grid,
                      cv=KFold(R5.N_INNER, shuffle=True, random_state=R5.SEED),
                      scoring=SCORING, n_jobs=_W["njobs"], error_score="raise")
    gs.fit(X[tr], y[tr])
    p = np.asarray(gs.best_estimator_.predict(X[te]), dtype=float).ravel()
    if p.shape != (len(te),) or not np.isfinite(p).all():
        raise ValueError(f"{arm} {target} fold {fold}: prediction shape {p.shape} or non-finite values")
    best = dict(gs.best_params_)
    rec = dict(key=key, arm=arm, target=target, fold=int(fold), n_features=len(cols),
               n_train=int(len(tr)), n_test=int(len(te)),
               mae=float(mean_absolute_error(y[te], p)), nmae=R5.nmae(y[te], p), r2=float(r2_score(y[te], p)),
               best_params=best, best_inner_mae=float(-gs.best_score_),
               edge_hits=sorted(k for k, v in best.items() if v in (min(grid[k]), max(grid[k]))),
               n_jobs=int(_W["njobs"]), seconds=round(time.time() - t0, 2), host=os.uname().nodename,
               test_rxn_id=[int(i) for i in data["rxn_id"].to_numpy()[te]], yhat=[float(v) for v in p])
    if not all(np.isfinite([rec["mae"], rec["nmae"], rec["r2"]])):
        raise ValueError(f"{arm} {target} fold {fold}: non-finite metric {rec['mae']} {rec['nmae']} {rec['r2']}")
    R5.write_json(out, rec)
    return rec


# ---------------------------------------------------------------- unit runner (main process)
class Runner:
    """Cached (arm, target, fold) units; missing ones run in a lazily started spawn ProcessPoolExecutor."""

    def __init__(self, data, folds, base_key, par, njobs):
        self.data, self.folds, self.base_key = data, folds, base_key
        self.par, self.njobs = par, njobs
        self.test_sha = [sha_text(ids_text(data["rxn_id"].to_numpy()[te])) for _, te in folds]
        self.done, self.futs, self.fut_unit = {}, {}, {}
        self.ex = None
        self.lock = threading.Lock()
        self.n_cached = self.n_submitted = self.n_finished = 0

    def key(self, u):
        arm, t, f = u
        cols = arm_cols(arm)
        return norm(dict(self.base_key, arm=arm, target=t, fold=int(f), n_features=len(cols),
                         columns_sha256=cols_sha(cols), n_test=int(len(self.folds[f][1])),
                         test_ids_sha256=self.test_sha[f]))

    def cached(self, u):
        p = unit_path(u)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            die(f"{p}: unreadable ({type(e).__name__}: {e}); delete it to recompute")
        if not same(rec.get("key"), self.key(u)):
            die(f"{p} was computed from other inputs (key fields {diff_keys(rec.get('key') or {}, self.key(u))}); "
                f"move {UNIT_DIR} away to recompute", EXIT_DIFF)
        vals = [rec.get(k) for k in ("mae", "nmae", "r2")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            die(f"{p}: missing / non-finite metrics {vals}")
        return rec

    def _start(self):
        UNIT_DIR.mkdir(parents=True, exist_ok=True)
        self.ex = ProcessPoolExecutor(max_workers=self.par, mp_context=mp.get_context("spawn"),
                                      initializer=_init_worker, initargs=(self.data, self.folds, self.njobs))
        print(f"process pool: {self.par} units at a time x GridSearchCV n_jobs {self.njobs}", flush=True)

    def _report(self, fut):
        u = self.fut_unit.get(fut)
        with self.lock:
            self.n_finished += 1
            n, m = self.n_finished, self.n_submitted
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            print(f"[{n}/{m}] FAILED {u}: {type(exc).__name__}: {exc}", flush=True)
            return
        r = fut.result()
        print(f"[{n}/{m}] {u[0]:26s} {short(u[1]):8s} fold {u[2]}  MAE {r['mae']:7.3f}  NMAE {r['nmae']:.4f}  "
              f"r2 {r['r2']:+.3f}  {r['seconds']:6.0f} s", flush=True)

    def submit(self, units):
        for u in units:
            if u in self.done or u in self.futs:
                continue
            rec = self.cached(u)
            if rec is not None:
                self.done[u] = rec
                self.n_cached += 1
                continue
            if self.ex is None:
                self._start()
            fut = self.ex.submit(_run_unit, u[0], u[1], int(u[2]), arm_cols(u[0]), self.key(u), str(unit_path(u)))
            self.futs[u] = fut
            self.fut_unit[fut] = u
            with self.lock:
                self.n_submitted += 1
            fut.add_done_callback(self._report)

    def fail(self, u, exc):
        print(f"### unit {u} failed:", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        self.abort()
        die(f"unit {u} failed: {type(exc).__name__}: {exc}", EXIT_FAIL)

    def get(self, units):
        """Records of `units` (submitting what is neither cached nor queued); blocks until they exist."""
        self.submit(units)
        pend = [self.futs[u] for u in units if u in self.futs]
        if pend:
            done, _ = wait(pend, return_when=FIRST_EXCEPTION)
            for fut in done:
                if not fut.cancelled() and fut.exception() is not None:
                    self.fail(self.fut_unit[fut], fut.exception())
        for u in units:
            if u in self.futs:
                self.done[u] = self.futs.pop(u).result()
        return {u: self.done[u] for u in units}

    def drain(self):
        self.get(list(self.futs))

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=True)
            self.ex = None

    def abort(self):
        """On any error in the main process: let the running units finish (cached), cancel the queued ones (without
        this, the interpreter's exit handler would still run every queued unit)."""
        if self.ex is not None:
            print("### aborting: waiting for the running units, cancelling the queued ones", flush=True)
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None


# ---------------------------------------------------------------- forward selection
def summarize(recs, arm):
    """-> (per-fold criterion array, {SEL target: NMAE mean over folds})."""
    folds = np.array([np.mean([recs[(arm, t, f)]["nmae"] for t in R5.SEL_TARGETS]) for f in FOLDS], dtype=float)
    per_t = {t: float(np.mean([recs[(arm, t, f)]["nmae"] for f in FOLDS])) for t in R5.SEL_TARGETS}
    return folds, per_t


def sel_units(arms):
    return [(a, t, f) for a in arms for t in R5.SEL_TARGETS for f in FOLDS]


def path_row(step, current, cand, block, folds, per_t, cur_folds, decision, rel_dec=None, lower=None):
    r = dict(step=int(step), current=current, candidate=cand, added_block=block, n_features=len(arm_cols(cand)),
             criterion=float(np.mean(folds)),
             criterion_current=None if cur_folds is None else float(np.mean(cur_folds)),
             rel_decrease=None if rel_dec is None else float(rel_dec),
             folds_lower=None if lower is None else int(lower),
             best=decision in ("accepted", "rejected"), accepted=decision == "accepted", decision=decision)
    r.update({f"crit_fold{f}": float(folds[f]) for f in FOLDS})
    r.update({f"nmae_{t}": float(per_t[t]) for t in R5.SEL_TARGETS})
    return r


def forward_select(crit_of):
    """The C-1 forward rule. crit_of(arms) -> {arm: summarize(...)} or None (status: not cached yet).
    -> (path rows, final dict or None if the path is not complete)."""
    rows = []
    got = crit_of(["BASE"])
    if got is None:
        return rows, None
    cur_blocks, cur_arm = [], "BASE"
    cur_f, cur_t = got["BASE"]
    rows.append(path_row(0, "", "BASE", "", cur_f, cur_t, None, "start"))
    step, stop = 0, None
    while True:
        remaining = [b for b in R5.BLOCK_ORDER if b not in cur_blocks]
        if not remaining:
            stop = f"all {len(R5.BLOCK_ORDER)} blocks accepted"
            break
        step += 1
        cands = [(b, arm_name(cur_blocks + [b])) for b in remaining]
        got = crit_of([a for _, a in cands])
        if got is None:
            return rows, None
        c0 = float(np.mean(cur_f))
        ev = []
        for b, a in cands:
            f, t = got[a]
            c = float(np.mean(f))
            ev.append(dict(block=b, arm=a, folds=f, per_t=t, crit=c, rel=(c0 - c) / c0, lower=int(np.sum(f < cur_f))))
        best = min(ev, key=lambda e: (e["crit"], R5.BLOCK_ORDER.index(e["block"])))
        ok = best["rel"] >= R5.SEL_MIN_REL and best["lower"] >= R5.SEL_MIN_FOLDS
        for e in ev:
            dec = ("accepted" if ok else "rejected") if e is best else "not_best"
            rows.append(path_row(step, cur_arm, e["arm"], e["block"], e["folds"], e["per_t"], cur_f, dec,
                                 e["rel"], e["lower"]))
        if not ok:
            stop = (f"step {step}: best candidate {best['arm']} has relative decrease {best['rel']:.4f} (needs >= "
                    f"{R5.SEL_MIN_REL}) and is lower in {best['lower']}/{R5.N_OUTER} folds (needs >= "
                    f"{R5.SEL_MIN_FOLDS})")
            break
        cur_blocks.append(best["block"])
        cur_arm, cur_f, cur_t = best["arm"], best["folds"], best["per_t"]
    return rows, dict(blocks=list(cur_blocks), arm=cur_arm, criterion=float(np.mean(cur_f)),
                      fold_criteria=[float(x) for x in cur_f], per_target=dict(cur_t), stop_reason=stop, n_steps=step,
                      base_criterion=rows[0]["criterion"], base_fold_criteria=[rows[0][f"crit_fold{f}"] for f in FOLDS])


# ---------------------------------------------------------------- C-2 tables
def fold_row(arm, alias, set_, r):
    row = dict(arm=arm, alias_of=alias, set=set_, blocks="+".join(arm_blocks(alias or arm)),
               n_features=int(r["n_features"]), target=r["target"], fold=int(r["fold"]),
               n_train=int(r["n_train"]), n_test=int(r["n_test"]),
               mae=float(r["mae"]), nmae=float(r["nmae"]), r2=float(r["r2"]), best_inner_mae=float(r["best_inner_mae"]))
    row.update({"best_" + k.rsplit("__", 1)[-1]: v for k, v in sorted(r["best_params"].items())})
    row.update(edge_hits=";".join(r["edge_hits"]), n_jobs=r.get("n_jobs"), seconds=r.get("seconds"))
    return row


def block_cv_tables(done, ext_arm, forward_arms):
    order = STAGE1_ARMS + [a for a in forward_arms if a not in STAGE1_ARMS]
    recs = []
    for arm in order:
        for t in R5.TARGETS9:
            have = [f for f in FOLDS if (arm, t, f) in done]
            if have and have != FOLDS:
                die(f"internal: {arm} {t} has folds {have} only")
            recs += [fold_row(arm, "", "stage1" if arm in STAGE1_ARMS else "forward", done[(arm, t, f)]) for f in have]
    recs += [fold_row("EXT_SEL", ext_arm, "EXT_SEL", done[(ext_arm, t, f)]) for t in R5.TARGETS9 for f in FOLDS]
    folds_df = pd.DataFrame(recs)
    agg = []
    for (arm, t), g in folds_df.groupby(["arm", "target"], sort=False):
        g = g.sort_values("fold")
        if g["fold"].tolist() != FOLDS:
            die(f"internal: {arm} {t} folds {g['fold'].tolist()}")
        m, n = g["mae"].to_numpy(dtype=float), g["nmae"].to_numpy(dtype=float)
        bm = np.array([done[("BASE", t, f)]["mae"] for f in FOLDS], dtype=float)
        bn = np.array([done[("BASE", t, f)]["nmae"] for f in FOLDS], dtype=float)
        first = g.iloc[0]
        agg.append(dict(arm=arm, alias_of=first["alias_of"], set=first["set"], blocks=first["blocks"],
                        n_features=int(first["n_features"]), target=t, n_folds=len(g),
                        mae_mean=float(m.mean()), mae_sd=float(m.std(ddof=1)),
                        nmae_mean=float(n.mean()), nmae_sd=float(n.std(ddof=1)),
                        r2_mean=float(g["r2"].mean()),
                        d_mae_vs_base=float(m.mean() - bm.mean()), d_nmae_vs_base=float(n.mean() - bn.mean()),
                        folds_nmae_below_base=int(np.sum(n < bn))))
    return folds_df, pd.DataFrame(agg)


def feature_target_r(data):
    rows = []
    for b in R5.BLOCK_ORDER:
        for c in R5.BLOCKS[b]:
            x = data[c].to_numpy(dtype=float)
            row = dict(block=b, feature=c, n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                       n_unique=int(len(np.unique(x))))
            row.update({f"r_{t}": pearson(x, data[t].to_numpy(dtype=float)) for t in R5.SEL_TARGETS})
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path, df):
    R5.write_atomic(path, lambda tmp: df.to_csv(tmp, index=False))
    print(f"wrote {rel(path)} ({len(df)} rows)", flush=True)


# ---------------------------------------------------------------- C-2 / C-3 select
def load_dev(lb):
    """Dev rows (ascending rxn_id) of the G1 + block features and the 9 targets, as float64 + rxn_id; asserted free of
    lockbox rows before any X / y is built. -> (data, load_ml info)."""
    missing = [str(p) for p in (R5.FEAT_G1, R5.FEAT_EXT) if not Path(p).is_file()]
    if missing:
        die(f"feature parquet missing: {missing} (ESPLEY_OUT={R5.XTB_ROOT})")
    try:                             # its gates sys.exit(<message>), i.e. status 1 ('a unit failed'); input here -> 2
        ml, info = TM.load_ml(feat_path=R5.FEAT_G1, rows_path=str(DEV_CSV), targets=list(R5.TARGETS9), geom="g1",
                              ext=True, ext_path=R5.FEAT_EXT)
    except SystemExit as e:
        die(f"train_ml_single.load_ml rejected the dev rows / feature files: {e.code}", EXIT_BAD)
    assert_dev_only(ml["rxn_id"].to_numpy(), lb, "the loaded selection rows")
    if info.get("rows_sha256") != lb["rec"]["dev_sha256"]:
        die(f"load_ml rows sha256 {info.get('rows_sha256')} != lockbox.json dev_sha256")
    cols = arm_cols("EXT_ALL")
    if set(cols) != set(TM.FEATURE_SETS[R5.BASE]) | set(R5.EXT_COLS) or len(cols) != len(set(cols)):
        die("internal: EXT_ALL columns != BASE + the 109 block columns")
    need = cols + list(R5.TARGETS9)
    if len(set(need)) != len(need):
        die("internal: a target is also a feature column")
    try:
        num = ml[need].astype(float)
    except (TypeError, ValueError) as e:
        die(f"non-numeric feature / target column in the dev rows: {e}")
    bad = [c for c in need if not np.isfinite(num[c].to_numpy()).all()]
    if bad:
        die(f"non-finite values in the dev rows, columns {bad[:10]}")
    data = pd.concat([ml[["rxn_id"]].astype(np.int64).reset_index(drop=True), num.reset_index(drop=True)], axis=1)
    assert_dev_only(data["rxn_id"].to_numpy(), lb, "X / y")
    return data, info


def outer_folds(data):
    n = len(data)
    folds = [(np.asarray(tr), np.asarray(te))
             for tr, te in KFold(R5.N_OUTER, shuffle=True, random_state=R5.SEED).split(np.arange(n))]
    fold_of = np.full(n, -1, dtype=int)
    for f, (tr, te) in enumerate(folds):
        if len(tr) + len(te) != n or np.intersect1d(tr, te).size:
            die(f"internal: outer fold {f} is not a partition")
        fold_of[te] = f
    if (fold_of < 0).any():
        die("internal: a dev row is in no outer test fold")
    text = "rxn_id,fold\n" + "".join(f"{int(i)},{int(f)}\n" for i, f in zip(data["rxn_id"], fold_of))
    if FOLDS_CSV.exists():
        if FOLDS_CSV.read_text() != text:
            die(f"{FOLDS_CSV} differs from the outer folds of these dev rows; move {SEL_DIR} away", EXIT_DIFF)
    else:
        R5.write_atomic(FOLDS_CSV, text)
    return folds


def cmd_select(args):
    lb = verify_lockbox()
    ncpu = env_int("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
    njobs = env_int("R5_SEL_NJOBS", 1)
    par = env_int("R5_SEL_PAR", max(1, ncpu // njobs))
    if par * njobs > ncpu:
        print(f"WARNING: R5_SEL_PAR {par} x R5_SEL_NJOBS {njobs} > {ncpu} cpus", flush=True)
    print(f"select: cpus {ncpu}, {par} units at a time x GridSearchCV n_jobs {njobs}; BLAS threads "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')}\n  features {R5.FEAT_G1} + {R5.FEAT_EXT}\n  cache {UNIT_DIR}",
          flush=True)
    data, info = load_dev(lb)
    folds = outer_folds(data)
    base_key = dict(model=MODEL, grid=json.dumps(TM.GRIDS[MODEL][1], sort_keys=True), scoring=SCORING,
                    outer=f"KFold({R5.N_OUTER}, shuffle=True, random_state={R5.SEED}), dev rows ascending",
                    inner=f"KFold({R5.N_INNER}, shuffle=True, random_state={R5.SEED})",
                    dev_sha256=lb["rec"]["dev_sha256"], n_dev=int(len(data)),
                    features_sha256=info["features_sha256"], ext_features_sha256=info["ext_features_sha256"],
                    versions=dict(FIT_VERSIONS))
    print(f"dev rows {len(data)} (lockbox {len(lb['lock'])} never loaded into X / y); outer folds test sizes "
          f"{[len(te) for _, te in folds]}; arms {', '.join(f'{a} ({len(arm_cols(a))})' for a in STAGE1_ARMS)}",
          flush=True)
    runner = Runner(data, folds, base_key, par, njobs)
    stage1 = [(a, t, f) for t in list(R5.SEL_TARGETS) + OTHER_TARGETS for a in STAGE1_ARMS for f in FOLDS]
    if args.check:
        n_c = sum(runner.cached(u) is not None for u in stage1)
        n_all = sum(1 for _ in UNIT_DIR.glob("*/*.json")) if UNIT_DIR.exists() else 0
        nb = len(R5.BLOCK_ORDER)
        n_fwd = nb * (nb - 1) // 2 * len(R5.SEL_TARGETS) * R5.N_OUTER           # steps 2..nb (step 1 = stage 1)
        print(f"CHECK OK: inputs valid; stage-1 units cached {n_c}/{len(stage1)} (keys verified), unit files in cache "
              f"{n_all}. A full run: {len(stage1)} stage-1 units + <= {n_fwd} forward + <= "
              f"{len(OTHER_TARGETS) * R5.N_OUTER} EXT_SEL report units; nothing fitted.", flush=True)
        return

    def crit_of(arms):
        recs = runner.get(sel_units(arms))
        return {a: summarize(recs, a) for a in arms}

    try:
        runner.submit(stage1)                                 # SEL targets first: forward step 1 needs only those
        rows, final = forward_select(crit_of)
        ext_arm = final["arm"]
        print(f"\nselection complete: EXT_SEL = {ext_arm} (blocks {final['blocks'] or 'none'}), criterion "
              f"{final['base_criterion']:.5f} -> {final['criterion']:.5f}; stop: {final['stop_reason']}", flush=True)
        runner.get([(ext_arm, t, f) for t in OTHER_TARGETS for f in FOLDS])   # report rows only
        runner.drain()
    except BaseException:
        runner.abort()
        raise
    runner.close()
    done = runner.done
    print(f"units: {len(done)} ({runner.n_cached} from cache, {runner.n_submitted} fitted in this run)", flush=True)

    ext_blocks = [b for b in R5.BLOCK_ORDER if b in final["blocks"]]
    n_base = len(arm_cols("BASE"))
    inputs = dict(n_dev=int(len(data)), n_lockbox=int(len(lb["lock"])), dev_ids=rel(DEV_CSV),
                  dev_sha256=lb["rec"]["dev_sha256"], lockbox_sha256=lb["rec"]["lockbox_sha256"],
                  rows_rev5_sha256=lb["rec"]["rows_sha256"],
                  features_file=Path(info["features_file"]).name, features_sha256=info["features_sha256"],
                  ext_features_file=Path(info["ext_features_file"]).name,
                  ext_features_sha256=info["ext_features_sha256"])
    rule = dict(text=RULE_SELECT, criterion=CRIT_TEXT, sel_targets=list(R5.SEL_TARGETS), sel_min_rel=R5.SEL_MIN_REL,
                sel_min_folds=R5.SEL_MIN_FOLDS, n_outer=R5.N_OUTER, n_inner=R5.N_INNER, seed=R5.SEED, model=MODEL,
                grid=TM.GRIDS[MODEL][1], scoring=SCORING, tie_break="BLOCK_ORDER", block_order=list(R5.BLOCK_ORDER))
    prereg = norm(dict(
        spec="docs/specs/REV5_FEATURES_FIGURES.md C-3 (PREREG_REV5b): EXT_SEL, fixed before Phase D",
        ext_sel_blocks=ext_blocks, ext_sel_acceptance_order=list(final["blocks"]), ext_sel_arm=ext_arm,
        base_arm=R5.BASE, n_features_base=n_base, n_features=len(arm_cols(ext_arm)),
        n_features_added=len(arm_cols(ext_arm)) - n_base,
        selection=dict(criterion=CRIT_TEXT, base_criterion=final["base_criterion"],
                       base_fold_criteria=final["base_fold_criteria"], final_criterion=final["criterion"],
                       final_fold_criteria=final["fold_criteria"], final_nmae_per_target=final["per_target"],
                       steps=[{k: r[k] for k in ("step", "candidate", "added_block", "criterion", "criterion_current",
                                                 "rel_decrease", "folds_lower", "accepted")}
                              for r in rows if r["best"]],
                       stop_reason=final["stop_reason"]),
        rule=rule,
        ext_all="EXT_ALL = BASE + B1..B6 is evaluated in Phase D regardless of the selection (spec C-1)",
        inputs=inputs, selection_path=rel(OUT_PATH_JSON)))
    if PREREG.exists():                                       # checked before any output is (re)written
        old = json.loads(PREREG.read_text())
        if not same(old, prereg):
            die(f"{PREREG} exists and differs from this selection in {diff_keys(old, prereg)}; the pre-registration "
                f"is never rewritten", EXIT_DIFF)

    forward_arms = list(dict.fromkeys(r["candidate"] for r in rows))
    folds_df, cv_df = block_cv_tables(done, ext_arm, forward_arms)
    path_df = pd.DataFrame(rows)
    path_df["folds_lower"] = path_df["folds_lower"].astype("Int64")
    r_df = feature_target_r(data)
    ref_r = {t: dict(feature=TM.PRE_ML[t], r=pearson(data[TM.PRE_ML[t]], data[t])) for t in R5.SEL_TARGETS}
    core_s = sum(float(r.get("seconds") or 0) * int(r.get("n_jobs") or 1) for r in done.values())
    path_json = dict(
        spec="docs/specs/REV5_FEATURES_FIGURES.md C-2 (selection path) / C-1 rule", rule=rule, inputs=inputs,
        stage1_arms=STAGE1_ARMS, arm_n_features={a: len(arm_cols(a)) for a in STAGE1_ARMS + forward_arms},
        folds=[dict(fold=f, n_train=int(len(tr)), n_test=int(len(te))) for f, (tr, te) in enumerate(folds)],
        dev_folds_file=str(FOLDS_CSV), dev_folds_sha256=R5.sha256(FOLDS_CSV),
        steps=rows, final=final, ext_sel_arm=ext_arm, ext_sel_blocks=ext_blocks, n_features=len(arm_cols(ext_arm)),
        reference_r_dev=ref_r, sd="sample SD over the outer folds (ddof 1)",
        units=dict(n=len(done), core_hours=round(core_s / 3600, 3), unit_cache=str(UNIT_DIR),
                   core_hours_note="sum over units of wall seconds x GridSearchCV n_jobs (from the unit caches)"),
        versions=dict(FIT_VERSIONS, pandas=pd.__version__))

    write_csv(OUT_FOLDS, folds_df)
    write_csv(OUT_CV, cv_df)
    write_csv(OUT_PATH_CSV, path_df)
    R5.write_json(OUT_PATH_JSON, path_json)
    print(f"wrote {rel(OUT_PATH_JSON)}", flush=True)
    write_csv(OUT_R, r_df)
    if PREREG.exists():
        print(f"unchanged {rel(PREREG)} (identical to this selection)", flush=True)
    else:
        R5.write_json(PREREG, prereg)                          # last: C-3 exists only after a complete path
        print(f"wrote {rel(PREREG)}", flush=True)
    if R5.arm_columns("EXT_SEL", TM.FEATURE_SETS) != arm_cols(ext_arm):                # what Phase D will read
        die(f"{PREREG}: rev5_common.arm_columns('EXT_SEL') != the columns of {ext_arm}")
    print(f"\nEXT_SEL = {ext_arm}: blocks {ext_blocks or '[] (= BASE)'}, {prereg['n_features']} features "
          f"({prereg['n_features_added']} added); core-h {core_s / 3600:.1f}", flush=True)


# ---------------------------------------------------------------- status
def cmd_status(args):
    print(f"results {R5.RES5}\nscratch {SEL_DIR}")
    for p in (ROWS5, LOCK_CSV, DEV_CSV, LOCK_JSON, PREREG, OUT_CV, OUT_PATH_JSON):
        print(f"  {'present' if p.exists() else 'absent '}  {rel(p)}")
    if LOCK_JSON.exists():
        rec = json.loads(LOCK_JSON.read_text())
        sha_ok = all(p.exists() and R5.sha256(p) == rec.get(k) for p, k in
                     ((ROWS5, "rows_sha256"), (LOCK_CSV, "lockbox_sha256"), (DEV_CSV, "dev_sha256")))
        print(f"lockbox: rows {rec.get('n_rows')}, lockbox {rec.get('n_lockbox')}, dev {rec.get('n_dev')}; files match "
              f"lockbox.json sha256: {sha_ok}")

    def load(u):
        p = unit_path(u)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    n_files = sum(1 for _ in UNIT_DIR.glob("*/*.json")) if UNIT_DIR.exists() else 0
    print(f"\nunit cache: {n_files} files (keys not re-validated here)\nstage 1 (cached folds per arm x target):")
    print(f"  {'arm':12s} " + " ".join(f"{short(t):>7s}" for t in R5.TARGETS9))
    tot = 0
    for a in STAGE1_ARMS:
        cnt = [sum(unit_path((a, t, f)).exists() for f in FOLDS) for t in R5.TARGETS9]
        tot += sum(cnt)
        print(f"  {a:12s} " + " ".join(f"{c:>7d}" for c in cnt))
    print(f"  stage 1: {tot}/{len(STAGE1_ARMS) * len(R5.TARGETS9) * R5.N_OUTER}")

    waiting = {}

    def crit_of(arms):
        units = sel_units(arms)
        recs = {u: load(u) for u in units}
        miss = [u for u, r in recs.items() if r is None]
        if miss:
            waiting.update(arms=arms, missing=len(miss), total=len(units))
            return None
        return {a: summarize(recs, a) for a in arms}

    rows, final = forward_select(crit_of)
    print("\nselection path (from the cache):")
    for r in rows:
        extra = "" if r["rel_decrease"] is None else f"  rel {r['rel_decrease']:+.4f}  lower {r['folds_lower']}/{R5.N_OUTER}"
        print(f"  step {r['step']}  {r['candidate']:26s} crit {r['criterion']:.5f}{extra}  {r['decision']}")
    if final is None:
        if waiting:
            print(f"  waiting: {waiting['missing']}/{waiting['total']} units of arms {waiting['arms']} not cached yet")
    else:
        print(f"  complete: EXT_SEL = {final['arm']}  ({final['stop_reason']})")
    if PREREG.exists():
        pr = json.loads(PREREG.read_text())
        print(f"\nprereg_rev5b.json: ext_sel_blocks {pr.get('ext_sel_blocks')}, n_features {pr.get('n_features')}, "
              f"sha256 {R5.sha256(PREREG)[:12]}")


def main():
    ap = argparse.ArgumentParser(description="rev 5 Phase C: lockbox, dev-CV block evaluation, forward selection")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lockbox", help="C-1 lockbox / dev split of rows_rev5.csv")
    s = sub.add_parser("select", help="C-2 / C-3 dev-CV block evaluation and forward selection -> prereg_rev5b.json")
    s.add_argument("--check", action="store_true", help="validate inputs and cached units only; fit nothing")
    sub.add_parser("status", help="print progress")
    a = ap.parse_args()
    {"lockbox": cmd_lockbox, "select": cmd_select, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
