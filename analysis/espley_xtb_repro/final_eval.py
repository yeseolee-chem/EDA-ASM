#!/usr/bin/env python3
"""final_eval.py — rev 5 Phase D-1: lockbox evaluation (docs/specs/REV5_FEATURES_FIGURES.md D-1), run by
r5_lockbox_eval.sh inside one SLURM allocation.

  python final_eval.py [--gates-only]

Train on every dev row (results_rev5/dev_ids.csv, file order), score on the lockbox rows (results_rev5/lockbox_ids.csv,
file order). Features: the G1 parquet (rev5_common.FEAT_G1) inner-merged with the blocks B1..B6 (rev5_common.FEAT_EXT)
through train_ml_single.load_ml on results_rev5/rows_rev5.csv (every row gated: ok, NaN-free, hygiene; nothing dropped).
  arms    BASE (= ESPLEY73), EXT_SEL (ESPLEY73 + prereg_rev5b.json blocks), EXT_ALL (ESPLEY73 + B1..B6);
          columns from train_ml_single.resolve_arm (rev5_common.arm_columns)
  models  KRR_rbf (headline), Ridge, SVR_rbf, XGB (appendix): train_ml_single.GRIDS / _make_pipe, tuned by
          GridSearchCV on the whole dev set, cv = KFold(N_INNER, shuffle=True, random_state=SEED), MAE scoring,
          n_jobs = R5_NJOBS or SLURM_CPUS_PER_TASK; the refit best estimator predicts the lockbox
  targets rev5_common.TARGETS9
Metrics per (arm, model, target) on the lockbox: MAE, NMAE (MAE / mean absolute deviation of the lockbox targets),
r², RMSE, Pearson r; reaction-level bootstrap: ONE index matrix np.random.default_rng(SEED).integers(0, n_lockbox,
size=(N_BOOT, n_lockbox)) shared by every arm / model / target, 95 % percentile CI of MAE and NMAE; paired MAE
difference EXT_SEL − BASE and EXT_ALL − BASE (same model, same target) with its percentile CI.

Gates before any fit (exit != 0 with a message, nothing trained):
  * the pre-registration files (PREREG_REV5b.md, prereg_rev5b.json, rows_rev5.csv, dev_ids.csv, lockbox_ids.csv)
    are tracked, in HEAD and identical to HEAD; the lockbox_ids.csv sha256 is written in a committed, unchanged
    results_rev5/PREREG_REV5*.md or prereg_rev5*.json (C-1 records it);
  * dev ∩ lockbox = ∅, dev ∪ lockbox = rows_rev5, |lockbox| = LOCKBOX_FRAC · |rows_rev5| ± 1;
  * first use of the lockbox: every file under $R5_SCRATCH/select (the Phase C caches, which store the test ids of
    each outer fold) is scanned for reaction ids (dict keys / columns / index names matching ID_KEY in .json /
    .parquet / .csv / .tsv / .npz / .npy / .pkl, also readable *.tmp leftovers; a truncated *.tmp is recorded, not
    fatal, as selection never read it). Exit 3 if one is a lockbox id; exit 2 if the ids found are not exactly the
    dev ids (vacuous or foreign audit), if a data file cannot be parsed, or if a file of another format is named like
    an id file.
Idempotent: one cache per (arm, model, target) in $R5_SCRATCH/lockbox/units/ (<arm>__<model>__<target>.parquet with
the lockbox predictions, then .json with the fit record = done marker, both atomic). A finished unit is reused, and
one made from other inputs (files / sha256, columns, grid, cv, package versions) stops the run: move it away. An arm
with exactly the columns of an earlier arm in ARMS = (BASE, EXT_SEL, EXT_ALL) is not refitted (deterministic fits):
when C selected no block, EXT_SEL is copied from BASE (same_columns_as = BASE; paired difference exactly 0); when it
selected all six, EXT_ALL is copied from EXT_SEL (same_columns_as = EXT_SEL). The copy carries `same_columns_as`, the
arm it was copied from keeps it empty. The report is rebuilt from the caches on every run (cheap, deterministic):
  $R5_SCRATCH/lockbox/lockbox_predictions.parquet   rxn_id, arm, model, target, y, yhat (every unit)
  results_rev5/D1_lockbox.csv                        one row per (arm, model, target)
  results_rev5/D1_lockbox_paired.csv                 EXT_SEL − BASE and EXT_ALL − BASE per (model, target)
  results_rev5/D1_lockbox.json                       inputs + sha256, partition, select audit, arms, bootstrap, headline
--gates-only runs the gates and the row load, prints the plan and exits 0 without fitting or writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev5_common as R5  # noqa: E402
import train_ml_single as T  # noqa: E402
from aggregate_ml import TARGET_LABEL  # noqa: E402

DEV_IDS = R5.RES5 / "dev_ids.csv"                      # C-1: rows_rev5 minus the lockbox
SELECT_DIR = R5.SCRATCH / "select"                     # Phase C caches (select_blocks.py)
OUT_DIR = R5.SCRATCH / "lockbox"
UNIT_DIR = OUT_DIR / "units"
OUT_PREDS = OUT_DIR / "lockbox_predictions.parquet"
OUT_CSV = R5.RES5 / "D1_lockbox.csv"
OUT_PAIRED = R5.RES5 / "D1_lockbox_paired.csv"
OUT_JSON = R5.RES5 / "D1_lockbox.json"
GATED = [R5.RES5 / "PREREG_REV5b.md", R5.PREREG5B_JSON, R5.ROWS5, DEV_IDS, R5.LOCKBOX]

ARMS = ("BASE", "EXT_SEL", "EXT_ALL")                  # BASE first: EXT_SEL may copy it
ARM_SOURCE = {"BASE": R5.BASE, "EXT_SEL": "EXT_SEL", "EXT_ALL": "EXT_ALL"}   # names for train_ml_single.resolve_arm
PAIRED = ("EXT_SEL", "EXT_ALL")                        # each minus BASE
MODELS = ("KRR_rbf", "Ridge", "SVR_rbf", "XGB")        # headline first (a clipped run has it)
HEAD_MODEL = "KRR_rbf"
SCORING = "neg_mean_absolute_error"
assert set(MODELS) == set(T.GRIDS), (MODELS, list(T.GRIDS))
assert set(R5.TARGETS9) <= set(TARGET_LABEL), R5.TARGETS9

# select-cache audit: keys / columns that hold reaction ids (rxn_id, rxn_ids, test_rxn_ids, test_ids, train_ids, ids);
# not fold_id, test_idx (positions), n_test (counts). A bare scalar is an id only under an rxn key.
ID_KEY = re.compile(r"(?i)rxn_?ids?|(^|_)ids$")
RXN_KEY = re.compile(r"(?i)rxn_?ids?")
DATA_EXT = {".json", ".parquet", ".csv", ".tsv", ".npz", ".npy", ".pkl", ".pickle"}


def die(msg, code=2):
    print(f"\nFATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ------------------------------------------------------------------------------------------ pre-registration gates
def git(*args):
    return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True)


def committed(path):
    """'' if path is tracked, present in HEAD and identical to HEAD (index and work tree), else the reason."""
    rel = os.path.relpath(path, HERE)
    if not Path(path).is_file():
        return f"{rel}: missing"
    if git("ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return f"{rel}: not tracked by git"
    if git("cat-file", "-e", f"HEAD:./{rel}").returncode != 0:
        return f"{rel}: not in HEAD (staged only?)"
    r = git("diff", "--quiet", "HEAD", "--", rel)
    if r.returncode == 1:
        return f"{rel}: differs from HEAD"
    if r.returncode != 0:
        return f"{rel}: git diff failed ({r.stderr.strip()})"
    return ""


def gate_prereg():
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        die(f"git rev-parse HEAD failed in {HERE}: {head.stderr.strip()}")
    bad = [m for m in map(committed, GATED) if m]
    if bad:
        die("PREREG gate (commit PREREG_REV5b, C-3, first):\n  " + "\n  ".join(bad))
    lock_sha = R5.sha256(R5.LOCKBOX)
    docs = sorted(R5.RES5.glob("PREREG_REV5*.md")) + sorted(R5.RES5.glob("prereg_rev5*.json"))
    rec = [p for p in docs if lock_sha in p.read_text(encoding="utf-8", errors="replace").lower()]
    rec_ok = [p for p in rec if not committed(p)]
    if not rec_ok:
        die(f"PREREG gate: sha256 {lock_sha} of {R5.LOCKBOX.name} is not written in any committed, unchanged "
            f"results_rev5/PREREG_REV5*.md / prereg_rev5*.json (searched {[p.name for p in docs]}, "
            f"found in {[p.name for p in rec]}); C-1 records it")
    return dict(git_head=head.stdout.strip(),
                files={os.path.relpath(p, HERE): R5.sha256(p) for p in GATED},
                lockbox_sha256_recorded_in=[p.name for p in rec_ok])


def read_ids(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:                                                   # noqa: BLE001
        die(f"{path}: unreadable ({type(e).__name__}: {e})")
    if "rxn_id" not in df:
        die(f"{path}: no rxn_id column (columns {list(df.columns)[:10]})")
    s = df["rxn_id"]
    if len(s) == 0 or s.isna().any() or not np.issubdtype(s.dtype, np.integer):
        die(f"{path}: rxn_id must be a non-empty integer column without NaN (dtype {s.dtype}, n {len(s)})")
    ids = s.astype(int).tolist()
    if len(set(ids)) != len(ids):
        die(f"{path}: {len(ids) - len(set(ids))} duplicate rxn_id")
    return ids


def gate_partition(rows, dev, lock):
    s_rows, s_dev, s_lock = set(rows), set(dev), set(lock)
    bad = []
    if s_dev & s_lock:
        bad.append(f"{len(s_dev & s_lock)} rxn_id in both dev and lockbox: {sorted(s_dev & s_lock)[:10]}")
    if s_dev | s_lock != s_rows:
        bad.append(f"dev ∪ lockbox != rows_rev5: {len(s_rows - s_dev - s_lock)} rows in neither, "
                   f"{len((s_dev | s_lock) - s_rows)} not in rows_rev5")
    want = R5.LOCKBOX_FRAC * len(rows)
    if abs(len(lock) - want) > 1:
        bad.append(f"|lockbox| = {len(lock)}, expected {R5.LOCKBOX_FRAC} x {len(rows)} = {want:.1f} ± 1")
    if bad:
        die("partition gate:\n  " + "\n  ".join(bad))
    return dict(n_rows=len(rows), n_dev=len(dev), n_lockbox=len(lock), lockbox_frac=len(lock) / len(rows))


# ------------------------------------------------------------------------------------------ select-cache audit
def collect_ints(v, out, odd):
    """Every integer under v (nested dict / list / array / Series); integral floats and digit strings count, other
    strings are noted in `odd`; a non-integral or NaN number raises (an id field must hold ids only)."""
    if v is None or isinstance(v, (bool, np.bool_)):
        return
    if isinstance(v, (int, np.integer)):
        out.append(int(v))
    elif isinstance(v, (float, np.floating)):
        if not (np.isfinite(v) and float(v).is_integer()):
            raise ValueError(f"non-integer value {v!r} in an id field")
        out.append(int(v))
    elif isinstance(v, str):
        if re.fullmatch(r"\s*-?\d+\s*", v):
            out.append(int(v))
        else:
            odd.append(v[:80])
    elif isinstance(v, dict):
        for x in v.values():
            collect_ints(x, out, odd)
    elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
        out.extend(int(x) for x in v.ravel())
    elif isinstance(v, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        for x in v:
            collect_ints(x, out, odd)


def walk(obj, path, acc):
    """Collect the ids under every ID_KEY dict key / DataFrame column of obj into acc."""
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            acc["keys_seen"][str(c)] += 1
            if ID_KEY.search(str(c)):
                ids, odd = [], []
                collect_ints(obj[c].to_numpy(), ids, odd)
                acc["hits"].append((f"{path}/{c}", ids, odd))
        for lvl, nm in enumerate(obj.index.names):                 # e.g. a frame indexed by rxn_id
            if nm is not None and ID_KEY.search(str(nm)):
                ids, odd = [], []
                collect_ints(obj.index.get_level_values(lvl).to_numpy(), ids, odd)
                acc["hits"].append((f"{path}/<index {nm}>", ids, odd))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            k = str(k)
            acc["keys_seen"][k] += 1
            if ID_KEY.search(k):
                if isinstance(v, (int, np.integer, float, np.floating)) and not RXN_KEY.search(k):
                    continue                                   # e.g. n_ids: a count, not an id
                ids, odd = [], []
                collect_ints(v, ids, odd)
                acc["hits"].append((f"{path}/{k}", ids, odd))
            else:
                walk(v, f"{path}/{k}", acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            walk(v, f"{path}[]", acc)


def read_cache(p, ext):
    """Parsed content of one select-cache file (json / parquet / csv / tsv / npz / npy / pickle)."""
    if ext == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext in (".csv", ".tsv"):
        return pd.read_csv(p, sep="\t" if ext == ".tsv" else ",")
    if ext == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if ext == ".npy":
        return {p.name.split(".")[0]: np.load(p, allow_pickle=False)}   # the file stem is the key
    return pd.read_pickle(p)                                            # .pkl / .pickle: our own Phase C files


def audit_select(dev, lock):
    """First-use assertion of the lockbox: the Phase C caches hold dev ids only, and all of them."""
    if not SELECT_DIR.is_dir():
        die(f"select audit: {SELECT_DIR} missing — Phase C selection (select_blocks.py) has not run")
    files = sorted(Path(d) / f for d, _, fs in os.walk(SELECT_DIR) for f in fs)
    if not files:
        die(f"select audit: {SELECT_DIR} is empty")
    acc = dict(keys_seen=Counter(), hits=[])
    per_file, errors, tmp_errors, unscanned, blocked = {}, [], [], [], []
    for p in files:
        rel = str(p.relative_to(SELECT_DIR))
        is_tmp = p.name.endswith(".tmp")                         # write_atomic leftover of a clipped job
        name = p.name[:-4] if is_tmp else p.name
        ext = Path(name).suffix.lower()
        if p.stat().st_size == 0:
            unscanned.append(rel)
            continue
        if ext not in DATA_EXT:
            (blocked if ID_KEY.search(Path(name).stem) or "rxn" in name.lower() else unscanned).append(rel)
            continue
        n0 = len(acc["hits"])
        try:
            obj = read_cache(p, ext)
        except Exception as e:                                           # noqa: BLE001
            # a truncated *.tmp was never renamed, so selection never read it; any other unreadable file is fatal
            (tmp_errors if is_tmp else errors).append(f"{rel}: {type(e).__name__}: {e}")
            continue
        try:
            walk(obj, rel, acc)
        except Exception as e:                                           # noqa: BLE001
            errors.append(f"{rel}: {type(e).__name__}: {e}")
            continue
        per_file[rel] = sum(len(h[1]) for h in acc["hits"][n0:])
    all_ids, test_ids, fields, odd = set(), set(), Counter(), []
    for path, ids, o in acc["hits"]:
        all_ids.update(ids)
        if "test" in path.lower():
            test_ids.update(ids)
        fields[re.sub(r"\d+", "#", path.rsplit("/", 1)[-1])] += len(ids)
        odd.extend(f"{path}: {x!r}" for x in o[:3])
    s_dev, s_lock = set(dev), set(lock)
    leak = sorted(all_ids & s_lock)
    rec = dict(select_dir=str(SELECT_DIR), n_files=len(files), n_files_scanned=len(per_file),
               n_files_with_ids=sum(1 for v in per_file.values() if v), n_id_values=sum(per_file.values()),
               n_distinct_ids=len(all_ids), id_fields=dict(fields.most_common(20)),
               n_test_ids=len(test_ids), test_ids_equal_dev=test_ids == s_dev,
               lockbox_ids_found=len(leak), ids_equal_dev=all_ids == s_dev,
               unscanned_files=unscanned[:50], n_unscanned=len(unscanned), unreadable_tmp_files=tmp_errors[:20],
               non_numeric_id_values=odd[:20])
    print(f"select audit: {rec['n_files']} files, {rec['n_files_with_ids']} with ids, {len(all_ids)} distinct ids "
          f"(dev {len(s_dev)}), lockbox ids found {len(leak)}, fields {dict(fields.most_common(6))}", flush=True)
    if leak:
        die(f"LOCKBOX LEAK: {len(leak)} lockbox rxn_id found in the Phase C caches under {SELECT_DIR}: {leak[:20]} "
            f"(fields {dict(fields.most_common(10))}; if these are positional indices rather than rxn_ids, the cache "
            f"format differs from the assumed one)", code=3)
    bad = []
    if errors:
        bad.append(f"{len(errors)} cache files could not be parsed: {errors[:5]}")
    if blocked:
        bad.append(f"{len(blocked)} files named like id files have an unsupported format {sorted(DATA_EXT)}: "
                   f"{blocked[:5]}")
    if not all_ids:
        bad.append(f"no reaction-id field (keys matching {ID_KEY.pattern!r}) in any cache, so the audit would be "
                   f"vacuous; keys seen: {[k for k, _ in acc['keys_seen'].most_common(40)]}")
    elif all_ids != s_dev:
        bad.append(f"the ids in the caches are not exactly the dev ids: {len(s_dev - all_ids)} dev ids absent, "
                   f"{len(all_ids - s_dev)} ids outside dev ({sorted(all_ids - s_dev)[:10]}); fields "
                   f"{dict(fields.most_common(10))}")
    if bad:
        die("select audit:\n  " + "\n  ".join(bad))
    return rec


# ------------------------------------------------------------------------------------------ fits
def versions():
    import sklearn
    import xgboost
    return dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, numpy=np.__version__,
                pandas=pd.__version__)


def jsonable(x):
    return json.loads(json.dumps(x, default=str))


def unit_paths(arm, model, tgt):
    stem = f"{arm}__{model}__{tgt}"
    return UNIT_DIR / f"{stem}.parquet", UNIT_DIR / f"{stem}.json"


def load_unit(arm, model, tgt, fp, lock_ids, y_lock):
    """(record, predictions) of a finished unit, None if not finished; exits if made from other inputs."""
    pp, pj = unit_paths(arm, model, tgt)
    if not (pj.exists() and pp.exists()):
        return None
    rec = json.loads(pj.read_text())
    if rec.get("fingerprint") != fp:
        diff = sorted(k for k in set(fp) | set(rec.get("fingerprint") or {})
                      if (rec.get("fingerprint") or {}).get(k) != fp.get(k))
        die(f"{pj} exists but was made with other inputs (keys {diff}) — move {UNIT_DIR} away to recompute")
    p = pd.read_parquet(pp)
    if p["rxn_id"].astype(int).tolist() != lock_ids or not np.array_equal(p["y"].to_numpy(float), y_lock):
        die(f"{pp}: lockbox rows or targets differ from the current lockbox — move {UNIT_DIR} away")
    return rec, p


def fit_unit(arm, model, tgt, fp, X_dev, y_dev, X_lock, y_lock, lock_ids, n_jobs):
    from sklearn.model_selection import GridSearchCV, KFold
    est, grid = T.GRIDS[model]
    t0 = time.time()
    gs = GridSearchCV(T._make_pipe(est), grid, cv=KFold(R5.N_INNER, shuffle=True, random_state=R5.SEED),
                      scoring=SCORING, n_jobs=n_jobs, error_score="raise")
    gs.fit(X_dev, y_dev)
    mdl = gs.best_estimator_
    yhat, yfit = mdl.predict(X_lock), mdl.predict(X_dev)
    if not (np.isfinite(yhat).all() and np.isfinite(yfit).all()):
        die(f"{arm} {model} {tgt}: non-finite predictions")
    best = dict(gs.best_params_)
    rec = dict(fingerprint=fp, arm=arm, model=model, target=tgt, same_columns_as=None,
               n_train=len(y_dev), n_test=len(y_lock), n_features=X_dev.shape[1],
               best_params=best, edge_hits={k: T._edge_hit(k, v, grid[k]) for k, v in best.items()},
               cv_mae=float(-gs.best_score_), cv_mae_sd=float(gs.cv_results_["std_test_score"][gs.best_index_]),
               train=T.mets(y_dev, yfit), test=T.mets(y_lock, yhat), seconds=round(time.time() - t0, 1),
               n_jobs=n_jobs, host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID", "local"), finished=now())
    preds = pd.DataFrame({"rxn_id": np.asarray(lock_ids, dtype=np.int64), "arm": arm, "model": model, "target": tgt,
                          "y": y_lock, "yhat": yhat})
    return rec, preds


def save_unit(rec, preds):
    pp, pj = unit_paths(rec["arm"], rec["model"], rec["target"])
    R5.write_atomic(pp, lambda t: preds.to_parquet(t, index=False))
    R5.write_json(pj, rec)                                       # JSON last: its presence marks a finished unit


# ------------------------------------------------------------------------------------------ bootstrap + report
def pct_ci(b):
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def report(units, lock_ids, info):
    """units: {(arm, model, target): (record, lockbox predictions)} in plan order (model, target, arm)."""
    n = len(lock_ids)
    rng = np.random.default_rng(R5.SEED)
    B = rng.integers(0, n, size=(R5.N_BOOT, n))                 # one matrix for every arm / model / target
    boot_sha = hashlib.sha256(B.astype("<i8").tobytes()).hexdigest()
    mad_b, rows, boot, mae = {}, [], {}, {}
    for (arm, model, tgt), (rec, p) in units.items():
        y, yhat = p["y"].to_numpy(float), p["yhat"].to_numpy(float)
        if tgt not in mad_b:                                    # resampled mean absolute deviation of the targets
            yb = y[B]
            mad_b[tgt] = np.abs(yb - yb.mean(axis=1, keepdims=True)).mean(axis=1)
        bm = np.abs(y - yhat)[B].mean(axis=1)                   # resampled MAE
        boot[(arm, model, tgt)] = bm
        m = T.mets(y, yhat)
        if abs(m["nmae"] - R5.nmae(y, yhat)) > 1e-9:
            die(f"{arm} {model} {tgt}: train_ml_single.mets NMAE != rev5_common.nmae")
        mae[(arm, model, tgt)] = m["mae"]
        (mlo, mhi), (nlo, nhi) = pct_ci(bm), pct_ci(bm / mad_b[tgt])
        rows.append(dict(arm=arm, arm_features=info["arms"][arm]["source"], n_features=rec["n_features"],
                         model=model, headline=model == HEAD_MODEL, target=tgt, label=TARGET_LABEL[tgt],
                         n_train=rec["n_train"], n_test=rec["n_test"],
                         mae=m["mae"], mae_ci_lo=mlo, mae_ci_hi=mhi, nmae=m["nmae"], nmae_ci_lo=nlo, nmae_ci_hi=nhi,
                         r2=m["r2"], rmse=m["rmse"], pearson_r=m["pearson_r"],
                         train_mae=rec["train"]["mae"], cv_mae=rec["cv_mae"],
                         best_params=json.dumps(rec["best_params"], sort_keys=True),
                         edge_hits_total=int(sum(rec["edge_hits"].values())),
                         same_columns_as=rec["same_columns_as"] or ""))
    tab = pd.DataFrame(rows)
    paired = []
    for model in MODELS:
        for tgt in R5.TARGETS9:
            for arm in PAIRED:
                d = boot[(arm, model, tgt)] - boot[("BASE", model, tgt)]   # paired: same resampled rows
                pt = mae[(arm, model, tgt)] - mae[("BASE", model, tgt)]
                lo, hi = pct_ci(d)
                paired.append(dict(model=model, headline=model == HEAD_MODEL, target=tgt, label=TARGET_LABEL[tgt],
                                   arm=arm, vs="BASE", n_test=n, mae_arm=mae[(arm, model, tgt)],
                                   mae_base=mae[("BASE", model, tgt)], diff_mae=pt, diff_ci_lo=lo, diff_ci_hi=hi,
                                   rel_diff=pt / mae[("BASE", model, tgt)],
                                   frac_boot_diff_ge0=float(np.mean(d >= 0)),
                                   ci_excludes_0=bool(hi < 0 or lo > 0),
                                   same_columns_as_base=info["arms"][arm]["columns"] == info["arms"]["BASE"]["columns"],
                                   n_boot=R5.N_BOOT))
    pair = pd.DataFrame(paired)
    return tab, pair, dict(n_boot=R5.N_BOOT, seed=R5.SEED, n_lockbox=n, ci="95 % percentile (np.percentile, linear)",
                           indices="np.random.default_rng(SEED).integers(0, n_lockbox, size=(N_BOOT, n_lockbox)) "
                                   "over lockbox_ids.csv file order; shared by every arm / model / target",
                           index_sha256=boot_sha)


def headline(tab, pair):
    out = []
    for tgt in R5.TARGETS9:
        r = {a: tab[(tab.arm == a) & (tab.model == HEAD_MODEL) & (tab.target == tgt)].iloc[0] for a in ARMS}
        d = {a: pair[(pair.arm == a) & (pair.model == HEAD_MODEL) & (pair.target == tgt)].iloc[0] for a in PAIRED}
        out.append(dict(target=tgt, label=TARGET_LABEL[tgt],
                        **{f"{a}_{k}": float(r[a][k]) for a in ARMS for k in ("mae", "mae_ci_lo", "mae_ci_hi",
                                                                               "nmae", "r2")},
                        **{f"{a}_minus_BASE_{k}": float(d[a][k]) for a in PAIRED
                           for k in ("diff_mae", "diff_ci_lo", "diff_ci_hi")}))
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gates-only", action="store_true", help="run every gate and the row load, fit nothing")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        die("final_eval.py runs inside a SLURM job only (sbatch r5_lockbox_eval.sh), never on the login node")
    t0 = time.time()
    n_jobs = int(os.environ.get("R5_NJOBS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    print(f"[D-1] {now()} host {socket.gethostname()} job {os.environ.get('SLURM_JOB_ID')} n_jobs {n_jobs}\n"
          f"  code {HERE}\n  scratch {R5.SCRATCH}", flush=True)

    # 1. pre-registration, partition, first-use of the lockbox
    prereg = gate_prereg()
    rows, dev, lock = read_ids(R5.ROWS5), read_ids(DEV_IDS), read_ids(R5.LOCKBOX)
    part = gate_partition(rows, dev, lock)
    audit = audit_select(dev, lock)
    print(f"gates ok: HEAD {prereg['git_head'][:10]}, rows {part['n_rows']} = dev {part['n_dev']} + lockbox "
          f"{part['n_lockbox']}, select caches hold exactly the dev ids", flush=True)

    # 2. rows (gated: ok, NaN-free, hygiene; nothing dropped), arms
    ml, rows_info = T.load_ml(feat_path=R5.FEAT_G1, rows_path=str(R5.ROWS5), targets=R5.TARGETS9, geom="g1",
                              ext=True, ext_path=R5.FEAT_EXT)
    rid = ml["rxn_id"].astype(int).to_numpy()
    if rid.tolist() != rows:
        die("load_ml returned rows in another order than rows_rev5.csv")
    pos = pd.Series(np.arange(len(rid)), index=rid)
    i_dev, i_lock = pos.loc[dev].to_numpy(), pos.loc[lock].to_numpy()
    cols = {a: T.resolve_arm(ARM_SOURCE[a]) for a in ARMS}
    ext_sel_blocks = R5.ext_sel_blocks()
    absent = sorted({c for a in ARMS for c in cols[a]} - set(ml.columns))
    if absent:
        die(f"arm columns absent from the ML rows: {absent[:10]}")
    X = {a: ml[cols[a]].to_numpy(dtype=float) for a in ARMS}
    for a in ARMS:
        if not np.isfinite(X[a]).all():
            badc = [c for c, ok in zip(cols[a], np.isfinite(X[a]).all(axis=0)) if not ok]
            die(f"arm {a}: non-finite values in columns {badc[:10]}")
    Y = {t: ml[t].to_numpy(dtype=float) for t in R5.TARGETS9}
    info = dict(arms={a: dict(source=ARM_SOURCE[a], n_features=len(cols[a]), columns=list(cols[a])) for a in ARMS},
                ext_sel_blocks=ext_sel_blocks)
    print("arms: " + ", ".join(f"{a} = {ARM_SOURCE[a]} ({len(cols[a])})" for a in ARMS)
          + f"; EXT_SEL blocks {ext_sel_blocks}", flush=True)
    vers = versions()
    fp_common = dict(features_file=rows_info["features_file"], features_sha256=rows_info["features_sha256"],
                     ext_features_file=rows_info["ext_features_file"],
                     ext_features_sha256=rows_info["ext_features_sha256"], rows_sha256=rows_info["rows_sha256"],
                     dev_sha256=R5.sha256(DEV_IDS), lockbox_sha256=R5.sha256(R5.LOCKBOX),
                     prereg_rev5b_sha256=R5.sha256(R5.PREREG5B_JSON),
                     inner_cv=dict(kind="KFold", n_splits=R5.N_INNER, shuffle=True, random_state=R5.SEED),
                     scoring=SCORING, versions=vers)
    plan = [(m, t, a) for m in MODELS for t in R5.TARGETS9 for a in ARMS]
    if args.gates_only:
        done = sum(all(p.exists() for p in unit_paths(a, m, t)) for m, t, a in plan)
        print(f"--gates-only: {len(plan)} units ({done} cached), dev {len(i_dev)} x lockbox {len(i_lock)}; "
              f"nothing fitted or written", flush=True)
        return

    # 3. units: (model, target, arm), headline model first
    lock_ids = rid[i_lock].tolist()
    same_as = {a: next((a0 for a0 in ARMS[:ARMS.index(a)] if cols[a0] == cols[a]), None) for a in ARMS}
    units, timing = {}, {}
    for k, (model, tgt, arm) in enumerate(plan, 1):
        fp = jsonable(dict(fp_common, arm=arm, model=model, target=tgt, columns=cols[arm], grid=T.GRIDS[model][1]))
        y_dev, y_lock = Y[tgt][i_dev], Y[tgt][i_lock]
        got = load_unit(arm, model, tgt, fp, lock_ids, y_lock)
        how = "cached"
        if got is None and same_as[arm] is not None:           # identical columns -> identical (deterministic) fit
            rec0, p0 = units[(same_as[arm], model, tgt)]
            rec = dict(jsonable(rec0), fingerprint=fp, arm=arm, same_columns_as=same_as[arm], seconds=0.0,
                       finished=now())
            got = (rec, p0.assign(arm=arm))
            save_unit(*got)
            how = f"copied from {same_as[arm]} (same columns)"
        elif got is None:
            got = fit_unit(arm, model, tgt, fp, X[arm][i_dev], y_dev, X[arm][i_lock], y_lock, lock_ids, n_jobs)
            save_unit(*got)
            how = f"fitted in {got[0]['seconds']:.0f} s"
        units[(arm, model, tgt)] = got
        timing[f"{arm}__{model}__{tgt}"] = got[0]["seconds"]
        r = got[0]["test"]
        print(f"[{k:3d}/{len(plan)}] {model:8s} {tgt:18s} {arm:8s} ({got[0]['n_features']:3d}) lockbox MAE "
              f"{r['mae']:6.3f} NMAE {r['nmae']:.3f} r2 {r['r2']:+.3f}  {how}", flush=True)

    # 4. report (rebuilt from the caches every run)
    tab, pair, boot = report(units, lock_ids, info)
    preds = pd.concat([units[(a, m, t)][1] for m, t, a in plan], ignore_index=True)
    if preds.duplicated(["arm", "model", "target", "rxn_id"]).any():
        die("duplicate (arm, model, target, rxn_id) prediction")
    R5.write_atomic(OUT_PREDS, lambda t: preds.to_parquet(t, index=False))
    R5.write_atomic(OUT_CSV, lambda t: tab.to_csv(t, index=False))
    R5.write_atomic(OUT_PAIRED, lambda t: pair.to_csv(t, index=False))
    head = headline(tab, pair)
    R5.write_json(OUT_JSON, dict(
        phase="D-1 lockbox evaluation (docs/specs/REV5_FEATURES_FIGURES.md)", created=now(),
        job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), code=str(HERE), prereg=prereg,
        inputs=dict(rows_info, dev_file=str(DEV_IDS), dev_sha256=fp_common["dev_sha256"],
                    lockbox_file=str(R5.LOCKBOX), lockbox_sha256=fp_common["lockbox_sha256"],
                    prereg_rev5b_json=str(R5.PREREG5B_JSON), prereg_rev5b_sha256=fp_common["prereg_rev5b_sha256"]),
        partition=part, select_audit=audit, arms=info["arms"], ext_sel_blocks=ext_sel_blocks,
        arm_same_columns_as={a: s for a, s in same_as.items() if s is not None},
        models=list(MODELS), headline_model=HEAD_MODEL, targets=R5.TARGETS9,
        tuning=dict(search="GridSearchCV on all dev rows (dev_ids.csv order), refit on all dev rows",
                    grids="train_ml_single.GRIDS", pipeline="train_ml_single._make_pipe (StandardScaler X, "
                    "TransformedTargetRegressor StandardScaler y)", **fp_common["inner_cv"], scoring=SCORING,
                    n_jobs=n_jobs),
        bootstrap=boot, headline=head, unit_seconds=timing, versions=vers,
        outputs=dict(table=str(OUT_CSV), paired=str(OUT_PAIRED), predictions=str(OUT_PREDS), units=str(UNIT_DIR)),
        wall_seconds=round(time.time() - t0, 1)))

    print(f"\nD-1 headline: {HEAD_MODEL}, lockbox n = {len(lock_ids)}, bootstrap {R5.N_BOOT} (95 % percentile CI)")
    print(f"{'target':18s} {'BASE MAE':>18s} {'EXT_SEL MAE':>18s} {'EXT_SEL-BASE [CI]':>26s} "
          f"{'EXT_ALL-BASE [CI]':>26s}")
    for h in head:
        print(f"{h['target']:18s} {h['BASE_mae']:6.3f} [{h['BASE_mae_ci_lo']:.2f},{h['BASE_mae_ci_hi']:.2f}] "
              f"{h['EXT_SEL_mae']:6.3f} [{h['EXT_SEL_mae_ci_lo']:.2f},{h['EXT_SEL_mae_ci_hi']:.2f}] "
              f"{h['EXT_SEL_minus_BASE_diff_mae']:+7.3f} [{h['EXT_SEL_minus_BASE_diff_ci_lo']:+.2f},"
              f"{h['EXT_SEL_minus_BASE_diff_ci_hi']:+.2f}] "
              f"{h['EXT_ALL_minus_BASE_diff_mae']:+7.3f} [{h['EXT_ALL_minus_BASE_diff_ci_lo']:+.2f},"
              f"{h['EXT_ALL_minus_BASE_diff_ci_hi']:+.2f}]")
    print(f"\nsaved -> {OUT_CSV}\n         {OUT_PAIRED}\n         {OUT_JSON}\n         {OUT_PREDS}", flush=True)


if __name__ == "__main__":
    main()
