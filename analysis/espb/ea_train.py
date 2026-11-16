#!/usr/bin/env python3
"""ea_train.py — ESPB Phase A (docs/specs/ESPB.md 5-1, E-A): Espley single-reaction models per outer fold; the pair
prediction is f(a) - f(b) (collect.py). Copied from analysis/d1_pair/d1_f0.py (D1PAIR Phase F0) and changed per spec 2-3.

  python ea_train.py list                  # the 125 units (5 models x 5 targets x 5 outer folds), NN first
  python ea_train.py run <i> <n>           # slice i of n (idempotent; exit 3 = STOP)
  python ea_train.py nn <stage> <json>     # internal: one NN stage in a fresh process (tune | cv | final)

Rows = the training reactions of outer fold k (a and b of the training pairs of results/splits/T1__f<k>.json of GNN3R2,
fold -1 pairs included); predicted = the test reactions (a and b of the test pairs); features = ESPLEY46
($EB/P/X46.parquet); targets = d1, d2, inter, solv, barrier (results/P_reactions.csv). Inner CV = the 4 core inner folds
of results/P_inner_folds.csv (GroupKFold(4) over the training pairs by core = the GNN3 inner split, P-6), not KFold(5).
Ridge / KRR (RBF) / SVR (RBF) = train_ml_single._make_pipe + GridSearchCV(cv = the 4 inner folds, neg MAE,
refit=False) over eb_common.EA_GRIDS; best = lowest mean inner MAE (ties: grid order); final fit on every training
reaction; inner OOF = the best cell refitted on each inner fold. NN-2 / NN-4 = r6_nn.tune + r6_nn.fit_predict
unchanged: tune (seed 23, TF_ENABLE_ONEDNN_OPTS=0) on inner fold 1 (train -> val); cv (seed 22) = one process per inner
fold, fit_predict with validation_data = that inner val, predictions = the OOF; final = seeds 22, 23, 14, 1, 2, each in
its own process, trained on every training reaction with validation_data = the inner fold 1 val reactions (Hyperband
objective / fit history only; no training row removed), prediction = the five-seed mean. Every NN stage runs in its own
process (their scripts seed once per process). Units: $EB/units/EA/<model>__<target>__f<k>.{json,parquet}; each json
carries an input fingerprint; a record made from other inputs exits 3.
Gate: results/PREREG_ESPB.md + prereg_espb.json committed and every registered sha256 unchanged (prereg.gate).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eb_common as C  # noqa: E402

C.path_setup()
CODE_VERSION = "espb-ea-1"
N_JOBS = int(os.environ.get("ESPB_NJOBS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
TCOL = {"d1": "d1_kcal", "d2": "d2_kcal", "inter": "inter", "solv": "solv", "barrier": "barrier"}   # P_reactions.csv
_CACHE = {}


def inputs():
    """(X46 indexed by rxn_id, P_reactions indexed by rxn_id, P_inner_folds, pair ends indexed by pair_id, sha dict)."""
    if "inp" in _CACHE:
        return _CACHE["inp"]
    X = pd.read_parquet(C.X46).set_index("rxn_id")
    R = pd.read_csv(C.P_REACTIONS).set_index("rxn_id")
    inner = pd.read_csv(C.P_INNER)
    pairs = pd.read_csv(C.PAIRS, usecols=["pair_id", "a_rxn", "b_rxn"]).set_index("pair_id")
    sha = {"X46": C.sha256(C.X46), "P_reactions": C.sha256(C.P_REACTIONS), "P_inner_folds": C.sha256(C.P_INNER),
           "F_pairs_r": C.sha256(C.PAIRS), **{f"T1__f{k}": C.sha256(C.SPLIT_DIR / f"T1__f{k}.json") for k in range(C.N_FOLDS)}}
    _CACHE["inp"] = (X, R, inner, pairs, sha)
    return _CACHE["inp"]


def rows(k):
    """(training reactions, test reactions, inner fold of each training reaction) of outer fold k, with the A-3 check."""
    key = ("rows", k)
    if key in _CACHE:
        return _CACHE[key]
    X, R, inner, pairs, _ = inputs()
    unit = C.split_unit(k)
    ends = lambda ids: sorted({int(r) for r in pairs.loc[ids, ["a_rxn", "b_rxn"]].to_numpy().ravel()})   # noqa: E731
    rtr, rte = ends(unit["train"]), ends(unit["test"])
    if set(rtr) & set(rte):
        print(f"STOP (A-3): fold {k}: {len(set(rtr) & set(rte))} test reactions are also training reactions", file=sys.stderr)
        sys.exit(3)
    cmap = inner[inner.outer_fold == k].set_index("core").inner_fold
    j = R.loc[rtr, "core"].map(cmap)
    if j.isna().any():
        print(f"STOP: fold {k}: training reactions without an inner fold", file=sys.stderr)
        sys.exit(3)
    _CACHE[key] = (np.asarray(rtr, dtype=int), np.asarray(rte, dtype=int), j.to_numpy(int))
    return _CACHE[key]


def inner_cv(inner_fold):
    """the 4 inner folds as (train positions, val positions) into the training reactions; j = 0 is 'inner fold 1'"""
    return [(np.where(inner_fold != j)[0], np.where(inner_fold == j)[0]) for j in range(C.N_INNER)]


def unit_list():
    out = [dict(model=m, target=t, fold=k) for m in C.MODELS for t in C.TARGETS_EA for k in range(C.N_FOLDS)]
    only = [m for m in os.environ.get("ESPB_MODELS", "").split("+") if m]          # job packing only
    if only:
        out = [u for u in out if u["model"] in only]
    rank = {m: i for i, m in enumerate(["NN4", "NN2", "SVR", "KRR", "Ridge"])}       # long ones first
    return sorted(out, key=lambda u: (rank[u["model"]], C.TARGETS_EA.index(u["target"]), u["fold"]))


def uname(u):
    return f"{u['model']}__{u['target']}__f{u['fold']}"


def versions():
    from importlib.metadata import version
    return {k: version(k) for k in ("scikit-learn", "numpy", "pandas", "tensorflow", "keras", "keras_tuner")}


def estimator(model):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    return {"Ridge": Ridge(), "KRR": KernelRidge(kernel="rbf"), "SVR": SVR(kernel="rbf")}[model]


def fingerprint(u):
    X, R, inner, pairs, sha = inputs()
    rtr, rte, j = rows(u["fold"])
    nn = u["model"].startswith("NN")
    grid = "r6_nn.HP_STORAGE/HB" if nn else repr(C.EA_GRIDS[u["model"]])
    return hashlib.sha256(json.dumps(dict(u=u, sha=sha, cols=list(X.columns), target=TCOL[u["target"]], grid=grid,
                                          tr=C.list_sha([(r,) for r in rtr]), te=C.list_sha([(r,) for r in rte]),
                                          inner=C.list_sha([(int(v),) for v in j]),
                                          seeds=list(C.NN_FINAL_SEEDS) if nn else None, versions=versions(),
                                          code=CODE_VERSION), sort_keys=True).encode()).hexdigest()


def rec_path(u):
    return C.UNITS_EA / f"{uname(u)}.json"


def load_rec(u, fp):
    f = rec_path(u)
    if not (f.is_file() and f.with_suffix(".parquet").is_file()):
        return None
    rec = json.loads(f.read_text())
    if rec.get("fingerprint") != fp:
        print(f"STOP (A-1): unit record {f.name} was made from other inputs; move {f.parent} away", file=sys.stderr)
        sys.exit(3)
    if rec.get("gate_A2_finite") is False or rec.get("gate_A4_seeds") is False:     # a recorded failure stays a STOP
        raise SystemExit(f"STOP (A-2/A-4): {f.stem} (cached failure)")
    return rec


# ---------------------------------------------------------------- sklearn
def fit_sklearn(u):
    import train_ml_single as T
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV
    X, R, _, _, _ = inputs()
    rtr, rte, j = rows(u["fold"])
    Xtr, Xte = X.loc[rtr].to_numpy(float), X.loc[rte].to_numpy(float)
    ytr = R.loc[rtr, TCOL[u["target"]]].to_numpy(float)
    cv = inner_cv(j)
    grid = C.EA_GRIDS[u["model"]]
    gs = GridSearchCV(T._make_pipe(estimator(u["model"])), grid, cv=cv, scoring="neg_mean_absolute_error",
                      n_jobs=N_JOBS, refit=False, error_score="raise")
    gs.fit(Xtr, ytr)
    mean = -np.asarray(gs.cv_results_["mean_test_score"], float)
    b = int(np.argmin(mean))                                                 # ties: the first cell in grid order
    best = dict(gs.cv_results_["params"][b])
    folds = [float(-gs.cv_results_[f"split{i}_test_score"][b]) for i in range(C.N_INNER)]
    final = clone(T._make_pipe(estimator(u["model"]))).set_params(**best).fit(Xtr, ytr)
    yte = final.predict(Xte)
    oof = np.full(len(rtr), np.nan)
    for a, v in cv:
        m = clone(T._make_pipe(estimator(u["model"]))).set_params(**best).fit(Xtr[a], ytr[a])
        oof[v] = m.predict(Xtr[v])
    oof_mae = [float(np.mean(np.abs(oof[v] - ytr[v]))) for _, v in cv]
    cells = [dict(params=dict(p), inner_mae=float(m_)) for p, m_ in zip(gs.cv_results_["params"], mean)]
    edge = {k: bool(T._edge_hit(k, val, grid[k])) for k, val in best.items()}
    pred = pd.concat([pd.DataFrame(dict(rxn_id=rte, role="test", inner_fold=-1, yhat=yte)),
                      pd.DataFrame(dict(rxn_id=rtr, role="oof", inner_fold=j, yhat=oof))], ignore_index=True)
    info = dict(best_params=best, inner_cv_mae=float(mean[b]), inner_fold_mae=folds, oof_fold_mae=oof_mae,
                edge_hits=edge, grid=cells, n_cells=len(cells),
                train_mae=float(np.mean(np.abs(final.predict(Xtr) - ytr))))
    return pred, info


# ---------------------------------------------------------------- NN (r6_nn unchanged, one stage per process)
def nn_stage(stage, payload):
    import r6_nn as NN
    u = payload["u"]
    X, R, _, _, _ = inputs()
    rtr, rte, j = rows(u["fold"])
    tname = u["target"]
    y = R[TCOL[tname]]
    size = C.NN_SIZE[u["model"]]
    cv = inner_cv(j)
    tr0, va0 = rtr[cv[0][0]], rtr[cv[0][1]]                                 # inner fold 1
    if stage == "tune":
        NN.set_seeds(NN.TUNE_SEED)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            hps = NN.tune(X.loc[tr0], y.loc[tr0], X.loc[va0], y.loc[va0], size, tname, td, "kt")
        return dict(hps=hps)
    hps = payload["hps"]
    if stage == "cv":
        NN.set_seeds(C.CV_SEED)
        a, v = cv[int(payload["inner_fold"])]
        trj, vaj = rtr[a], rtr[v]
        pr, hist = NN.fit_predict(X.loc[trj], y.loc[trj], X.loc[vaj], y.loc[vaj], {"held": X.loc[vaj]}, size, hps, tname)
        return dict(rxn=[int(r) for r in vaj], pred=[float(p) for p in pr["held"]],
                    mae=float(np.mean(np.abs(pr["held"] - y.loc[vaj].to_numpy(float)))), final_loss=hist["loss"][-1])
    if stage == "final":
        NN.set_seeds(int(payload["seed"]))
        pr, hist = NN.fit_predict(X.loc[rtr], y.loc[rtr], X.loc[va0], y.loc[va0], {"test": X.loc[rte], "train": X.loc[rtr]},
                                  size, hps, tname)
        return dict(rxn=[int(r) for r in rte], pred=[float(p) for p in pr["test"]],
                    train_mae=float(np.mean(np.abs(pr["train"] - y.loc[rtr].to_numpy(float)))),
                    final_loss=hist["loss"][-1], final_val_loss=hist["val_loss"][-1])
    raise ValueError(stage)


def nn_call(stage, payload, wd, tag):
    wd.mkdir(parents=True, exist_ok=True)
    psha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    out = wd / f"{tag}.out.json"
    if out.is_file():
        rec = json.loads(out.read_text())
        if rec.get("payload_sha") == psha:
            return rec["result"]
    inp = wd / f"{tag}.in.json"
    C.write_json_safe(inp, payload)
    env = dict(os.environ)
    if stage == "tune":
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "ea_train.py"), "nn", stage, str(inp)], env=env, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nn {tag} failed: {r.stderr[-3000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    C.write_json_safe(out, dict(payload_sha=psha, result=res, seconds=time.time() - t0))
    return res


def fit_nn(u, fp):
    rtr, rte, j = rows(u["fold"])
    wd = C.NN_WORK / "EA" / uname(u)
    base = dict(u=u, fp=fp)
    hps = nn_call("tune", base, wd, "tune")["hps"]
    oof = pd.Series(np.nan, index=rtr)
    cv_mae = []
    for i in range(C.N_INNER):
        r = nn_call("cv", dict(base, hps=hps, inner_fold=i), wd, f"cv_j{i}")
        oof.loc[r["rxn"]] = r["pred"]
        cv_mae.append(r["mae"])
    seeds = {}
    for s in C.NN_FINAL_SEEDS:
        r = nn_call("final", dict(base, hps=hps, seed=s), wd, f"final_s{s}")
        if r["rxn"] != [int(x) for x in rte]:
            raise RuntimeError(f"nn final seed {s}: test reactions out of order")
        seeds[s] = (np.asarray(r["pred"], float), r["train_mae"])
    test = pd.DataFrame(dict(rxn_id=rte, role="test", inner_fold=-1,
                             yhat=np.mean([seeds[s][0] for s in C.NN_FINAL_SEEDS], axis=0)))
    for s in C.NN_FINAL_SEEDS:
        test[f"yhat_s{s}"] = seeds[s][0]
    ofr = pd.DataFrame(dict(rxn_id=rtr, role="oof", inner_fold=j, yhat=oof.to_numpy(float)))
    for s in C.NN_FINAL_SEEDS:
        ofr[f"yhat_s{s}"] = np.nan
    info = dict(best_params=hps, inner_cv_mae=float(np.mean(cv_mae)), inner_fold_mae=cv_mae, oof_fold_mae=cv_mae,
                edge_hits=None, grid="r6_nn.HP_STORAGE/HB", val_source="inner fold 1 (Hyperband); 4 inner folds (cv)",
                seeds=list(C.NN_FINAL_SEEDS), train_mae_per_seed={str(s): seeds[s][1] for s in C.NN_FINAL_SEEDS})
    return pd.concat([test, ofr], ignore_index=True), info


def run_unit(u):
    fp = fingerprint(u)
    if load_rec(u, fp) is not None:
        return "cached"
    t0 = time.time()
    pred, info = fit_nn(u, fp) if u["model"].startswith("NN") else fit_sklearn(u)
    rtr, rte, _ = rows(u["fold"])
    vals = pred.drop(columns=["rxn_id", "role", "inner_fold"])
    finite = bool(np.isfinite(pred.yhat.to_numpy(float)).all() and np.isfinite(info["inner_fold_mae"]).all())
    seeds_ok = (not u["model"].startswith("NN")) or all(
        np.isfinite(pred.loc[pred.role == "test", f"yhat_s{s}"].to_numpy(float)).all() for s in C.NN_FINAL_SEEDS)
    rec = dict(u, fingerprint=fp, n_train=len(rtr), n_test=len(rte), seconds=time.time() - t0, host=os.uname().nodename,
               versions=versions(), gate_A2_finite=finite, gate_A4_seeds=seeds_ok, value_columns=list(vals.columns), **info)
    C.write_atomic(C.UNITS_EA / f"{uname(u)}.parquet", lambda t: pred.to_parquet(t, index=False))
    C.write_json_safe(rec_path(u), rec)
    if not finite:
        raise SystemExit(f"STOP (A-2): {uname(u)} has NaN / inf predictions or inner MAE")
    if not seeds_ok:
        raise SystemExit(f"STOP (A-4): {uname(u)} misses a seed prediction")
    return f"done inner {rec['inner_cv_mae']:.3f}"


def _worker(u):
    t0 = time.time()
    try:
        return uname(u), run_unit(u), time.time() - t0
    except SystemExit as e:
        return uname(u), f"STOP {e.code}", time.time() - t0


def cmd_run(i, n):
    """slice i of n of the units; ESPB_WORKERS units in parallel spawn processes (NN stages: TF intra 2 / inter 1),
    sklearn GridSearchCV n_jobs = ESPB_NJOBS."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import prereg
    prereg.gate()
    us = unit_list()[i::n]
    workers = int(os.environ.get("ESPB_WORKERS", "1"))
    print(f"[EA run {i}/{n}] {len(us)} units, {workers} workers, n_jobs {N_JOBS}, versions {versions()}", flush=True)
    stop = []
    if workers == 1:
        for u in us:
            nm, st, dt = _worker(u)
            print(f"  {nm}: {st} ({dt:.0f} s)", flush=True)
            if st.startswith("STOP"):
                stop.append(nm)
                break
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_worker, u) for u in us]
            for fu in as_completed(futs):
                nm, st, dt = fu.result()
                print(f"  {nm}: {st} ({dt:.0f} s)", flush=True)
                if st.startswith("STOP"):
                    stop.append(nm)
                    ex.shutdown(wait=True, cancel_futures=True)    # running units finish (atomic), queued ones cancelled
                    break
    return 3 if stop else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "nn":
        print(json.dumps(nn_stage(a[1], json.loads(Path(a[2]).read_text()))))
        sys.exit(0)
    if cmd == "list":
        for u in unit_list():
            print(uname(u))
        sys.exit(0)
    if cmd == "run":
        sys.exit(cmd_run(int(a[1]), int(a[2])))
    raise SystemExit(__doc__)
