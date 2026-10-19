#!/usr/bin/env python3
"""d1_f0.py — D1PAIR Phase F0 (§3-4): the D0 models f0, trained once on all 4,963 rev 6 appendix A rows, then frozen.

  python d1_f0.py list                  # the 60 units (10 targets x 6 models), NN first
  python d1_f0.py fit <i> <n>           # slice i of n: Ridge / KRR / SVR / XGB fitted + saved; NN tuned + 5-fold CV
  python d1_f0.py nn <stage> <json>     # internal: one NN stage in a fresh process (tune | cv | fit)
  python d1_f0.py report                # f0_cv.csv, representative per target, gate F0-1 -> f0_report.json (exit 3 = STOP)
  python d1_f0.py predict <i> <n>       # after Phase F: predictions on every D1 row of X182_d1.parquet (NN: final fit)
  python d1_f0.py collect               # f0_d1.parquet (D1 rxn_id x 10 targets x 6 models) + sha256 in f0_report.json

Rows = the union of rxn_id in results_rev6/S_splits_appendixA.csv (must be 4,963); features = the 182 columns of
X182_all.parquet (rev6_common.our_columns(train_ml_single.FEATURE_SETS)); targets = the rev 6 r6_train.tcol columns of
S_targets.parquet. Ridge / KRR / SVR / XGB = train_ml_single.GRIDS + _make_pipe + GridSearchCV(cv=KFold(5,
shuffle=True, random_state=22), neg MAE) on all rows; the best_estimator_ is saved (joblib); CV MAE = -best_score_.
NN-2 / NN-4 = r6_nn.tune + r6_nn.fit_predict unchanged: a fixed 10 % (train_test_split(rows, test_size=0.1,
random_state=22)) is validation_data only (Hyperband objective and the fit history), never trained on; tuning with
seed 23 (TF_ENABLE_ONEDNN_OPTS=0, as r6_train); CV MAE = KFold(5, shuffle=True, random_state=22) on the 90 % training
rows, each fold fitted with fit_predict (validation_data = the held-out fold), as r6_train's cv stage; the final fit
(seed 22) runs in `predict` with X_pred = the D1 rows (fit_predict returns predictions, not the model).
Every NN stage runs in its own process (their scripts seed once per process). Cache: $D1PAIR_SCRATCH/f0/; each unit
record carries an input fingerprint; a record made from other inputs exits 3.
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
import d1pair_common as C  # noqa: E402

CODE_VERSION = "d1pair-f0-1"
F0 = C.SCRATCH / "f0"
D0_X182 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/xtb_r6/X182_all.parquet")      # = rev6_common.X182_ALL
D0_TARGETS = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/S/S_targets.parquet")        # = rev6_common.S_TARGETS
D0_SPLITS_A = C.ESP_DIR / "results_rev6" / "S_splits_appendixA.csv"                # = rev6_common.S_SPLITS_A
SHA_EXPECT = {"X182_all": "e79791b76ea6257cb9d19f0f7e6cd2a16d467b6d2fe9c40ca205f1717f8be254",
              "S_targets": "6b02630fd73dd37d6fdb9792063e3094e8e8d8b2cd1353bc13b1ad91b90272bf",
              "S_splits_appendixA": "b08a700fb47d32fca7e7c9086328a8073a7e87e8830cf8817fa149f5fc777a78"}
GKEY = {"Ridge": "Ridge", "KRR": "KRR_rbf", "SVR": "SVR_rbf", "XGB": "XGB"}
N_JOBS = int(os.environ.get("D1PAIR_NJOBS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
_CACHE = {}


def inputs():
    """(X DataFrame 4,963 x 182 indexed by rxn_id, Y DataFrame, rows list, sha dict); exits 3 on a sha256 / row error."""
    if "inp" in _CACHE:
        return _CACHE["inp"]
    sha = {"X182_all": C.sha256(D0_X182), "S_targets": C.sha256(D0_TARGETS), "S_splits_appendixA": C.sha256(D0_SPLITS_A)}
    if sha != SHA_EXPECT:
        print(f"STOP: D0 input sha256 {sha} != {SHA_EXPECT}", file=sys.stderr); sys.exit(3)
    import train_ml_single as T
    cols = C.R6.our_columns(T.FEATURE_SETS)
    rows = sorted(int(r) for r in pd.read_csv(D0_SPLITS_A).rxn_id.unique())
    if len(rows) != C.N_D0:
        print(f"STOP: {len(rows)} f0 rows != {C.N_D0}", file=sys.stderr); sys.exit(3)
    X = pd.read_parquet(D0_X182).set_index("rxn_id")
    Y = pd.read_parquet(D0_TARGETS).set_index("rxn_id")
    if set(rows) - set(X.index) or set(rows) - set(Y.index):
        print("STOP: f0 rows missing in X182_all / S_targets", file=sys.stderr); sys.exit(3)
    X = X.loc[rows, cols]
    Y = Y.loc[rows, [C.D0_TCOL[t] for t in C.TARGETS]]
    if X.isna().any().any() or Y.isna().any().any():
        print("STOP: NaN in the f0 rows", file=sys.stderr); sys.exit(3)
    _CACHE["inp"] = (X, Y, rows, sha)
    return _CACHE["inp"]


def nn_split(rows):
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(np.asarray(rows), test_size=0.1, random_state=C.CV_SEED)
    return [int(r) for r in tr], [int(r) for r in va]


def unit_list():
    out = [dict(target=t, model=m) for t in C.TARGETS for m in C.MODELS]
    return sorted(out, key=lambda u: (not u["model"].startswith("NN"), C.MODELS.index(u["model"]), C.TARGETS.index(u["target"])))


def uname(u):
    return f"{u['target']}__{u['model']}"


def versions():
    from importlib.metadata import version
    return {k: version(k) for k in ("scikit-learn", "xgboost", "numpy", "pandas", "tensorflow", "keras", "keras_tuner")}


def fingerprint(u):
    import train_ml_single as T
    X, Y, rows, sha = inputs()
    grid = "r6_nn.HP_STORAGE/HB" if u["model"].startswith("NN") else repr(T.GRIDS[GKEY[u["model"]]][1])
    return hashlib.sha256(json.dumps(dict(u=u, sha=sha, cols=list(X.columns), target=C.D0_TCOL[u["target"]], grid=grid,
                                          rows=C.list_sha([(r,) for r in rows]), versions=versions(), code=CODE_VERSION),
                                     sort_keys=True).encode()).hexdigest()


def rec_path(u):
    return F0 / "units" / f"{uname(u)}.json"


def load_rec(u, fp):
    f = rec_path(u)
    if not f.is_file():
        return None
    rec = json.loads(f.read_text())
    if rec.get("fingerprint") != fp:
        print(f"STOP: f0 record {f.name} was made from other inputs; move {f.parent} away", file=sys.stderr)
        sys.exit(3)
    return rec


# ---------------------------------------------------------------- sklearn
def fit_sklearn(u):
    import joblib
    import train_ml_single as T
    from sklearn.model_selection import GridSearchCV, KFold
    X, Y, rows, _ = inputs()
    y = Y[C.D0_TCOL[u["target"]]].to_numpy(float)
    est, grid = T.GRIDS[GKEY[u["model"]]]
    gs = GridSearchCV(T._make_pipe(est), grid, cv=KFold(5, shuffle=True, random_state=C.CV_SEED),
                      scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise")
    gs.fit(X.to_numpy(float), y)
    mp = F0 / "models" / f"{uname(u)}.joblib"
    C.write_atomic(mp, lambda t: joblib.dump(gs.best_estimator_, t))
    i = gs.best_index_
    folds = [float(-gs.cv_results_[f"split{k}_test_score"][i]) for k in range(5)]
    best = {k: (v if isinstance(v, (str, type(None))) else float(v)) for k, v in gs.best_params_.items()}
    edge = {k: bool(T._edge_hit(k, v, grid[k])) for k, v in gs.best_params_.items()}
    return dict(cv_mae=float(-gs.best_score_), cv_fold_mae=folds, best_params=best, edge_hits=edge,
                model_path=str(mp), model_sha256=C.sha256(mp),
                train_mae=float(np.mean(np.abs(gs.best_estimator_.predict(X.to_numpy(float)) - y))))


# ---------------------------------------------------------------- NN (r6_nn unchanged, one stage per process)
def nn_stage(stage, payload):
    import r6_nn as NN
    u = payload["u"]
    X, Y, rows, _ = inputs()
    tname = C.D0_TCOL[u["target"]]
    y = Y[tname]
    size = C.NN_SIZE[u["model"]]
    tr, va = nn_split(rows)
    if stage == "tune":
        NN.set_seeds(NN.TUNE_SEED)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            hps = NN.tune(X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], size, tname, td, "kt")
        return dict(hps=hps)
    hps = payload["hps"]
    NN.set_seeds(C.CV_SEED)
    if stage == "cv":
        from sklearn.model_selection import KFold
        trn = np.asarray(tr)
        errs = []
        for a, b in KFold(5, shuffle=True, random_state=C.CV_SEED).split(trn):
            pr, _ = NN.fit_predict(X.loc[trn[a]], y.loc[trn[a]], X.loc[trn[b]], y.loc[trn[b]], {"held": X.loc[trn[b]]},
                                   size, hps, tname)
            errs.append(float(np.mean(np.abs(pr["held"] - y.loc[trn[b]].to_numpy(float)))))
        return dict(cv_fold_mae=errs, val_mae=float(np.mean(errs)))
    if stage == "fit":
        Xd1 = pd.read_parquet(payload["x_d1"]).set_index("rxn_id")[list(X.columns)]
        pr, hist = NN.fit_predict(X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], {"d1": Xd1, "d0": X}, size, hps, tname)
        return dict(pred_d1=[float(v) for v in pr["d1"]], rxn_d1=[int(r) for r in Xd1.index],
                    train_mae=float(np.mean(np.abs(pr["d0"][[rows.index(r) for r in tr]] - y.loc[tr].to_numpy(float)))),
                    final_loss=hist["loss"][-1], final_val_loss=hist["val_loss"][-1])
    raise ValueError(stage)


def nn_call(stage, payload, wd):
    wd.mkdir(parents=True, exist_ok=True)
    psha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    out = wd / f"{stage}.out.json"
    if out.is_file():
        rec = json.loads(out.read_text())
        if rec.get("payload_sha") == psha:
            return rec["result"]
    inp = wd / f"{stage}.in.json"
    C.write_json_safe(inp, payload)
    env = dict(os.environ)
    if stage == "tune":
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "d1_f0.py"), "nn", stage, str(inp)], env=env, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nn {stage} failed: {r.stderr[-3000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    C.write_json_safe(out, dict(payload_sha=psha, result=res, seconds=time.time() - t0))
    return res


def fit_nn(u, fp):
    wd = F0 / "nn_work" / uname(u)
    base = dict(u=u, fp=fp)
    hps = nn_call("tune", base, wd)["hps"]
    cv = nn_call("cv", dict(base, hps=hps), wd)
    return dict(cv_mae=cv["val_mae"], cv_fold_mae=cv["cv_fold_mae"], best_params=hps, edge_hits=None,
                val_source="nn_5fold_cv_on_90pct_train")


def run_unit(u):
    fp = fingerprint(u)
    if load_rec(u, fp) is not None:
        return "cached"
    t0 = time.time()
    info = fit_nn(u, fp) if u["model"].startswith("NN") else fit_sklearn(u)
    X, _, rows, sha = inputs()
    rec = dict(u, fingerprint=fp, n_train=len(rows), inputs_sha256=sha, seconds=time.time() - t0,
               host=os.uname().nodename, versions=versions(), **info)
    C.write_json_safe(rec_path(u), rec)
    return f"done cv {rec['cv_mae']:.3f}"


def _worker(u):
    t0 = time.time()
    try:
        return uname(u), run_unit(u), time.time() - t0
    except SystemExit as e:
        return uname(u), f"STOP exit {e.code}", time.time() - t0


def cmd_fit(i, n):
    """slice i of n of the units; D1PAIR_WORKERS units in parallel spawn processes (NN stages: TF intra 2 / inter 1),
    sklearn GridSearchCV n_jobs = D1PAIR_NJOBS."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    only = [m for m in os.environ.get("D1PAIR_F0_MODELS", "").split("+") if m]     # e.g. NN2+NN4 (sbatch --export splits at commas)
    us = [u for u in unit_list() if not only or u["model"] in only][i::n]
    workers = int(os.environ.get("D1PAIR_WORKERS", "1"))
    print(f"[f0 fit {i}/{n}] {len(us)} units, {workers} workers, n_jobs {N_JOBS}, versions {versions()}", flush=True)
    stop = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for nm, st, dt in ex.map(_worker, us):
            print(f"  {nm}: {st} ({dt:.0f} s)", flush=True)
            if st.startswith("STOP"):
                stop.append(nm)
    return 3 if stop else 0


# ---------------------------------------------------------------- report / predict
def cmd_report():
    rows = []
    for u in unit_list():
        rec = load_rec(u, fingerprint(u))
        rows.append(dict(target=u["target"], model=u["model"], cv_mae=None if rec is None else rec["cv_mae"],
                         cv_fold_mae=None if rec is None else json.dumps(rec["cv_fold_mae"]),
                         best_params=None if rec is None else json.dumps(rec["best_params"]),
                         edge_hits=None if rec is None else json.dumps(rec.get("edge_hits")),
                         train_mae=None if rec is None else rec.get("train_mae"),
                         seconds=None if rec is None else rec["seconds"]))
    df = pd.DataFrame(rows)
    C.write_atomic(C.RES / "f0_cv.csv", lambda t: df.to_csv(t, index=False))
    missing = df[df.cv_mae.isna()][["target", "model"]].to_dict("records")
    rep = dict(n_rows=C.N_D0, inputs_sha256=inputs()[3], missing=missing, representative={}, gate_F0_1={})
    for t, g in df.dropna(subset=["cv_mae"]).groupby("target"):
        b = g.sort_values(["cv_mae", "model"]).iloc[0]
        lim = C.F0_FACTOR * C.F0_REF_MAE[t]
        rep["representative"][t] = b.model
        rep["gate_F0_1"][t] = dict(model=b.model, cv_mae=float(b.cv_mae), limit=lim, ok=bool(b.cv_mae <= lim))
    rep["stop"] = bool(missing) or not all(v["ok"] for v in rep["gate_F0_1"].values()) or \
        len(rep["gate_F0_1"]) != len(C.TARGETS)
    C.write_json_safe(C.RES / "f0_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else 0


def pred_path(u):
    return F0 / "pred_d1" / f"{uname(u)}.parquet"


def cmd_predict(i, n):
    import joblib
    if not C.X182_D1.is_file():
        print(f"{C.X182_D1} missing (Phase F)", file=sys.stderr); return 2
    X, Y, rows, _ = inputs()
    Xd1 = pd.read_parquet(C.X182_D1).set_index("rxn_id")[list(X.columns)]
    xsha = C.sha256(C.X182_D1)
    for u in unit_list()[i::n]:
        fp = fingerprint(u)
        rec = load_rec(u, fp)
        if rec is None:
            print(f"STOP: {uname(u)} not fitted", file=sys.stderr); return 3
        pp = pred_path(u)
        pj = pp.with_suffix(".json")
        if pp.is_file() and pj.is_file() and json.loads(pj.read_text()).get("x_d1_sha256") == xsha \
                and json.loads(pj.read_text()).get("unit_fingerprint") == fp \
                and json.loads(pj.read_text()).get("model_sha256") == rec.get("model_sha256"):
            print(f"  {uname(u)}: cached", flush=True); continue
        t0 = time.time()
        if u["model"].startswith("NN"):
            res = nn_call("fit", dict(u=u, fp=fp, hps=rec["best_params"], x_d1=str(C.X182_D1), x_d1_sha256=xsha),
                          F0 / "nn_work" / uname(u))
            yhat, rid, extra = np.asarray(res["pred_d1"]), res["rxn_d1"], dict(train_mae=res["train_mae"])
        else:
            m = joblib.load(rec["model_path"])
            yhat, rid, extra = m.predict(Xd1.to_numpy(float)), [int(r) for r in Xd1.index], {}
        df = pd.DataFrame(dict(rxn_id=rid, target=u["target"], model=u["model"], yhat=yhat))
        C.write_atomic(pp, lambda t: df.to_parquet(t, index=False))
        C.write_json_safe(pp.with_suffix(".json"), dict(x_d1_sha256=xsha, unit_fingerprint=fp, model_sha256=rec.get("model_sha256"),
                                                        n=len(df), seconds=time.time() - t0, **extra))
        print(f"  {uname(u)}: {len(df)} D1 predictions ({time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_collect():
    xsha = C.sha256(C.X182_D1)
    parts = []
    for u in unit_list():
        pp = pred_path(u)
        meta = json.loads(pp.with_suffix(".json").read_text()) if pp.with_suffix(".json").is_file() else {}
        if meta.get("x_d1_sha256") != xsha or meta.get("unit_fingerprint") != fingerprint(u):
            print(f"STOP: {pp.name} missing or made from another X182_d1", file=sys.stderr); return 3
        parts.append(pd.read_parquet(pp))
    df = pd.concat(parts, ignore_index=True)
    out = C.RES / "f0_d1.parquet"
    C.write_atomic(out, lambda t: df.to_parquet(t, index=False))
    rep = json.loads((C.RES / "f0_report.json").read_text())
    rep.update(f0_d1=dict(path=str(out), sha256=C.sha256(out), n_rows=len(df), x_d1_sha256=xsha))
    C.write_json_safe(C.RES / "f0_report.json", rep)
    print(json.dumps(rep["f0_d1"], indent=1))
    return 0


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
    fn = {"fit": lambda: cmd_fit(int(a[1]), int(a[2])), "report": cmd_report,
          "predict": lambda: cmd_predict(int(a[1]), int(a[2])), "collect": cmd_collect}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
