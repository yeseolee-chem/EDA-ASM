#!/usr/bin/env python3
"""r6_train.py — rev 6 our-pipeline units (arms OUR-O / ESP-O; spec §3-2, §8, §9). Espley-pinned env (env_espley).

  python r6_train.py list <group>               # the unit list of a group (stable order)
  python r6_train.py run <group> <i> <n>        # slice i of n of the group's units (idempotent, exit 3 = STOP)
  python r6_train.py nn <stage> <unit-json>     # internal: one NN stage in a fresh process (tune | cv | fit)

Groups (all on the §2-3 split; rows = S_splits.csv, features / targets = the Phase S files):
  B     OUR-O, ESP-O x 6 models x d1, d2, Int, ΔE‡ x 5 seeds on C            (Phase B)
  A     OUR-O x 6 models x elst..cds + c_ghost, ESP-O x 6 models x elst..cds, 5 seeds, on C   (Phase A main; d1/d2 = B)
  Call  OUR-O x 6 models x d1, d2, Int, ΔE‡ x 5 seeds on C_all              (C_all sensitivity)
  Nocref OUR-O x 6 models x d1, d2, Int, ΔE‡ x 5 seeds on C_nocref = C minus constrained_ref rows (retrained)
  AppA  OUR-O x 6 models x 8 channels x 5 seeds, Protocol A rows (appendix)
Pipeline = rev 4/5 Protocol A: Ridge / KRR / SVR / XGB = train_ml_single._make_pipe + GRIDS, GridSearchCV(cv =
KFold(5, shuffle=True, random_state=seed), neg MAE) on train_s; validation MAE = -best_score_. NN-2 / NN-4 = Espley's
NN (r6_nn.py): Hyperband (their search space and settings, seed 23, TF_ENABLE_ONEDNN_OPTS=0 as their tuning script) on
train_s with validation_data = val_s (the second half of the hold-out ∩ rows: their NN uses it the same way); the
validation MAE for the representative rule = 5-fold CV (KFold(5, shuffle=True, random_state=seed)) of the tuned NN on
train_s; final fit on train_s (their 500-epoch routine, validation_data = val_s for the history only), seed = the split
seed set at process start. Every NN stage runs in its own process (their scripts seed once per process).
Cache: $R6_SCRATCH/units/<group>/<unit>.{json,parquet}; the json carries an input fingerprint (data sha256, columns,
grid, rows sha256, package versions, code version); a cached unit with another fingerprint exits 3.
Gate: PREREG_REV6.md / prereg_rev6.json committed and the Phase S inputs unchanged (rev6_common.gate_prereg).
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
import rev6_common as R6  # noqa: E402

CODE_VERSION = "r6-train-1"
UNITS = R6.SCRATCH / "units"
N_JOBS = int(os.environ.get("R6_NJOBS", "1"))


# ---------------------------------------------------------------- inputs
_CACHE = {}


def splits():
    if "spl" not in _CACHE:
        _CACHE["spl"] = pd.read_csv(R6.RES6 / "S_splits.csv")
        _CACHE["splA"] = pd.read_csv(R6.S_SPLITS_A)
    return _CACHE["spl"], _CACHE["splA"]


def X_of(feat):
    key = f"X_{feat}"
    if key not in _CACHE:
        if feat == R6.FEAT_OUR:
            import train_ml_single as T
            cols = R6.our_columns(T.FEATURE_SETS)
            df = pd.read_parquet(R6.X182_ALL)
            _CACHE[key] = (df.set_index("rxn_id")[cols], R6.sha256(R6.X182_ALL))
        else:
            S = json.loads((R6.RES6 / "S.json").read_text())
            df = pd.read_parquet(R6.S_X_ESP)
            _CACHE[key] = (df.set_index("rxn_id")[S["esp_cols_46"]], R6.sha256(R6.S_X_ESP))
    return _CACHE[key]


def y_all():
    if "y" not in _CACHE:
        _CACHE["y"] = (pd.read_parquet(R6.S_TARGETS).set_index("rxn_id"), R6.sha256(R6.S_TARGETS))
    return _CACHE["y"]


def rows_of(u):
    """(train, test, val) rxn_id lists of a unit, in split order."""
    spl, splA = splits()
    if u["rowset"] == "appA":
        d = splA[splA.seed == u["seed"]]
        get = lambda p: d[d.part == p].sort_values("order").rxn_id.tolist()      # noqa: E731
    else:
        d = spl[(spl.seed == u["seed"]) & spl[u["rowset"]]]
        get = lambda p: d[d.part == p].sort_values("order").rxn_id.tolist()      # noqa: E731
    tr, te, va = get("train"), get("test"), get("val")
    assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va)), "split overlap"
    return tr, te, va


# ---------------------------------------------------------------- units
def unit_list(group):
    B, CH = list(R6.TARGETS_B), list(R6.CHANNELS8)
    six = [c for c in CH if c not in ("d1", "d2")]
    spec = {"B": [(a, B, "C") for a in ("OUR-O", "ESP-O")],
            "A": [("OUR-O", six + ["c_ghost"], "C"), ("ESP-O", six, "C")],
            "Call": [("OUR-O", B, "C_all")],
            "Nocref": [("OUR-O", B, "C_nocref")],
            "AppA": [("OUR-O", CH, "appA")]}[group]
    out = []
    for arm, tks, rowset in spec:
        for t in tks:
            for m in R6.MODELS:
                for s in R6.SEEDS:
                    out.append(dict(group=group, arm=arm, target=t, model=m, seed=s, rowset=rowset,
                                    feat=R6.ARMS[arm][0]))
    # NN units first (the long ones), then the rest; stable
    return sorted(out, key=lambda u: (not u["model"].startswith("NN"), R6.MODELS.index(u["model"]), u["arm"],
                                      u["target"], u["seed"]))


def uname(u):
    return f"{u['arm']}__{u['target']}__{u['model']}__s{u['seed']}__{u['rowset']}"


def tcol(t):
    return {**R6.TARGETS_B, **R6.TARGETS_A}[t]


def versions():
    from importlib.metadata import version
    return {k: version(k) for k in ("scikit-learn", "xgboost", "numpy", "pandas", "tensorflow", "keras",
                                    "keras_tuner")}


def fingerprint(u, tr, te, va):
    import train_ml_single as T
    X, xsha = X_of(u["feat"])
    _, ysha = y_all()
    grid = "espley_nn" if u["model"].startswith("NN") else repr(T.GRIDS[{"KRR": "KRR_rbf", "SVR": "SVR_rbf"}
                                                                        .get(u["model"], u["model"])][1])
    h = hashlib.sha256(json.dumps(dict(u=u, X=xsha, y=ysha, cols=list(X.columns), target=tcol(u["target"]),
                                       grid=grid, tr=tr, te=te, va=va, versions=versions(), code=CODE_VERSION),
                                  sort_keys=True).encode()).hexdigest()
    return h


def write_unit(u, rec, pred):
    d = UNITS / u["group"]
    R6.write_atomic(d / f"{uname(u)}.parquet", lambda t: pred.to_parquet(t, index=False))
    R6.write_json(d / f"{uname(u)}.json", rec)


def cached(u, fp):
    f = UNITS / u["group"] / f"{uname(u)}.json"
    if not f.is_file() or not (UNITS / u["group"] / f"{uname(u)}.parquet").is_file():
        return False
    rec = json.loads(f.read_text())
    if rec.get("fingerprint") != fp:
        print(f"STOP: cached unit {f.name} was made from other inputs (fingerprint {rec.get('fingerprint', '')[:12]} "
              f"!= {fp[:12]}); move {f.parent} away", file=sys.stderr)
        sys.exit(3)
    return True


def run_sklearn(u, tr, te, va):
    import train_ml_single as T
    from sklearn.model_selection import GridSearchCV, KFold
    X, _ = X_of(u["feat"])
    Y, _ = y_all()
    y = Y[tcol(u["target"])]
    key = {"KRR": "KRR_rbf", "SVR": "SVR_rbf"}.get(u["model"], u["model"])
    est, grid = T.GRIDS[key]
    gs = GridSearchCV(T._make_pipe(est), grid, cv=KFold(5, shuffle=True, random_state=u["seed"]),
                      scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise")
    gs.fit(X.loc[tr].to_numpy(float), y.loc[tr].to_numpy(float))
    p = gs.predict(X.loc[te].to_numpy(float))
    best = {k: (v if isinstance(v, (str, type(None))) else float(v)) for k, v in gs.best_params_.items()}
    edge = {k: bool(T._edge_hit(k, v, grid[k])) for k, v in gs.best_params_.items()}
    return p, dict(best_params=best, edge_hits=edge, val_mae=float(-gs.best_score_), val_source="inner_cv_best_score")


def nn_stage(stage, payload):
    """One NN stage in this (fresh) process. payload: unit + rows + paths."""
    import r6_nn as NN
    u, tr, te, va = payload["u"], payload["tr"], payload["te"], payload["va"]
    X, _ = X_of(u["feat"])
    Y, _ = y_all()
    tname = tcol(u["target"])
    y = Y[tname]
    size = R6.NN_SIZE[u["model"]]
    Xtr, ytr, Xva, yva = X.loc[tr], y.loc[tr], X.loc[va], y.loc[va]
    if stage == "tune":
        NN.set_seeds(NN.TUNE_SEED)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            hps = NN.tune(Xtr, ytr, Xva, yva, size, tname, td, "kt")
        return dict(hps=hps)
    hps = payload["hps"]
    NN.set_seeds(u["seed"])
    if stage == "cv":
        from sklearn.model_selection import KFold
        trn = np.asarray(tr)
        errs = []
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=u["seed"]).split(trn)):
            pr, _ = NN.fit_predict(X.loc[trn[a]], y.loc[trn[a]], X.loc[trn[b]], y.loc[trn[b]],
                                   {"held": X.loc[trn[b]]}, size, hps, tname)
            errs.append(float(np.mean(np.abs(pr["held"] - y.loc[trn[b]].to_numpy(float)))))
        return dict(cv_fold_mae=errs, val_mae=float(np.mean(errs)))
    if stage == "fit":
        pr, hist = NN.fit_predict(Xtr, ytr, Xva, yva, {"test": X.loc[te]}, size, hps, tname)
        return dict(pred=[float(v) for v in pr["test"]], final_loss=hist["loss"][-1],
                    final_val_loss=hist["val_loss"][-1])
    raise ValueError(stage)


def nn_call(stage, payload, wd):
    """Run nn_stage in a fresh python process (their per-process seeding); tune with oneDNN off as hyp_tuning.py."""
    wd.mkdir(parents=True, exist_ok=True)
    inp, out = wd / f"{stage}.in.json", wd / f"{stage}.out.json"
    if out.is_file():
        rec = json.loads(out.read_text())
        if rec.get("payload_sha") == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest():
            return rec["result"]
    R6.write_json(inp, payload)
    env = dict(os.environ)
    if stage == "tune":
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "r6_train.py"), "nn", stage, str(inp)], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nn {stage} failed: {r.stderr[-2000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    R6.write_json(out, dict(payload_sha=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                            result=res, seconds=time.time() - t0))
    return res


def run_nn(u, tr, te, va):
    wd = UNITS / u["group"] / "nn_work" / uname(u)
    base = dict(u=u, tr=tr, te=te, va=va)
    hps = nn_call("tune", base, wd)["hps"]
    cv = nn_call("cv", dict(base, hps=hps), wd)
    fit = nn_call("fit", dict(base, hps=hps), wd)
    return np.asarray(fit["pred"]), dict(best_params=hps, edge_hits=None, val_mae=cv["val_mae"],
                                         cv_fold_mae=cv["cv_fold_mae"], val_source="nn_5fold_cv_on_train",
                                         final_loss=fit["final_loss"], final_val_loss=fit["final_val_loss"])


def run_unit(u):
    tr, te, va = rows_of(u)
    fp = fingerprint(u, tr, te, va)
    if cached(u, fp):
        return "cached"
    t0 = time.time()
    Y, _ = y_all()
    yte = Y.loc[te, tcol(u["target"])].to_numpy(float)
    p, info = (run_nn if u["model"].startswith("NN") else run_sklearn)(u, tr, te, va)
    pred = pd.DataFrame(dict(group=u["group"], arm=u["arm"], model=u["model"], target=u["target"], seed=u["seed"],
                             rowset=u["rowset"], rxn_id=te, y_true=yte, y_pred=p))
    rec = dict(u, fingerprint=fp, n_train=len(tr), n_test=len(te), n_val=len(va),
               train_sha256=hashlib.sha256(",".join(map(str, tr)).encode()).hexdigest(),
               test_sha256=hashlib.sha256(",".join(map(str, te)).encode()).hexdigest(),
               mae=float(np.mean(np.abs(p - yte))), nmae=R6.nmae(yte, p), seconds=time.time() - t0,
               host=os.uname().nodename, versions=versions(), **info)
    write_unit(u, rec, pred)
    return f"done {rec['mae']:.3f}"


def _worker(u):
    t0 = time.time()
    try:
        return uname(u), run_unit(u), time.time() - t0
    except SystemExit as e:                                  # fingerprint STOP inside a worker
        return uname(u), f"STOP exit {e.code}", time.time() - t0


def cmd_run(group, i, n):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    R6.gate_prereg()
    units = unit_list(group)[i::n]
    workers = int(os.environ.get("R6_WORKERS", "1"))
    print(f"[{group} {i}/{n}] {len(units)} units, {workers} workers, versions {versions()}", flush=True)
    stop = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for k, (nm, st, dt) in enumerate(ex.map(_worker, units)):
            print(f"  [{k + 1}/{len(units)}] {nm}: {st} ({dt:.0f} s)", flush=True)
            if st.startswith("STOP"):
                stop.append(nm)
    return 3 if stop else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["nn"]:
        payload = json.loads(Path(a[2]).read_text())
        print(json.dumps(nn_stage(a[1], payload)))
        sys.exit(0)
    if a[:1] == ["list"]:
        for u in unit_list(a[1]):
            print(uname(u))
        sys.exit(0)
    if a[:1] == ["run"]:
        sys.exit(cmd_run(a[1], int(a[2]), int(a[3])))
    raise SystemExit(__doc__)
