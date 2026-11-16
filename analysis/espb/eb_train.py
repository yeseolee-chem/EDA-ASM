#!/usr/bin/env python3
"""eb_train.py — ESPB Phase B (docs/specs/ESPB.md 5-2, E-B): Espley-type models trained on the pair differences directly
(antisymmetric pair models of D1PAIR arm C). Copied from analysis/d1_pair/d1_train.py and changed per spec 2-3.

  python eb_train.py list                    # the 100 units (arm EB x 5 models x 4 targets x 5 outer folds)
  python eb_train.py run <i> <n>             # slice i of n of the units (idempotent; exit 3 = STOP)
  python eb_train.py nn <stage> <json>       # internal: one NN stage in a fresh process (tune | cv | fit)

Pairs = the GNN3R2 data set (results/F_pairs_r.csv + S_folds_r.csv of analysis/gnn3r2, 8,344 pairs); outer fold k:
training = the train list of results/splits/T1__f<k>.json, test = its test list. Target y = the channel ΔΔ (strain,
inter, solv, barrier = gnn3.channels.labels_to_channels of the dd columns = gnn3.metrics.channel_truth); nothing is
subtracted. Features = ESPLEY46 ($EB/P/X46.parquet). ESPB_MODELS=<m+m> restricts a run to some models (job packing only).
Scale (§3-2) on the (inner) training pairs: z = [dx / s_delta, (xbar - mu) / s], no mean removed from dx; columns with
s < 1e-12 dropped; y / s_y without mean removal; recomputed in every inner fold.
g models (§3-3), all antisymmetric g(-dx, xbar) = -g(dx, xbar), g(0, xbar) = 0:
  Ridge  Ridge(alpha, fit_intercept=False) on z_delta only
  KRR    KernelRidge(alpha, kernel="precomputed"), K(p,q) = [k(dx_p, dx_q) - k(dx_p, -dx_q)] k(xbar_p, xbar_q), RBF k
  SVR    SVR(C, epsilon, kernel="precomputed") on the same kernel, trained both ways ((dx, y), (-dx, -y)),
         prediction 1/2 [s(dx) - s(-dx)]
  NN-2/4 shared sub-network h = r6_nn._layers (ELU + L2 + dropout, Espley structure), g = h([dx, xbar]) - h([-dx, xbar]);
         Hyperband (r6_nn.HP_STORAGE, r6_nn.HB, seed 23) on inner fold 1, CV MAE = the 4 inner folds refitted (500
         epochs, batch 32, shuffle off, seed 22), final fit on all training pairs with seeds 22, 23, 14, 1, 2 (one
         process each), prediction = the five-seed mean
Tuning: GroupKFold(4), group = core, on the training pairs; best = lowest mean inner MAE (kcal/mol; ties: grid order).
Gate M-1 per unit (NN: per seed) on 200 random test pairs: |g(dx) + g(-dx)| <= 1e-6 and |g(0)| <= 1e-6 kcal/mol, else
exit 3. Cache: $EB/units/EB/<model>__<target>__f<k>.{json,parquet}; the json carries an input fingerprint (exit 3 on a
mismatch). Gate: results/PREREG_ESPB.md + prereg_espb.json committed and every registered sha256 unchanged (prereg.gate).
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
CODE_VERSION = "espb-eb-1"
UNITS = C.UNITS_EB
N_JOBS = int(os.environ.get("ESPB_NJOBS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
_C = {}


# ================================================================ inputs
def data():
    """(S pairs table with the channel ΔΔ columns, X46 indexed by rxn_id, sha dict)."""
    if "d" in _C:
        return _C["d"]
    from gnn3.channels import labels_to_channels
    F = pd.read_csv(C.PAIRS)
    A = pd.read_csv(C.S_FOLDS, usecols=["pair_id", "fold_T1", "in_P_main"])
    S = F[["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "core", "pos"]
          + [f"dd_{t}" for t in C.D1_TCOL]].merge(A, on="pair_id", how="left", validate="1:1")
    ch = labels_to_channels(S[[f"dd_{t}" for t in C.D1_TCOL]].to_numpy(float))
    for i, c in enumerate(C.CHANNELS):
        S[c] = ch[:, i]
    X = pd.read_parquet(C.X46).set_index("rxn_id")
    sha = {"F_pairs_r": C.sha256(C.PAIRS), "S_folds_r": C.sha256(C.S_FOLDS), "X46": C.sha256(C.X46),
           **{f"T1__f{k}": C.sha256(C.SPLIT_DIR / f"T1__f{k}.json") for k in range(C.N_FOLDS)}}
    _C["d"] = (S, X, sha)
    return _C["d"]


def unit_list():
    out = [dict(arm="EB", model=m, target=t, fold=k) for m in C.MODELS for t in C.TARGETS_EB for k in range(C.N_FOLDS)]
    only = [m for m in os.environ.get("ESPB_MODELS", "").split("+") if m]
    if only:
        out = [u for u in out if u["model"] in only]
    rank = {m: i for i, m in enumerate(["NN4", "NN2", "SVR", "KRR", "Ridge"])}       # long ones first
    return sorted(out, key=lambda u: (rank[u["model"]], C.TARGETS_EB.index(u["target"]), u["fold"]))


def uname(u):
    return f"{u['model']}__{u['target']}__f{u['fold']}"


def masks(u):
    """(train index, test index) of the S rows for a unit (split-file order), with the in-unit leak checks."""
    S = data()[0]
    row = {q: i for i, q in enumerate(S.pair_id)}
    unit = C.split_unit(u["fold"])
    tr = np.asarray([row[q] for q in unit["train"]])
    te = np.asarray([row[q] for q in unit["test"]])
    if set(S.loc[te, "core"]) & set(S.loc[tr, "core"]):
        raise SystemExit(f"STOP: {uname(u)} test core also in training")
    if (set(S.loc[te, "a_job"]) | set(S.loc[te, "b_job"])) & (set(S.loc[tr, "a_job"]) | set(S.loc[tr, "b_job"])):
        raise SystemExit(f"STOP: {uname(u)} test reaction also in training")
    return tr, te


def pair_x(idx):
    """(x_a, x_b) arrays (n, 46) of the S rows idx."""
    S, X = data()[0], data()[1]
    return X.loc[S.loc[idx, "a_rxn"]].to_numpy(float), X.loc[S.loc[idx, "b_rxn"]].to_numpy(float)


def target_y(u, idx):
    return data()[0].loc[idx, u["target"]].to_numpy(float)


def inner_folds(tr):
    from sklearn.model_selection import GroupKFold
    S = data()[0]
    return [(tr[a], tr[b]) for a, b in GroupKFold(C.N_INNER).split(tr, groups=S.loc[tr, "core"].to_numpy())]


# ================================================================ scaling (§3-2)
class Scale:
    def __init__(self, xa, xb, y):
        d, m = xa - xb, (xa + xb) / 2
        self.sd = d.std(axis=0)
        self.mu, self.s = m.mean(axis=0), m.std(axis=0)
        self.kd, self.km = self.sd >= C.SCALE_EPS, self.s >= C.SCALE_EPS
        self.sy = float(np.std(y))

    def z(self, xa, xb):
        d, m = xa - xb, (xa + xb) / 2
        return d[:, self.kd] / self.sd[self.kd], (m[:, self.km] - self.mu[self.km]) / self.s[self.km]

    def info(self):
        return dict(n_dropped_delta=int((~self.kd).sum()), n_dropped_xbar=int((~self.km).sum()), s_y=self.sy)


def prep(tr_idx, te_idx, u):
    xa, xb = pair_x(tr_idx)
    ytr = target_y(u, tr_idx)
    sc = Scale(xa, xb, ytr)
    zd, zm = sc.z(xa, xb)
    ta, tb = pair_x(te_idx)
    td, tm = sc.z(ta, tb)
    return sc, (zd, zm, ytr / sc.sy, ytr), (td, tm)


# ================================================================ kernels
def sqd(A, B):
    an, bn = (A * A).sum(1), (B * B).sum(1)
    return np.maximum(an[:, None] + bn[None, :] - 2.0 * (A @ B.T), 0.0)


def dist3(zd1, zm1, zd2, zm2):
    """(|d1 - d2|^2, |d1 + d2|^2, |m1 - m2|^2) for the antisymmetric kernel."""
    return sqd(zd1, zd2), sqd(zd1, -zd2), sqd(zm1, zm2)


def kern(D, gd, gm):
    Dm, Dp, Dx = D
    return (np.exp(-gd * Dm) - np.exp(-gd * Dp)) * np.exp(-gm * Dx)


# ================================================================ grid workers (one inner fold x one grid cell)
def _krr_cell(Dtr, Dte, ytr_s, yte, sy, gd, gm, alphas):
    from sklearn.kernel_ridge import KernelRidge
    Ktr, Kte = kern(Dtr, gd, gm), kern(Dte, gd, gm)
    out = []
    for a in alphas:
        m = KernelRidge(alpha=a, kernel="precomputed").fit(Ktr, ytr_s)
        out.append(float(np.mean(np.abs(m.predict(Kte) * sy - yte))))
    return out


def _svr_cell(Dtr, Dte, ytr_s, yte, sy, gd, gm, cs, epss):
    from sklearn.svm import SVR
    K, Kte = kern(Dtr, gd, gm), kern(Dte, gd, gm)
    K2 = np.block([[K, -K], [-K, K]])
    y2 = np.r_[ytr_s, -ytr_s]
    Kte2 = np.c_[Kte, -Kte]
    out = []
    for c in cs:
        for e in epss:
            m = SVR(kernel="precomputed", C=c, epsilon=e, cache_size=2000).fit(K2, y2)
            g = 0.5 * (m.predict(Kte2) - m.predict(-Kte2))
            out.append(float(np.mean(np.abs(g * sy - yte))))
    return out


def grid_cells(model):
    if model == "Ridge":
        return [dict(alpha=a) for a in C.EB_RIDGE_ALPHA]
    if model == "KRR":
        return [dict(alpha=a, gamma_delta=gd, gamma_xbar=gm) for gd in C.GAMMAS for gm in C.GAMMAS
                for a in C.EB_KRR_ALPHA]
    if model == "SVR":
        return [dict(C=c, epsilon=e, gamma_delta=gd, gamma_xbar=gm) for gd in C.GAMMAS for gm in C.GAMMAS
                for c in C.SVR_C for e in C.SVR_EPS]
    raise KeyError(model)


def _svr_one(Dtr, Dte, ytr_s, yte, sy, gd, gm, c, e):
    """one SVR grid cell on one inner fold (same fit and score as _svr_cell, finer parallel unit)."""
    return _svr_cell(Dtr, Dte, ytr_s, yte, sy, gd, gm, [c], [e])[0]


_MMDIR = []


def mm(arrs):
    """write arrays to .npy files in the current temp folder and reopen them as read-only np.memmap: joblib then sends
    workers a file reference instead of hashing and dumping each large distance matrix for every task (same values)."""
    out = []
    for a in arrs:
        f = os.path.join(_MMDIR[-1], f"m{len(os.listdir(_MMDIR[-1]))}.npy")
        np.save(f, a)
        out.append(np.load(f, mmap_mode="r"))
    return tuple(out)


def tune_grid(u, tr):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"), prefix="ebm_") as d:
        _MMDIR.append(d)
        try:
            return _tune_grid(u, tr)
        finally:
            _MMDIR.pop()


def _tune_grid(u, tr):
    """inner GroupKFold(4) MAE of every grid cell -> (cells, mean MAE per cell, per-fold MAEs)."""
    from joblib import Parallel, delayed
    cells = grid_cells(u["model"])
    per_fold = []
    if u["model"] == "SVR":
        # load balance: one parallel task per (inner fold, grid cell) = 4 x 768 SVR fits (identical fits / scores to
        # _svr_cell; the per-gamma-pair grouping left most workers idle behind the slow large-C cells)
        folds = []
        for itr, ite in inner_folds(tr):
            sc, (zd, zm, ys, _), (td, tm) = prep(itr, ite, u)
            folds.append((mm(dist3(zd, zm, zd, zm)), mm(dist3(td, tm, zd, zm)), ys, target_y(u, ite), sc.sy))
        res = Parallel(n_jobs=N_JOBS, max_nbytes="1M")(
            delayed(_svr_one)(*folds[k][:5], c["gamma_delta"], c["gamma_xbar"], c["C"], c["epsilon"])
            for k in range(len(folds)) for c in cells)
        pf = np.asarray(res, float).reshape(len(folds), len(cells))
        return cells, pf.mean(axis=0), pf
    for itr, ite in inner_folds(tr):
        sc, (zd, zm, ys, _), (td, tm) = prep(itr, ite, u)
        yte = target_y(u, ite)
        if u["model"] == "Ridge":
            from sklearn.linear_model import Ridge
            per_fold.append([float(np.mean(np.abs(Ridge(alpha=c["alpha"], fit_intercept=False).fit(zd, ys).predict(td) * sc.sy
                                                  - yte))) for c in cells])
        elif u["model"] == "KRR":
            Dtr, Dte = mm(dist3(zd, zm, zd, zm)), mm(dist3(td, tm, zd, zm))
            gg = [(gd, gm) for gd in C.GAMMAS for gm in C.GAMMAS]
            res = Parallel(n_jobs=N_JOBS, max_nbytes="1M")(delayed(_krr_cell)(Dtr, Dte, ys, yte, sc.sy, gd, gm,
                                                                               C.EB_KRR_ALPHA) for gd, gm in gg)
            per_fold.append([v for r in res for v in r])               # order = grid_cells order
        else:
            raise KeyError(u["model"])
    pf = np.asarray(per_fold)
    return cells, pf.mean(axis=0), pf


def fit_final(u, tr, te, best):
    """fit on all training pairs with the best cell; returns (g_test kcal, g_train kcal, M-1 values, scale info)."""
    sc, (zd, zm, ys, ytr), (td, tm) = prep(tr, te, u)
    rng = np.random.default_rng(C.SEED + int(hashlib.sha256(uname(u).encode()).hexdigest()[:8], 16) % 10**6)
    k = rng.choice(len(te), size=min(C.ANTISYM_N, len(te)), replace=False)
    sd, sm = td[k], tm[k]
    if u["model"] == "Ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=best["alpha"], fit_intercept=False).fit(zd, ys)
        g = lambda a, b: m.predict(a)                                # noqa: E731
    elif u["model"] in ("KRR", "SVR"):
        Dtr = dist3(zd, zm, zd, zm)
        K = kern(Dtr, best["gamma_delta"], best["gamma_xbar"])
        if u["model"] == "KRR":
            from sklearn.kernel_ridge import KernelRidge
            m = KernelRidge(alpha=best["alpha"], kernel="precomputed").fit(K, ys)
            g = lambda a, b: m.predict(kern(dist3(a, b, zd, zm), best["gamma_delta"], best["gamma_xbar"]))   # noqa: E731
        else:
            from sklearn.svm import SVR
            m = SVR(kernel="precomputed", C=best["C"], epsilon=best["epsilon"], cache_size=2000).fit(
                np.block([[K, -K], [-K, K]]), np.r_[ys, -ys])

            def g(a, b):
                Ka = kern(dist3(a, b, zd, zm), best["gamma_delta"], best["gamma_xbar"])
                return 0.5 * (m.predict(np.c_[Ka, -Ka]) - m.predict(np.c_[-Ka, Ka]))
    else:
        raise KeyError(u["model"])
    gte, gtr = g(td, tm) * sc.sy, g(zd, zm) * sc.sy
    anti = np.abs(g(sd, sm) + g(-sd, sm)) * sc.sy
    zero = np.abs(g(np.zeros_like(sd), sm)) * sc.sy
    return gte, gtr, dict(max_sum=float(anti.max()) if len(k) else 0.0, max_zero=float(zero.max()) if len(k) else 0.0,
                          n=int(len(k))), sc.info()


# ================================================================ NN (one stage per process)
def nn_model(n_in_d, n_in_m, size, get, target):
    import r6_nn as NN
    from keras import layers, models
    from keras.optimizers import Adam
    from tensorflow import keras
    D = n_in_d + n_in_m
    hin = keras.Input(shape=(D,), name="h_input")
    h = models.Model(hin, NN._layers(hin, size, get, target)["out_lay"], name="h")
    inp = keras.Input(shape=(D,), name="input")
    sign = np.r_[-np.ones(n_in_d), np.ones(n_in_m)].astype("float32")
    neg = layers.Lambda(lambda t: t * sign, name="flip_dx")(inp)
    out = layers.Subtract(name="g")([h(inp), h(neg)])
    model = models.Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=get("learning_rate")), loss="mean_absolute_error",
                  metrics=["mean_absolute_error"])
    return model


def nn_builder(n_in_d, n_in_m, size, target):
    """r6_nn.make_builder's hyperparameter definitions (same names, values and order) around the antisymmetric model."""
    import r6_nn as NN

    def builder(hp):
        spec = {}
        for k in range(1, size + 1):
            for key in NN.HP_STORAGE:
                if key == "learning_rate":
                    spec[key] = hp.Choice(key, values=NN.HP_STORAGE[key])
                else:
                    spec[f"{key}_{k}"] = hp.Choice(f"{key}_{k}", values=NN.HP_STORAGE[key])
        return nn_model(n_in_d, n_in_m, size, spec.__getitem__, target)
    return builder


def nn_stage(stage, payload):
    import r6_nn as NN
    u = payload["u"]
    tr, te = masks(u)
    size = C.NN_SIZE[u["model"]]
    tname = f"dd_{u['target']}"
    folds = inner_folds(tr)
    if stage == "tune":
        NN.set_seeds(NN.TUNE_SEED)
        import keras_tuner as kt
        itr, ite = folds[0]
        sc, (zd, zm, ys, _), (td, tm) = prep(itr, ite, u)
        yv = target_y(u, ite) / sc.sy
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tuner = kt.Hyperband(nn_builder(zd.shape[1], zm.shape[1], size, tname), project_name="kt", directory=d, **NN.HB)
            tuner.search(x=np.c_[zd, zm], y=ys, validation_data=(np.c_[td, tm], yv), epochs=NN.SEARCH_EPOCHS, verbose=0)
            best = tuner.get_best_hyperparameters(num_trials=1)[0]
        return dict(hps=dict(best.values))
    hps = payload["hps"]
    if stage == "cv":
        NN.set_seeds(C.CV_SEED)
        errs = []
        for itr, ite in folds:
            sc, (zd, zm, ys, _), (td, tm) = prep(itr, ite, u)
            yte = target_y(u, ite)
            m = nn_model(zd.shape[1], zm.shape[1], size, hps.__getitem__, tname)
            m.fit(x=np.c_[zd, zm], y=ys, validation_data=(np.c_[td, tm], yte / sc.sy), **NN.FIT)
            errs.append(float(np.mean(np.abs(m.predict(np.c_[td, tm], verbose=0).ravel() * sc.sy - yte))))
        return dict(cv_fold_mae=errs, val_mae=float(np.mean(errs)))
    if stage == "fit":
        NN.set_seeds(int(payload["seed"]))                   # ESPB: final fit per seed 22, 23, 14, 1, 2
        sc, (zd, zm, ys, ytr), (td, tm) = prep(tr, te, u)
        m = nn_model(zd.shape[1], zm.shape[1], size, hps.__getitem__, tname)
        h = m.fit(x=np.c_[zd, zm], y=ys, **NN.FIT)
        rng = np.random.default_rng(C.SEED + int(hashlib.sha256(uname(u).encode()).hexdigest()[:8], 16) % 10**6)
        k = rng.choice(len(te), size=min(C.ANTISYM_N, len(te)), replace=False)
        P = lambda a, b: m.predict(np.c_[a, b], verbose=0, batch_size=256).ravel().astype(float)   # noqa: E731
        gte, gtr = P(td, tm) * sc.sy, P(zd, zm) * sc.sy
        anti = np.abs(P(td[k], tm[k]) + P(-td[k], tm[k])) * sc.sy
        zero = np.abs(P(np.zeros_like(td[k]), tm[k])) * sc.sy
        return dict(g_test=gte.tolist(), g_train=gtr.tolist(), antisym=dict(max_sum=float(anti.max()),
                    max_zero=float(zero.max()), n=int(len(k))), scale=sc.info(), final_loss=float(h.history["loss"][-1]))
    raise ValueError(stage)


def nn_call(stage, payload, wd, tag=None):
    tag = tag or stage
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
    r = subprocess.run([sys.executable, str(HERE / "eb_train.py"), "nn", stage, str(inp)], env=env, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nn {tag} failed: {r.stderr[-3000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    C.write_json_safe(out, dict(payload_sha=psha, result=res, seconds=time.time() - t0))
    return res


# ================================================================ one unit
def versions():
    from importlib.metadata import version
    return {k: version(k) for k in ("scikit-learn", "numpy", "pandas", "tensorflow", "keras", "keras_tuner")}


def fingerprint(u, tr, te):
    S, X, sha = data()
    nn = u["model"].startswith("NN")
    grid = "nn" if nn else repr(grid_cells(u["model"]))
    return hashlib.sha256(json.dumps(dict(u=u, sha=sha, cols=list(X.columns), grid=grid,
                                          tr=C.list_sha([(i,) for i in S.loc[tr, "pair_id"]]),
                                          te=C.list_sha([(i,) for i in S.loc[te, "pair_id"]]),
                                          seeds=list(C.NN_FINAL_SEEDS) if nn else None, versions=versions(),
                                          code=CODE_VERSION), sort_keys=True).encode()).hexdigest()


def cached(u, fp):
    f = UNITS / f"{uname(u)}.json"
    if not (f.is_file() and f.with_suffix(".parquet").is_file()):
        return False
    rec = json.loads(f.read_text())
    if rec.get("fingerprint") != fp:
        print(f"STOP (B-1): cached unit {f.name} was made from other inputs; move {f.parent} away", file=sys.stderr)
        sys.exit(3)
    if rec.get("gate_M1") is False:                       # a recorded M-1 failure stays a STOP on every rerun
        raise SystemExit(f"STOP: gate M-1 {f.stem} (cached failure): {rec.get('antisym')}")
    if rec.get("gate_B2_finite") is False or rec.get("gate_B4_seeds") is False:     # likewise B-2 / B-4
        raise SystemExit(f"STOP (B-2/B-4): {f.stem} (cached failure)")
    return True


def run_unit(u):
    tr, te = masks(u)
    fp = fingerprint(u, tr, te)
    if cached(u, fp):
        return "cached"
    t0 = time.time()
    S = data()[0]
    y_te = target_y(u, te)
    y_tr = target_y(u, tr)
    info = dict(n_train=len(tr), n_test=len(te), n_test_P_main=int(S.loc[te, "in_P_main"].sum()),
                n_train_anchor=int((S.loc[tr, "ref_kind"] == "anchor").sum()))
    seed_cols = {}
    if u["model"].startswith("NN"):
        wd = C.NN_WORK / "EB" / uname(u)
        base = dict(u=u, fp=fp)
        hps = nn_call("tune", base, wd)["hps"]
        cv = nn_call("cv", dict(base, hps=hps), wd)
        fits = {s: nn_call("fit", dict(base, hps=hps, seed=s), wd, f"fit_s{s}") for s in C.NN_FINAL_SEEDS}
        gtes = {s: np.asarray(fits[s]["g_test"], float) for s in C.NN_FINAL_SEEDS}
        gte = np.mean([gtes[s] for s in C.NN_FINAL_SEEDS], axis=0)
        gtr = np.mean([np.asarray(fits[s]["g_train"], float) for s in C.NN_FINAL_SEEDS], axis=0)
        seed_cols = {f"yhat_s{s}": gtes[s] for s in C.NN_FINAL_SEEDS}
        anti = dict(max_sum=float(np.max([fits[s]["antisym"]["max_sum"] for s in C.NN_FINAL_SEEDS])),     # NaN propagates
                    max_zero=float(np.max([fits[s]["antisym"]["max_zero"] for s in C.NN_FINAL_SEEDS])),
                    n=fits[C.NN_FINAL_SEEDS[0]]["antisym"]["n"],
                    per_seed={str(s): fits[s]["antisym"] for s in C.NN_FINAL_SEEDS})
        scl = fits[C.NN_FINAL_SEEDS[0]]["scale"]
        info.update(best_params=hps, inner_cv_mae=cv["val_mae"], inner_fold_mae=cv["cv_fold_mae"],
                    seeds=list(C.NN_FINAL_SEEDS), final_loss_per_seed={str(s): fits[s]["final_loss"] for s in C.NN_FINAL_SEEDS})
    else:
        cells, mean, pf = tune_grid(u, tr)
        b = int(np.argmin(mean))
        best = cells[b]
        gte, gtr, anti, scl = fit_final(u, tr, te, best)
        edge = {}
        for k, v in best.items():
            vals = sorted({c[k] for c in cells})
            edge[k] = bool(len(vals) > 1 and v in (vals[0], vals[-1]))
        info.update(best_params=best, inner_cv_mae=float(mean[b]), inner_fold_mae=pf[:, b].tolist(), edge_hits=edge,
                    grid_mae=[round(float(v), 6) for v in mean], n_cells=len(cells))
    info.update(antisym=anti, scale=scl)
    if not (anti["max_sum"] <= C.ANTISYM_TOL and anti["max_zero"] <= C.ANTISYM_TOL):
        info["gate_M1"] = False
    finite = bool(np.isfinite(gte).all() and np.isfinite(info["inner_fold_mae"]).all()
                  and all(np.isfinite(v).all() for v in seed_cols.values()))
    seeds_ok = (not u["model"].startswith("NN")) or len(seed_cols) == len(C.NN_FINAL_SEEDS)
    pred = pd.DataFrame(dict(pair_id=S.loc[te, "pair_id"].to_numpy(), ref_kind=S.loc[te, "ref_kind"].to_numpy(),
                             fold=u["fold"], arm=u["arm"], model=u["model"], target=u["target"], y=y_te, yhat=gte,
                             **seed_cols))
    rec = dict(u, fingerprint=fp, **info, gate_B2_finite=finite, gate_B4_seeds=seeds_ok,
               train_mae=float(np.mean(np.abs(gtr - y_tr))),
               test_mae_P_main=float(np.mean(np.abs(gte - y_te)[S.loc[te, "in_P_main"].to_numpy(bool)])),
               train_pair_sha256=C.list_sha([(i,) for i in S.loc[tr, "pair_id"]]), seconds=time.time() - t0,
               host=os.uname().nodename, versions=versions())
    rec.setdefault("gate_M1", True)
    C.write_atomic(UNITS / f"{uname(u)}.parquet", lambda t: pred.to_parquet(t, index=False))
    C.write_json_safe(UNITS / f"{uname(u)}.json", rec)
    if not rec["gate_M1"]:
        raise SystemExit(f"STOP: gate M-1 {uname(u)}: {rec.get('antisym')}")
    if not finite:
        raise SystemExit(f"STOP (B-2): {uname(u)} has NaN / inf predictions or inner MAE")
    if not seeds_ok:
        raise SystemExit(f"STOP (B-4): {uname(u)} misses a seed prediction")
    return f"done {rec['test_mae_P_main']:.3f}"


def _worker(u):
    t0 = time.time()
    try:
        return uname(u), run_unit(u), time.time() - t0
    except SystemExit as e:
        return uname(u), f"STOP {e.code}", time.time() - t0


def cmd_run(i, n):
    import prereg
    prereg.gate()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    us = unit_list()[i::n]
    if os.environ.get("ESPB_ORDER") == "reverse":         # helper job: same slice from the other end (cache skips done units)
        us = us[::-1]
    workers = int(os.environ.get("ESPB_WORKERS", "1"))
    print(f"[EB {i}/{n}] {len(us)} units, {workers} workers x n_jobs {N_JOBS}; {versions()}", flush=True)
    stop = []
    if workers == 1:
        res = (_worker(u) for u in us)
        for nm, st, dt in res:
            print(f"  {nm}: {st} ({dt:.0f} s)", flush=True)
            if st.startswith("STOP"):
                stop.append(nm); break
    else:
        from concurrent.futures import as_completed
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
