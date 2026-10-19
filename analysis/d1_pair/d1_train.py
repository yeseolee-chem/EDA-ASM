#!/usr/bin/env python3
"""d1_train.py — D1PAIR Phase M (§3, §9): the pair-difference models g (arms B, C, B_ref) and the pooled model (arm P).

  python d1_train.py list <group>            # units of a group (stable order)
  python d1_train.py run <group> <i> <n>     # slice i of n of the group's units (idempotent; exit 3 = STOP)
  python d1_train.py nn <stage> <json>       # internal: one NN stage in a fresh process (tune | cv | fit)
  python d1_train.py collect                 # M_predictions.parquet, M_units.csv, tuning-cache audit (exit 3 = STOP)

Groups: T1 / LODO / LOSO = arms B, C x 6 models x 10 targets x outer folds; T2 = arm B x 6 x 10 x 5 (training without
the 11 V families); PB = arm P (KRR f_pool) and arm B_ref (KRR g on P_main only) x 10 targets x T1 folds.
D1PAIR_MODELS=<m+m> restricts a run to some models (job packing only).

Pairs (results/S_folds.csv + F_pairs.csv): training = training side T_main (P_main + A_main; B_ref: P_main; T2: minus
t2_exclude); test = held-out P_sens + A_main (primary metrics use P_main, anchors separately). Target y = dd_t (C) or
dd_t - [f0(x_a) - f0(x_b)] (B, B_ref; f0 = the f0 representative model, results/f0_d1.parquet).
Scale (§3-2) on the (inner) training pairs: z = [dx / s_delta, (xbar - mu) / s], no mean removed from dx; columns with
s < 1e-12 dropped; y / s_y without mean removal; recomputed in every inner fold.
g models (§3-3), all antisymmetric g(-dx, xbar) = -g(dx, xbar), g(0, xbar) = 0:
  Ridge  Ridge(alpha, fit_intercept=False) on z_delta only
  KRR    KernelRidge(alpha, kernel="precomputed"), K(p,q) = [k(dx_p, dx_q) - k(dx_p, -dx_q)] k(xbar_p, xbar_q), RBF k
  SVR    SVR(C, epsilon, kernel="precomputed") on the same kernel, trained both ways ((dx, y), (-dx, -y)),
         prediction 1/2 [s(dx) - s(-dx)]
  XGB    train_ml_single.GRIDS["XGB"] estimator settings, both ways, prediction 1/2 [h(dx) - h(-dx)]
  NN-2/4 shared sub-network h = r6_nn._layers (ELU + L2 + dropout, Espley structure), g = h([dx, xbar]) - h([-dx, xbar]);
         Hyperband (r6_nn.HP_STORAGE, r6_nn.HB, seed 23) on inner fold 1, CV MAE = the 4 inner folds refitted (500
         epochs, batch 32, shuffle off), final fit on all training pairs (seed 22)
Tuning: GroupKFold(4), group = core, on the training pairs; best = lowest mean inner MAE (kcal/mol; ties: grid order).
Gate M-1 per unit on 200 random test pairs: |g(dx) + g(-dx)| <= 1e-6 and |g(0)| <= 1e-6 kcal/mol, else exit 3.
Arm P: train_ml_single.GRIDS["KRR_rbf"] + _make_pipe on the D0 4,963 rows + the training-side D1 reactions (a and b of
the training T_main pairs); inner GroupKFold(4) over the D1 reactions by core, D0 rows in every inner training fold.
Cache: $D1PAIR_SCRATCH/M/units/<group>/<unit>.{json,parquet}; the json carries an input fingerprint (exit 3 on a mismatch).
Gate: results/PREREG_D1PAIR.md + prereg_d1pair.json committed and every input unchanged (d1_split.gate_prereg).
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

CODE_VERSION = "d1pair-train-1"
M_DIR = C.SCRATCH / "M"
UNITS = M_DIR / "units"
N_JOBS = int(os.environ.get("D1PAIR_NJOBS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
GROUPS = ("T1", "LODO", "LOSO", "T2", "PB")
GCOL = {"T1": "core", "T2": "core", "LODO": "dipole", "LOSO": "skeleton"}
_C = {}


# ================================================================ inputs
def data():
    """(S pairs table with dd columns, X182_d1 indexed by rxn_id, f0 representative predictions, sha dict)."""
    if "d" in _C:
        return _C["d"]
    S = pd.read_csv(C.RES / "S_folds.csv")
    F = pd.read_csv(C.RES / "F_pairs.csv", usecols=["pair_id"] + [f"dd_{t}" for t in C.TARGETS])
    S = S.merge(F, on="pair_id", how="left", validate="1:1")
    X = pd.read_parquet(C.X182_D1).set_index("rxn_id")
    f0rep = json.loads((C.RES / "f0_report.json").read_text())["representative"]
    f0 = pd.read_parquet(C.RES / "f0_d1.parquet")
    f0 = pd.concat([f0[(f0.target == t) & (f0.model == m)] for t, m in f0rep.items()])
    f0 = f0.pivot(index="rxn_id", columns="target", values="yhat")
    sha = {k: C.sha256(p) for k, p in (("S_folds", C.RES / "S_folds.csv"), ("F_pairs", C.RES / "F_pairs.csv"),
                                       ("X182_d1", C.X182_D1), ("f0_d1", C.RES / "f0_d1.parquet"))}
    _C["d"] = (S, X, f0, f0rep, sha)
    return _C["d"]


def unit_list(group):
    out = []
    if group in ("T1", "LODO", "LOSO", "T2"):
        nf = {"T1": 5, "T2": 5, "LODO": 5, "LOSO": 10}[group]
        arms = ("B",) if group == "T2" else C.ARMS_G
        for arm in arms:
            for m in C.MODELS:
                for t in C.TARGETS:
                    for k in range(nf):
                        out.append(dict(group=group, split=group, arm=arm, model=m, target=t, fold=k))
    elif group == "PB":
        for arm in ("P", "B_ref"):
            for t in C.TARGETS:
                for k in range(5):
                    out.append(dict(group=group, split="T1", arm=arm, model="KRR", target=t, fold=k))
    else:
        raise KeyError(group)
    only = [m for m in os.environ.get("D1PAIR_MODELS", "").split("+") if m]
    if only:
        out = [u for u in out if u["model"] in only]
    rank = {m: i for i, m in enumerate(["NN4", "NN2", "SVR", "KRR", "XGB", "Ridge"])}       # long ones first
    return sorted(out, key=lambda u: (rank[u["model"]], u["arm"], C.TARGETS.index(u["target"]), u["fold"]))


def uname(u):
    return f"{u['split']}__{u['arm']}__{u['model']}__{u['target']}__f{u['fold']}"


def masks(u):
    """(train index, test index) of the S rows for a unit, with the in-unit leak checks."""
    S = data()[0]
    col = f"fold_{u['split']}"
    te = S[(S[col] == u["fold"]) & (S.in_P_sens | S.in_A_main)].index
    trm = (S[col] != u["fold"]) & S.in_T_main
    if u["arm"] == "B_ref":
        trm &= S.in_P_main
    if u["split"] == "T2":
        trm &= ~S.t2_exclude
    tr = S[trm].index
    g = GCOL[u["split"]]
    if set(S.loc[te, g]) & set(S.loc[tr, g]):
        raise SystemExit(f"STOP: {uname(u)} test {g} also in training")
    if (set(S.loc[te, "a_job"]) | set(S.loc[te, "b_job"])) & (set(S.loc[tr, "a_job"]) | set(S.loc[tr, "b_job"])):
        raise SystemExit(f"STOP: {uname(u)} test reaction also in training")
    return np.asarray(tr), np.asarray(te)


def pair_x(idx):
    """(x_a, x_b) arrays (n, 182) of the S rows idx."""
    S, X = data()[0], data()[1]
    return X.loc[S.loc[idx, "a_rxn"]].to_numpy(float), X.loc[S.loc[idx, "b_rxn"]].to_numpy(float)


def dfo(idx, t):
    S, f0 = data()[0], data()[2]
    return f0.loc[S.loc[idx, "a_rxn"], t].to_numpy(float) - f0.loc[S.loc[idx, "b_rxn"], t].to_numpy(float)


def target_y(u, idx):
    y = data()[0].loc[idx, f"dd_{u['target']}"].to_numpy(float)
    return y - dfo(idx, u["target"]) if u["arm"] in ("B", "B_ref") else y


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


def _xgb_fit(zd, zm, ys, params):
    import train_ml_single as T
    from sklearn.base import clone
    est = clone(T.GRIDS["XGB"][0]).set_params(**params)
    Z = np.r_[np.c_[zd, zm], np.c_[-zd, zm]]
    return est.fit(Z, np.r_[ys, -ys])


def _xgb_g(m, zd, zm):
    return 0.5 * (m.predict(np.c_[zd, zm]) - m.predict(np.c_[-zd, zm]))


def _xgb_cell(zd, zm, ys, td, tm, yte, sy, params):
    m = _xgb_fit(zd, zm, ys, params)
    return float(np.mean(np.abs(_xgb_g(m, td, tm) * sy - yte)))


def grid_cells(model):
    if model == "Ridge":
        return [dict(alpha=a) for a in C.grid_values("Ridge")["alpha"]]
    if model == "KRR":
        return [dict(alpha=a, gamma_delta=gd, gamma_xbar=gm) for gd in C.GAMMAS for gm in C.GAMMAS
                for a in C.grid_values("KRR_rbf")["alpha"]]
    if model == "SVR":
        return [dict(C=c, epsilon=e, gamma_delta=gd, gamma_xbar=gm) for gd in C.GAMMAS for gm in C.GAMMAS
                for c in C.SVR_C for e in C.SVR_EPS]
    if model == "XGB":
        g = C.grid_values("XGB")
        import itertools
        ks = sorted(g)
        return [dict(zip(ks, v)) for v in itertools.product(*[g[k] for k in ks])]
    raise KeyError(model)


def tune_grid(u, tr):
    """inner GroupKFold(4) MAE of every grid cell -> (cells, mean MAE per cell, per-fold MAEs)."""
    from joblib import Parallel, delayed
    cells = grid_cells(u["model"])
    per_fold = []
    for itr, ite in inner_folds(tr):
        sc, (zd, zm, ys, _), (td, tm) = prep(itr, ite, u)
        yte = target_y(u, ite)
        if u["model"] == "Ridge":
            from sklearn.linear_model import Ridge
            per_fold.append([float(np.mean(np.abs(Ridge(alpha=c["alpha"], fit_intercept=False).fit(zd, ys).predict(td) * sc.sy
                                                  - yte))) for c in cells])
        elif u["model"] in ("KRR", "SVR"):
            Dtr, Dte = dist3(zd, zm, zd, zm), dist3(td, tm, zd, zm)
            gg = [(gd, gm) for gd in C.GAMMAS for gm in C.GAMMAS]
            if u["model"] == "KRR":
                al = C.grid_values("KRR_rbf")["alpha"]
                res = Parallel(n_jobs=N_JOBS, max_nbytes="1M")(delayed(_krr_cell)(Dtr, Dte, ys, yte, sc.sy, gd, gm, al)
                                                               for gd, gm in gg)
            else:
                res = Parallel(n_jobs=N_JOBS, max_nbytes="1M")(delayed(_svr_cell)(Dtr, Dte, ys, yte, sc.sy, gd, gm, C.SVR_C,
                                                                                  C.SVR_EPS) for gd, gm in gg)
            per_fold.append([v for r in res for v in r])               # order = grid_cells order
        else:
            res = Parallel(n_jobs=N_JOBS)(delayed(_xgb_cell)(zd, zm, ys, td, tm, yte, sc.sy, c) for c in cells)
            per_fold.append(res)
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
        m = _xgb_fit(zd, zm, ys, best)
        g = lambda a, b: _xgb_g(m, a, b)                              # noqa: E731
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
    NN.set_seeds(C.CV_SEED)
    if stage == "cv":
        errs = []
        for itr, ite in folds:
            sc, (zd, zm, ys, _), (td, tm) = prep(itr, ite, u)
            yte = target_y(u, ite)
            m = nn_model(zd.shape[1], zm.shape[1], size, hps.__getitem__, tname)
            m.fit(x=np.c_[zd, zm], y=ys, validation_data=(np.c_[td, tm], yte / sc.sy), **NN.FIT)
            errs.append(float(np.mean(np.abs(m.predict(np.c_[td, tm], verbose=0).ravel() * sc.sy - yte))))
        return dict(cv_fold_mae=errs, val_mae=float(np.mean(errs)))
    if stage == "fit":
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
    r = subprocess.run([sys.executable, str(HERE / "d1_train.py"), "nn", stage, str(inp)], env=env, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nn {stage} failed: {r.stderr[-3000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    C.write_json_safe(out, dict(payload_sha=psha, result=res, seconds=time.time() - t0))
    return res


# ================================================================ arm P (pooled single-reaction KRR)
def run_pool_arm(u, tr, te):
    import train_ml_single as T
    from sklearn.model_selection import GridSearchCV, GroupKFold
    S, X = data()[0], data()[1]
    import d1_f0 as F0
    X0, Y0, rows0, _ = F0.inputs()
    lab = {int(d["rxn_id"]): d for d in C.load_d1_labels()}
    rx = sorted(set(S.loc[tr, "a_rxn"]) | set(S.loc[tr, "b_rxn"]))
    core = {}
    for i in tr:
        core[int(S.loc[i, "a_rxn"])] = S.loc[i, "core"]; core[int(S.loc[i, "b_rxn"])] = S.loc[i, "core"]
    Xp = np.r_[X0.to_numpy(float), X.loc[rx].to_numpy(float)]
    yp = np.r_[Y0[C.D0_TCOL[u["target"]]].to_numpy(float), np.array([lab[r][C.D1_TCOL[u["target"]]] for r in rx], float)]
    n0 = len(rows0)
    d1i = np.arange(n0, n0 + len(rx))
    cv = [(np.r_[np.arange(n0), d1i[a]], d1i[b]) for a, b in
          GroupKFold(C.N_INNER).split(d1i, groups=[core[r] for r in rx])]
    est, grid = T.GRIDS["KRR_rbf"]
    gs = GridSearchCV(T._make_pipe(est), grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise")
    gs.fit(Xp, yp)
    f = lambda idx: gs.predict(X.loc[S.loc[idx, "a_rxn"]].to_numpy(float)) - gs.predict(X.loc[S.loc[idx, "b_rxn"]].to_numpy(float))  # noqa: E731
    best = {k: float(v) for k, v in gs.best_params_.items()}
    return f(te), f(tr), dict(best_params=best, inner_cv_mae=float(-gs.best_score_), n_train_rows=len(yp), n_d0=n0,
                              n_d1_reactions=len(rx), edge_hits={k: bool(T._edge_hit(k, v, grid[k])) for k, v in gs.best_params_.items()})


# ================================================================ one unit
def versions():
    from importlib.metadata import version
    return {k: version(k) for k in ("scikit-learn", "xgboost", "numpy", "pandas", "tensorflow", "keras", "keras_tuner")}


def fingerprint(u, tr, te):
    S, X, f0, f0rep, sha = data()
    grid = "nn" if u["model"].startswith("NN") else repr(grid_cells(u["model"]))
    return hashlib.sha256(json.dumps(dict(u={k: v for k, v in u.items() if k != "group"}, sha=sha, cols=list(X.columns),
                                          f0rep=f0rep, grid=grid, tr=C.list_sha([(i,) for i in S.loc[tr, "pair_id"]]),
                                          te=C.list_sha([(i,) for i in S.loc[te, "pair_id"]]), versions=versions(),
                                          code=CODE_VERSION), sort_keys=True).encode()).hexdigest()


def cached(u, fp):
    f = UNITS / u["group"] / f"{uname(u)}.json"
    if not (f.is_file() and f.with_suffix(".parquet").is_file()):
        return False
    rec = json.loads(f.read_text())
    if rec.get("fingerprint") != fp:
        print(f"STOP: cached unit {f.name} was made from other inputs; move {f.parent} away", file=sys.stderr)
        sys.exit(3)
    if rec.get("gate_M1") is False:                       # a recorded M-1 failure stays a STOP on every rerun
        raise SystemExit(f"STOP: gate M-1 {f.stem} (cached failure): {rec.get('antisym')}")
    return True


def run_unit(u):
    tr, te = masks(u)
    fp = fingerprint(u, tr, te)
    if cached(u, fp):
        return "cached"
    t0 = time.time()
    S = data()[0]
    y_te = S.loc[te, f"dd_{u['target']}"].to_numpy(float)
    y_tr = S.loc[tr, f"dd_{u['target']}"].to_numpy(float)
    info = dict(n_train=len(tr), n_test=len(te), n_test_P_main=int(S.loc[te, "in_P_main"].sum()),
                n_train_anchor=int((S.loc[tr, "ref_kind"] == "anchor").sum()))
    if u["arm"] == "P":
        pte, ptr, extra = run_pool_arm(u, tr, te)
        info.update(extra)
    else:
        base_te = dfo(te, u["target"]) if u["arm"] in ("B", "B_ref") else np.zeros(len(te))
        base_tr = dfo(tr, u["target"]) if u["arm"] in ("B", "B_ref") else np.zeros(len(tr))
        if u["model"].startswith("NN"):
            wd = UNITS / u["group"] / "nn_work" / uname(u)
            base = dict(u=u, fp=fp)
            hps = nn_call("tune", base, wd)["hps"]
            cv = nn_call("cv", dict(base, hps=hps), wd)
            fit = nn_call("fit", dict(base, hps=hps), wd)
            gte, gtr, anti, scl = np.asarray(fit["g_test"]), np.asarray(fit["g_train"]), fit["antisym"], fit["scale"]
            info.update(best_params=hps, inner_cv_mae=cv["val_mae"], inner_fold_mae=cv["cv_fold_mae"])
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
                        grid_mae=[round(float(v), 6) for v in mean])
        info.update(antisym=anti, scale=scl)
        if not (anti["max_sum"] <= C.ANTISYM_TOL and anti["max_zero"] <= C.ANTISYM_TOL):
            info["gate_M1"] = False
        pte, ptr = base_te + gte, base_tr + gtr
    pred = pd.DataFrame(dict(pair_id=S.loc[te, "pair_id"].to_numpy(), ref_kind=S.loc[te, "ref_kind"].to_numpy(),
                             fold=u["fold"], split=u["split"], arm=u["arm"], model=u["model"], target=u["target"],
                             y=y_te, yhat=pte))
    rec = dict(u, fingerprint=fp, **info, train_mae=float(np.mean(np.abs(ptr - y_tr))),
               test_mae_P_main=float(np.mean(np.abs(pte - y_te)[S.loc[te, "in_P_main"].to_numpy()])),
               train_pair_sha256=C.list_sha([(i,) for i in S.loc[tr, "pair_id"]]), seconds=time.time() - t0,
               host=os.uname().nodename, versions=versions())
    rec.setdefault("gate_M1", True)
    d = UNITS / u["group"]
    C.write_atomic(d / f"{uname(u)}.parquet", lambda t: pred.to_parquet(t, index=False))
    C.write_json_safe(d / f"{uname(u)}.json", rec)
    if not rec["gate_M1"]:
        raise SystemExit(f"STOP: gate M-1 {uname(u)}: {rec.get('antisym')}")
    return f"done {rec['test_mae_P_main']:.3f}"


def _worker(u):
    t0 = time.time()
    try:
        return uname(u), run_unit(u), time.time() - t0
    except SystemExit as e:
        return uname(u), f"STOP {e.code}", time.time() - t0


def cmd_run(group, i, n):
    import d1_split as SP
    SP.gate_prereg()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    us = unit_list(group)[i::n]
    workers = int(os.environ.get("D1PAIR_WORKERS", "1"))
    print(f"[M {group} {i}/{n}] {len(us)} units, {workers} workers x n_jobs {N_JOBS}; {versions()}", flush=True)
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


def cmd_collect():
    S = data()[0]
    recs, parts, missing = [], [], []
    for g in GROUPS:
        for u in unit_list(g):
            f = UNITS / g / f"{uname(u)}.json"
            if not f.is_file():
                missing.append(uname(u)); continue
            rec = json.loads(f.read_text())
            recs.append({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in rec.items() if k != "grid_mae"})
            parts.append(pd.read_parquet(f.with_suffix(".parquet")))
    U = pd.DataFrame(recs)
    Pm = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # tuning-cache audit (§9): no test pair_id inside any unit's training pairs
    leak = []
    for r in recs:
        u = {k: r[k] for k in ("group", "split", "arm", "model", "target", "fold")}
        tr, te = masks(u)
        if C.list_sha([(i,) for i in S.loc[tr, "pair_id"]]) != r["train_pair_sha256"]:
            leak.append(f"{uname(u)}: training pairs differ from the recorded ones")
        if set(S.loc[tr, "pair_id"]) & set(S.loc[te, "pair_id"]):
            leak.append(f"{uname(u)}: test pair in training")
    # NN stages recompute their masks from the unit spec (masks()), so their tuning data are the audited training pairs
    C.write_atomic(C.RES / "M_units.csv", lambda t: U.to_csv(t, index=False))
    C.write_atomic(C.RES / "M_predictions.parquet", lambda t: Pm.to_parquet(t, index=False))
    rep = dict(n_units=len(U), missing=missing[:50], n_missing=len(missing), m1_fail=U.loc[~U.gate_M1.astype(bool), "split"].count()
               if len(U) else 0, leak=leak[:20], n_leak=len(leak), predictions=dict(n=len(Pm), sha256=C.sha256(C.RES / "M_predictions.parquet")),
               units=dict(n=len(U), sha256=C.sha256(C.RES / "M_units.csv")),
               antisym_max=float(max((json.loads(a)["max_sum"] for a in U.antisym.dropna()), default=0.0)) if len(U) else None,
               antisym_zero_max=float(max((json.loads(a)["max_zero"] for a in U.antisym.dropna()), default=0.0)) if len(U) else None)
    rep["stop"] = bool(rep["n_leak"] or rep["m1_fail"])
    C.write_json_safe(C.RES / "M_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else (2 if missing else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "nn":
        print(json.dumps(nn_stage(a[1], json.loads(Path(a[2]).read_text()))))
        sys.exit(0)
    if cmd == "list":
        for u in unit_list(a[1]):
            print(uname(u))
        sys.exit(0)
    if cmd == "run":
        sys.exit(cmd_run(a[1], int(a[2]), int(a[3])))
    if cmd == "collect":
        sys.exit(cmd_collect())
    raise SystemExit(__doc__)
