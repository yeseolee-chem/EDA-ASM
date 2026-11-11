#!/usr/bin/env python3
r"""xgb_abs.py — ABS3 XGB absolute-value units (docs/specs/ABS3_MODELS.md 0-3, 0-4, 3, 5, 7). CPU, env_espley
(xgboost 3.2.0, scikit-learn 1.3.1) through jobs/cpu_abs.slurm ABS3ENV=espley, --cpus-per-task=32
(LAD_XGB_NJOBS = joblib workers; results do not depend on it). Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python xgb_abs.py list                    unit names (A__lf3__S__XGB__f<0..4>)
  python xgb_abs.py smoke                   C-8 (XGB): fold 0, grid cells first + last -> $ABS3/smoke/xgb (printed)
  python xgb_abs.py overfit                 C-O (XGB): 64 fold-0 training reactions, last (largest) cell (printed)
  python xgb_abs.py run <i|@I> <n>          units [i::n] -> $ABS3/units/A__lf3__S__XGB__f<k>.{json,parquet}
                                            Gates: PREREG (prereg_abs.check), results/C_tests.json all_pass on this code
  python xgb_abs.py collect                 XGB units + the TF · GNN part ($ABS3/collect/nn.parquet, train_abs.py collect)
                                            -> results/KA_predictions.parquet (reactions), KA_pairs.parquet (test-pair ΔΔ
                                            = yhat(a) - yhat(b), C-P), KA_val.csv, KA_report.json

Rules (spec 0-4): estimator clone(train_ml_single.GRIDS["XGB"][0]) with the 12 cells of d1_train.grid_cells("XGB")
(= the pair XGB, imported through analysis/d1_ladder/train_xgb.py); per target separately: inner 4 folds = the pair XGB
rule (the unfiltered outer-train pair list of the split file -> GroupKFold(4) by core -> usable pairs -> their
reactions; fold 0 = the split-file inner lists = the TF · GNN inner reactions), cell = lowest mean inner MAE (kcal),
ties grid order; final fit on all outer training reactions. X = the 182 columns z-scored with the fit's training
reactions, y standardised (abs_xgbcore). One fit (random_state 42): yhat_seed_sd = NaN.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402
import abs_pairs as AP  # noqa: E402
import abs_xgbcore as XC  # noqa: E402

sys.path.insert(0, str(C.REPO / "analysis" / "d1_ladder"))
import train_xgb as TX  # noqa: E402               (paths of d1_train / train_ml_single, versions, N_JOBS, cost)

D1T, TMS = TX.D1T, TX.TMS
CODE_VERSION = "abs3-xgb-1"
MODEL, ARM = "XGB", C.ARM
T3 = list(C.TARGETS3)
NT = len(T3)
NF = C.NF
GRID = D1T.grid_cells("XGB")                      # 12 cells (sorted-key product, = LADDER / CH3M)
ESTIMATOR = repr(TMS.GRIDS["XGB"][0])
N_JOBS = TX.N_JOBS
WORKER_PATHS = list(TX.WORKER_PATHS) + [str(HERE)]
CODE_FILES = C.XGB_TRAINER_FILES
EXT_FILES = ("d1_ladder/train_xgb.py", "d1_pair/d1_train.py", "d1_pair/d1pair_common.py",
             "espley_xtb_repro/train_ml_single.py")
CODE_SHA = {**{f: C.sha256(HERE / f) for f in CODE_FILES},
            **{f: C.sha256(C.REPO / "analysis" / f) for f in EXT_FILES}}
TRAINER_FILES = CODE_FILES
OVERFIT_N, OVERFIT_FRAC = 64, 0.10
_GATE: dict = {}

if abs(XC.SCALE_EPS - TX.C.SCALE_EPS) > 0:
    C.stop(f"abs_xgbcore SCALE_EPS {XC.SCALE_EPS} != d1pair_common SCALE_EPS {TX.C.SCALE_EPS}")


def uname(u):
    return C.uname(MODEL, u["fold"])


def unit_list():
    return [dict(rung=C.RUNG, arm=ARM, model=MODEL, fold=k) for k in range(NF)]


# ================================================================ data (pack)
class PackAbs:
    """the abs_pack.py pack (reactions, X, y4, fold sets, pairs, split-file lists)."""

    def __init__(self):
        if not (C.PACK_NPZ.is_file() and C.PACK_JSON.is_file()):
            C.not_ready(f"{C.PACK_NPZ} / {C.PACK_JSON} missing (abs_pack.py)")
        info = C.read_json(C.PACK_JSON)
        if not info.get("ok") or C.sha256(C.PACK_NPZ) != info.get("npz_sha256"):
            C.stop(f"{C.PACK_NPZ}: pack not ok or changed after its json")
        z = np.load(C.PACK_NPZ, allow_pickle=False)
        self.z = z
        self.info = info
        self.rids = z["rids"].astype(np.int64)
        self.row = {int(r): k for k, r in enumerate(self.rids)}
        self.X = z["X"].astype(float)
        self.y4 = z["y4"].astype(float)
        self.attr = {c: z[c].astype(str) for c in ("dipole", "skeleton", "series", "attach", "core", "role")}
        self.pid = z["pair_id"].astype(str)
        self.pid_row = {q: i for i, q in enumerate(self.pid)}
        self.ra, self.rb = z["ra"].astype(int), z["rb"].astype(int)
        self.usable_pair = z["usable_pair"].astype(bool)
        self.pair_core = z["pair_core"].astype(str)
        self.feat_cols = list(info["glob_cols"])
        self.sha = dict(pack_npz=info["npz_sha256"], pack_json=C.sha256(C.PACK_JSON), **info["sources_sha256"])
        self._splits = {}

    def rxn_ids(self, rows):
        return self.rids[np.asarray(rows, dtype=int)]

    def _rx(self, prow):
        return np.unique(np.r_[self.ra[prow], self.rb[prow]]).astype(int)

    def split(self, k):
        """reaction sets of fold k + the 4 XGB inner folds (pair XGB rule, then reactions)."""
        k = int(k)
        if k in self._splits:
            return self._splits[k]
        z = self.z
        out = {key: z[f"f{k}_{key}"].astype(int) for key in C.FOLD_KEYS}
        from sklearn.model_selection import GroupKFold
        tr_all = np.array([self.pid_row[q] for q in z[f"p{k}_train"].astype(str)], dtype=int)
        inner = []
        for j, (ia, ib) in enumerate(GroupKFold(TX.C.N_INNER).split(tr_all, groups=self.pair_core[tr_all])):
            itr_all, iva_all = tr_all[ia], tr_all[ib]
            if j == 0 and (self.pid[itr_all].tolist() != z[f"p{k}_inner_train"].astype(str).tolist()
                           or self.pid[iva_all].tolist() != z[f"p{k}_inner_val"].astype(str).tolist()):
                C.stop(f"fold {k}: GroupKFold(4) fold 0 differs from the split file inner_train / inner_val")
            if set(self.pair_core[itr_all].tolist()) & set(self.pair_core[iva_all].tolist()):
                C.stop(f"fold {k} inner {j}: a core on both sides")
            itr_p, iva_p = itr_all[self.usable_pair[itr_all]], iva_all[self.usable_pair[iva_all]]
            itr, iva = self._rx(itr_p), self._rx(iva_p)
            if not len(itr) or not len(iva) or set(itr.tolist()) & set(iva.tolist()):
                C.stop(f"fold {k} inner {j}: empty or overlapping inner reaction sets (spec LS-3 rule)")
            if not set(itr.tolist()) | set(iva.tolist()) <= set(out["train"].tolist()):
                C.stop(f"fold {k} inner {j}: inner reactions outside the outer training reactions")
            inner.append(dict(train=itr, val=iva, n_pairs=dict(train=int(len(itr_p)), val=int(len(iva_p)))))
        if not (np.array_equal(inner[0]["train"], out["inner_train"]) and np.array_equal(inner[0]["val"], out["inner_val"])):
            C.stop(f"fold {k}: XGB inner fold 0 reactions differ from the pack inner_train / inner_val (TF · GNN)")
        if set(out["train"].tolist()) & set(out["test"].tolist()):
            C.stop(f"fold {k}: test reaction in train")

        def ls(rows):
            return TX.C.list_sha([(str(int(r)),) for r in self.rids[rows]])
        out.update(inner=inner, rxn_sha=TX.C.list_sha([(ls(out[key]),) for key in C.FOLD_KEYS]),
                   test_sha=ls(out["test"]), val_sha=ls(out["inner_val"]),
                   inner_sha=TX.C.list_sha([(ls(f["train"]), ls(f["val"])) for f in inner]))
        self._splits[k] = out
        return out


# ================================================================ fits
def prep(P, rows, evals):
    """one fit's scaling (training reactions only) and z of every eval set."""
    rows = np.unique(np.asarray(rows, dtype=int))
    mu, s, keep = XC.scale(P.X, rows)
    y = P.y4[rows]
    if not (np.isfinite(P.X[rows]).all() and np.isfinite(y).all()):
        C.stop("non-finite X / y4 among the training reactions")
    ymu, ysd = y.mean(axis=0), y.std(axis=0)
    if not (ysd > 0).all():
        C.stop(f"y std not positive: {ysd}")
    ztr = XC.z(P.X, rows, mu, s, keep)
    ev = {name: XC.z(P.X, er, mu, s, keep) for name, er in evals.items()}
    return dict(rows=rows, ztr=ztr, ys=[(y[:, t] - ymu[t]) / ysd[t] for t in range(NT)], ymu=ymu, ysd=ysd, evals=ev,
                eval_rows={k: np.asarray(v, dtype=int) for k, v in evals.items()}, mu=mu, s=s, keep=keep,
                scale=dict(n_dropped=int((~keep).sum()), n_rows=int(len(rows)), x_stats_sha256=XC.stats_sha(mu, s, keep)))


def leak_check(pf, others, test_rows):
    """C-5: scaling reactions = the fit's training reactions; validation / test reactions not among them."""
    used = set(pf["rows"].tolist())
    for o in others:
        if used & set(np.asarray(o).tolist()):
            C.stop("C-5: a validation / test reaction is among the fit's training / scaling reactions")
    if used & set(np.asarray(test_rows).tolist()):
        C.stop("C-5: a test reaction is among the fit's training reactions")
    return dict(n_fit_rxn=len(used), n_test_rxn_overlap=0, ok=True)


def run_parallel(calls):
    """calls = [(ztr, ys, ymu, ysd, params, evals)] -> [(kcal dict, seconds)] in order."""
    if N_JOBS <= 1:
        return [XC.task(WORKER_PATHS, *c) for c in calls]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=N_JOBS, max_nbytes="200K")(delayed(XC.task)(WORKER_PATHS, *c) for c in calls)


# ================================================================ unit
def _check_root(root, smoke):
    r = C.assert_write_ok(Path(root) / "probe").parent
    p = Path(C.UNITS).resolve()
    if r == p or p in r.parents:
        if smoke:
            C.stop(f"smoke / test runs never write {p}")
        if not _GATE.get("prereg"):
            C.stop(f"writing {p} needs the PREREG gate (xgb_abs.py run)")
    return r


def fingerprint(P, u, sp, idx, smoke=False):
    return C.norm(dict(code=CODE_VERSION, code_sha256=CODE_SHA, rung=u["rung"], arm=ARM, model=MODEL, targets=T3,
                       conversion=C.M_ROWS, inputs_sha256=P.sha, split_rxn=sp["rxn_sha"], inner_folds=sp["inner_sha"],
                       prereg=None if smoke else _GATE.get("prereg"), grid=[GRID[i] for i in idx], estimator=ESTIMATOR,
                       feat_cols=TX.C.list_sha([(c,) for c in P.feat_cols]), smoke=bool(smoke),
                       rules=dict(inner="pair XGB rule: GroupKFold(4) by core on the unfiltered outer-train pair list, "
                                        "usable filter, reactions of the pairs; fold 0 = split-file inner lists",
                                  select="per target lowest mean inner MAE (kcal), ties grid order; final fit on all "
                                         "training reactions",
                                  scale="X z-score and y standardisation on the fit's training reactions; X columns with "
                                        f"std < {XC.SCALE_EPS:g} dropped", seeds="one fit, GRIDS estimator random_state 42"),
                       versions=TX.versions()))


def run_unit(P, u, root, cells=None, smoke=False):
    root = _check_root(root, smoke)
    root.mkdir(parents=True, exist_ok=True)
    if u.get("model") != MODEL or u.get("arm") != ARM:
        C.stop(f"unit {u} is not an XGB unit")
    name = uname(u)
    idx = list(range(len(GRID))) if cells is None else sorted({int(c) for c in cells})
    sp = P.split(u["fold"])
    fp = fingerprint(P, u, sp, idx, smoke)
    js = root / f"{name}.json"
    if js.is_file():
        old = C.read_json(js)
        ofp = old.get("fingerprint") or {}
        if ofp != fp:
            C.stop(f"cache {js} has a different fingerprint: keys {sorted(k for k in set(fp) | set(ofp) if ofp.get(k) != fp.get(k))}")
        print(f"  {name}: cached", flush=True)
        return old
    t_unit = time.time()
    tr, te, inner = sp["train"], sp["test"], sp["inner"]
    if set(te.tolist()) & (set(tr.tolist()) | {r for f in inner for key in ("train", "val") for r in f[key].tolist()}):
        C.stop(f"C-5: {name}: a test reaction is in the training or an inner set")
    fits = []
    for f in inner:
        pf = prep(P, f["train"], dict(val=f["val"]))
        pf["leak"] = leak_check(pf, (f["val"], te), te)
        fits.append(pf)
    jobs = sorted([(j, ti, ci) for ti in range(NT) for j in range(len(fits)) for ci in idx], key=lambda x: -TX.cost(GRID[x[2]]))
    print(f"  {name}: {len(jobs)} tuning fits on {N_JOBS} workers ({len(P.feat_cols)} features)", flush=True)
    res = run_parallel([(fits[j]["ztr"], fits[j]["ys"][ti], fits[j]["ymu"][ti], fits[j]["ysd"][ti], GRID[ci],
                         dict(val=fits[j]["evals"]["val"])) for j, ti, ci in jobs])
    mae = np.full((NT, len(fits), len(idx)), np.nan)
    tune_s = []
    for (j, ti, ci), (kc, sec) in zip(jobs, res):
        yv = P.y4[inner[j]["val"], ti]
        mae[ti, j, idx.index(ci)] = float(np.mean(np.abs(kc["val"] - yv)))
        tune_s.append(sec)
    if not np.isfinite(mae).all():
        C.stop(f"{name}: non-finite inner MAE")
    tune, best = {}, {}
    for ti, t in enumerate(T3):
        means = mae[ti].mean(axis=0)
        b = int(np.argmin(means))
        best[t] = idx[b]
        tune[t] = dict(grid_index=idx, grid_mae=means.tolist(), fold_mae=mae[ti].tolist(),
                       best=dict(cell=GRID[idx[b]], grid_index=idx[b]), inner_cv_mae=float(means[b]),
                       inner_fold_mae=mae[ti][:, b].tolist(), val_kcal_mae=float(mae[ti, 0, b]),
                       ysd_inner=[float(f["ysd"][ti]) for f in fits])
    fin = prep(P, tr, dict(test=te, train=tr))
    fin["leak"] = leak_check(fin, (te,), te)
    fres = run_parallel([(fin["ztr"], fin["ys"][ti], fin["ymu"][ti], fin["ysd"][ti], GRID[best[t]],
                          dict(test=fin["evals"]["test"], train=fin["evals"]["train"])) for ti, t in enumerate(T3)])
    yhat = np.zeros((len(te), NT))
    final = {}
    for ti, (t, (kc, sec)) in enumerate(zip(T3, fres)):
        yhat[:, ti] = kc["test"]
        final[t] = dict(cell=GRID[best[t]], grid_index=best[t], ymu=float(fin["ymu"][ti]), ysd=float(fin["ysd"][ti]),
                        train_mae=float(np.mean(np.abs(kc["train"] - P.y4[tr, ti]))), seconds=float(sec))
    if not np.isfinite(yhat).all():
        C.stop(f"{name}: non-finite test predictions")
    rid = P.rxn_ids(te)
    pred = pd.concat([pd.DataFrame(dict(rxn_id=rid, fold=int(u["fold"]), model=MODEL, target=t, y=P.y4[te, j],
                                        yhat=yhat[:, j], yhat_seed_sd=np.nan)) for j, t in enumerate(T3)], ignore_index=True)
    pq = root / f"{name}.parquet"
    C.write_parquet(pq, pred)
    rec = dict(unit=u, name=name, model=MODEL, targets=T3, fingerprint=fp, smoke=bool(smoke), n_seeds=1,
               chosen={t: dict(cell=GRID[best[t]], grid_index=best[t]) for t in T3}, tune=tune, final=final,
               val_kcal_mae={t: tune[t]["val_kcal_mae"] for t in T3}, val_rxn_sha256=sp["val_sha"],
               test_rxn_sha256=sp["test_sha"], n_train=int(len(tr)), n_test=int(len(te)),
               n_inner_train=int(len(sp["inner_train"])), n_inner_val=int(len(sp["inner_val"])),
               inner_folds=[dict(n_train=int(len(f["train"])), n_val=int(len(f["val"])), n_pairs=f["n_pairs"],
                                 scale=pf["scale"], leak=pf["leak"]) for f, pf in zip(inner, fits)],
               final_scale=fin["scale"], final_leak=fin["leak"], n_features=len(P.feat_cols), parquet_sha256=C.sha256(pq),
               parquet_columns=list(pred.columns), fit_seconds=dict(tune=tune_s, final=[final[t]["seconds"] for t in T3]),
               cpu_seconds=float(sum(tune_s) + sum(final[t]["seconds"] for t in T3)), n_jobs=N_JOBS,
               wall_seconds=time.time() - t_unit, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    C.write_json(js, rec)
    print(f"  {name}: done ({rec['wall_seconds']:.0f} s wall, {rec['cpu_seconds'] / 3600:.2f} CPU-h) val "
          f"{json.dumps({t: round(v, 4) for t, v in rec['val_kcal_mae'].items()})}", flush=True)
    return rec


# ================================================================ C helpers (tests_abs.py xgb)
def overfit(P):
    tr0 = P.split(0)["train"]
    rng = np.random.default_rng(C.SEED)
    rows = np.sort(rng.choice(tr0, size=OVERFIT_N, replace=False))
    cell = GRID[-1]
    pf = prep(P, rows, dict(train=rows))
    lk = leak_check(pf, (), np.zeros(0, dtype=int))
    res = [XC.task(WORKER_PATHS, pf["ztr"], pf["ys"][t], pf["ymu"][t], pf["ysd"][t], cell, dict(train=pf["evals"]["train"]))
           for t in range(NT)]
    yh = np.stack([r[0]["train"] for r in res], 1)
    y = P.y4[rows]
    mae, sd64 = np.abs(yh - y).mean(0), y.std(0)
    return dict(model=MODEL, cell=cell, n=OVERFIT_N, seed=C.SEED, rxn_ids=P.rxn_ids(rows).tolist(),
                train_mae=dict(zip(T3, mae.tolist())), std_64=dict(zip(T3, sd64.tolist())),
                ratio=dict(zip(T3, (mae / sd64).tolist())),
                threshold=f"train MAE <= {OVERFIT_FRAC:g} x std (ddof 0) of the 64 reactions, every target", leak=lk,
                ok=bool((mae <= OVERFIT_FRAC * sd64).all()))


def determinism(P):
    """C-X: fold 0, target barrier, grid cell 0, fit twice -> test predictions identical."""
    sp = P.split(0)
    pf = prep(P, sp["train"], dict(test=sp["test"]))
    ti = T3.index("barrier")
    a = XC.task(WORKER_PATHS, pf["ztr"], pf["ys"][ti], pf["ymu"][ti], pf["ysd"][ti], GRID[0], dict(test=pf["evals"]["test"]))
    b = XC.task(WORKER_PATHS, pf["ztr"], pf["ys"][ti], pf["ymu"][ti], pf["ysd"][ti], GRID[0], dict(test=pf["evals"]["test"]))
    d = float(np.abs(a[0]["test"] - b[0]["test"]).max())
    return dict(fold=0, target="barrier", cell=GRID[0], n_test=int(len(sp["test"])), max_abs_diff=d, ok=d == 0.0)


def ct_checks(P, rec, pq):
    """C-T (XGB): 4 targets in the record and the predictions, training-reaction scaling recomputed independently."""
    pred = pd.read_parquet(pq)
    sp = P.split(rec["unit"]["fold"])
    tr = sp["train"]
    x = P.X[tr]
    s = x.std(0)
    y = P.y4[tr]
    fin = rec["final"]
    mu_r, s_r = x.mean(0), x.std(0)                      # independent recomputation from the training reactions
    sha_tr = XC.stats_sha(mu_r, s_r, s_r >= XC.SCALE_EPS)
    alt = np.r_[tr, sp["test"][:1]]
    xa = P.X[alt]
    sha_alt = XC.stats_sha(xa.mean(0), xa.std(0), xa.std(0) >= XC.SCALE_EPS)
    checks = dict(final_x_stats_from_train=rec["final_scale"].get("x_stats_sha256") == sha_tr,
                  x_stats_check_sensitive=sha_alt != sha_tr,
                  record_targets_4=list(rec["targets"]) == T3 and sorted(rec["chosen"]) == sorted(T3),
                  prediction_targets_4=sorted(pred.target.unique()) == sorted(T3),
                  prediction_columns=list(pred.columns) == ["rxn_id", "fold", "model", "target", "y", "yhat", "yhat_seed_sd"],
                  final_scale_rows_eq_train=rec["final_scale"]["n_rows"] == len(tr),
                  final_dropped_cols_recomputed=rec["final_scale"]["n_dropped"] == int((s < XC.SCALE_EPS).sum()),
                  final_ymu_ysd_from_train=all(abs(fin[t]["ymu"] - float(y[:, j].mean())) <= 1e-12
                                               and abs(fin[t]["ysd"] - float(y[:, j].std())) <= 1e-12
                                               for j, t in enumerate(T3)),
                  y_equals_pack=bool(np.allclose(pred.pivot(index="rxn_id", columns="target", values="y")[T3].to_numpy(),
                                                 P.y4[[P.row[int(r)] for r in sorted(pred.rxn_id.unique())]], atol=0, rtol=0)))
    return dict(ok=bool(all(checks.values())), checks=checks)


def smoke(P):
    root = C.SMOKE / "xgb"
    _check_root(root, True)
    if root.is_dir():
        for p in root.glob("A__*"):
            C.assert_write_ok(p)
            p.unlink()
    u = dict(rung=C.RUNG, arm=ARM, model=MODEL, fold=0)
    t0 = time.time()
    rec = run_unit(P, u, root, cells=[0, len(GRID) - 1], smoke=True)
    return rec, root / f"{uname(u)}.parquet", time.time() - t0


# ================================================================ gates
def gate_prereg():
    import prereg_abs as PR
    rec = PR.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    p = C.RES / "C_tests.json"
    c = C.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        C.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K)")
    rec = (c.get("trainer_code_sha256") or {}).get("xgb_abs") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA[f]]
    if diff:
        C.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}")
    return dict(c_tests_sha256=C.sha256(p))


# ================================================================ commands
def cmd_run(i, n):
    i, n = C.array_task(i), int(n)
    us = unit_list()[i::n]
    if not us:
        return 0
    pre = gate_prereg()
    gate_c_tests()
    P = PackAbs()
    print(f"[K] PREREG ok (git {pre.get('git_head')}), C_tests ok, {TX.versions()}, N_JOBS {N_JOBS}, XGB slice {i}/{n}: "
          f"{len(us)} units", flush=True)
    for u in us:
        run_unit(P, u, C.UNITS)
    return 0


def load_units():
    names = [uname(u) for u in unit_list()]
    missing = [x for x in names if not ((C.UNITS / f"{x}.json").is_file() and (C.UNITS / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: C.read_json(C.UNITS / f"{x}.json") for x in names}
    for x, r in recs.items():
        if C.sha256(C.UNITS / f"{x}.parquet") != r.get("parquet_sha256") or r.get("smoke"):
            C.stop(f"{x}: parquet changed or smoke unit")
        if not (all(f["leak"]["ok"] for f in r["inner_folds"]) and r["final_leak"]["ok"]):
            C.stop(f"{x}: C-5 leak check failed")
        if list(r.get("targets") or []) != T3:
            C.stop(f"{x}: targets {r.get('targets')}")
    return recs, []


def labels_check(rx):
    """spec 12 stage 8: reaction y = labels_d1_cr2.json @ M (read here, independent of the pack)."""
    by = {int(r["rxn_id"]): r for r in json.loads(C.LABELS_JSON.read_text())}
    ids = sorted(rx.rxn_id.unique().tolist())
    y10 = np.array([[float(by[i][C.LABEL_KEYS[t]]) for t in C.TARGETS10] for i in ids])
    y4 = pd.DataFrame(y10 @ C.M(), index=ids, columns=T3)
    dy = {t: float(np.abs(rx[rx.target == t].y.to_numpy(float) - y4.loc[rx[rx.target == t].rxn_id, t].to_numpy(float)).max())
          for t in T3}
    return dy


def cmd_collect():
    import prereg_abs as PR
    _GATE["prereg"] = C.sha256(PR.MD) if PR.MD.is_file() else None
    nn_p, nn_r = C.COLLECT / "nn.parquet", C.COLLECT / "nn_report.json"
    if not (nn_p.is_file() and nn_r.is_file()):
        C.not_ready(f"{nn_p} missing (train_abs.py collect)")
    nnr = C.read_json(nn_r)
    if C.sha256(nn_p) != nnr.get("parquet_sha256"):
        C.stop(f"{nn_p} changed after its report")
    recs, missing = load_units()
    if missing:
        print(f"GATE: {len(missing)} XGB units missing: {missing}", file=sys.stderr)
        return 2
    P = PackAbs()
    units = {}
    parts = [pd.read_parquet(nn_p)]
    for x, r in recs.items():
        u = r["unit"]
        now = fingerprint(P, u, P.split(u["fold"]), list(range(len(GRID))))
        if r["fingerprint"] != now:
            C.stop(f"{x} was computed with other code / inputs: keys "
                   f"{sorted(k for k in set(now) | set(r['fingerprint']) if r['fingerprint'].get(k) != now.get(k))}")
        parts.append(pd.read_parquet(C.UNITS / f"{x}.parquet"))
        units[x] = dict(chosen=r["chosen"], val_kcal_mae=r["val_kcal_mae"], n_train=r["n_train"], n_test=r["n_test"],
                        cpu_seconds=r["cpu_seconds"], wall_seconds=r["wall_seconds"], host=r["host"], n_jobs=r["n_jobs"])
    rx = pd.concat(parts, ignore_index=True)
    rx["model"] = rx.model.map(C.ABS_LABEL)
    if rx.model.isna().any() or rx.duplicated(["rxn_id", "fold", "model", "target"]).any():
        C.stop("unknown model label or duplicated reaction prediction rows")
    # every fold's test reactions predicted by each model exactly once
    for k in range(NF):
        want = set(P.rxn_ids(P.split(k)["test"]).tolist())
        for m in C.ABS_LABEL.values():
            got = set(rx[(rx.fold == k) & (rx.model == m) & (rx.target == "barrier")].rxn_id.tolist())
            if got != want:
                C.stop(f"{m} fold {k}: predicted reactions != the test reactions ({len(got)} vs {len(want)})")
    dy = labels_check(rx)
    if max(dy.values()) > 1e-9:
        C.stop(f"spec 12 stage 8: reaction y differs from labels_d1_cr2.json @ M: {dy}")
    ri = np.array([P.row[int(r)] for r in rx.rxn_id])
    for c in ("dipole", "skeleton", "series", "attach"):
        rx[c] = P.attr[c][ri]
    KA = rx[C.RXN_COLS].sort_values(["model", "fold", "target", "rxn_id"]).reset_index(drop=True)
    out = C.RES / "KA_predictions.parquet"
    C.write_parquet(out, KA)
    # pair ΔΔ (spec 7) + C-P
    pairs = pd.concat([AP.pack_test_pairs(P.z, k) for k in range(NF)], ignore_index=True)
    truth = AP.labels_truth(C.LABELS, C.M())
    pp, cp = [], {}
    for m in C.ABS_LABEL.values():
        q = rx[rx.model == m]
        a = AP.assemble(q, pairs, m, rung=C.PAIR_RUNG, arm=ARM)
        cp[m] = AP.cp_check(a, q, pairs, truth)
        pp.append(a)
    KP = pd.concat(pp, ignore_index=True)
    if not all(v["ok"] for v in cp.values()):
        C.stop(f"C-P failed at collect (spec 12 stage 7): {cp}")
    outp = C.RES / "KA_pairs.parquet"
    C.write_parquet(outp, KP[C.PAIR_COLS])
    vrows = []
    for x, r in {**nnr["units"], **units}.items():
        model = x.split("__")[3]
        for t, v in r["val_kcal_mae"].items():
            vrows.append(dict(model=C.ABS_LABEL[model], fold=int(x.split("__f")[-1]), target=t, val_kcal_mae=float(v)))
    C.write_csv(C.RES / "KA_val.csv", pd.DataFrame(vrows))
    rep = dict(spec="docs/specs/ABS3_MODELS.md 7", written=C.now(), n_units=len(nnr["units"]) + len(units),
               n_rows_reactions=int(len(KA)), n_rows_pairs=int(len(KP)),
               n_test_reactions={m: int(KA[(KA.model == m) & (KA.target == "barrier")].shape[0]) for m in C.ABS_LABEL.values()},
               n_test_pairs={m: int(KP[(KP.model == m) & (KP.target == "barrier")].shape[0]) for m in C.ABS_LABEL.values()},
               C_P=cp, y_vs_labels_json=dy, units_nn=nnr["units"], units_xgb=units, gpu_hours=nnr.get("gpu_hours"),
               xgb_cpu_hours=float(sum(u["cpu_seconds"] for u in units.values()) / 3600), nn_report_sha256=C.sha256(nn_r),
               predictions=str(out), predictions_sha256=C.sha256(out), pairs=str(outp), pairs_sha256=C.sha256(outp),
               val_csv_sha256=C.sha256(C.RES / "KA_val.csv"), code_sha256=CODE_SHA, pack_sha256=P.sha["pack_npz"])
    C.write_json(C.RES / "KA_report.json", rep)
    print(json.dumps(C.clean(dict(n_units=rep["n_units"], n_rows_reactions=rep["n_rows_reactions"],
                                  n_rows_pairs=rep["n_rows_pairs"], C_P={m: v["ok"] for m, v in cp.items()},
                                  gpu_hours=rep["gpu_hours"], xgb_cpu_hours=rep["xgb_cpu_hours"])), indent=1))
    return 0


def main(argv):
    a = argv[1:]
    if a == ["list"]:
        print("\n".join(uname(u) for u in unit_list()))
        return 0
    if a == ["smoke"]:
        rec, pq, sec = smoke(PackAbs())
        print(f"[smoke] XGB {sec:.0f} s chosen {rec['chosen']}", flush=True)
        return 0
    if a == ["overfit"]:
        print(json.dumps(C.clean(overfit(PackAbs())), indent=1))
        return 0
    if a[:1] == ["run"] and len(a) == 3:
        return cmd_run(a[1], a[2])
    if a == ["collect"]:
        return cmd_collect()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
