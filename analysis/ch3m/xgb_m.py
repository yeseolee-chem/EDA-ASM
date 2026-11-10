#!/usr/bin/env python3
r"""xgb_m.py — CH3M XGB units (docs/specs/CH3M_MODELS.md 3, 5, 7). CPU, env_espley (xgboost 3.2.0, scikit-learn 1.3.1)
through jobs/cpu_m.slurm CH3MENV=espley, --cpus-per-task=32 (LAD_XGB_NJOBS = workers; results do not depend on it).
Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python xgb_m.py list                     unit names (XGB 10 and 3 modes, folds 0-4)
  python xgb_m.py smoke                    C-8 (XGB): 3 mode, fold 0, grid cells first + last -> results/C8_smoke_xgb.json
                                           (scratch $CH3M/smoke/xgb)
  python xgb_m.py cr                       C-RX: g2 pack, 10 mode, fold 0, full grid -> $CH3M/cr/xgb, compared with the
                                           LADDER unit L__g2__S__XGB__f0 (L_xgb_report.json chosen per target,
                                           L_predictions.parquet g2 . S . XGB fold 0 yhat) -> results/CR_xgb.json
  python xgb_m.py run <10|3> <i|@I> <n>    units [i::n] of the mode on the lf pack -> $CH3M/units/D__<lf10|lf3>__S__XGB__f<k>.*
                                           Gates: PREREG (prereg_m.check), results/C_tests.json all_pass on this code
  python xgb_m.py collect                  XGB units + the TF · GNN part ($CH3M/collect/nn.parquet, train_m.py collect)
                                           -> results/KM_predictions.parquet, KM_val.csv, KM_report.json; C-XB check

Rules = LADDER train_xgb.py arm S (= rev 9 arm C): run_unit below is analysis/d1_ladder/train_xgb.py run_unit
(L1301-1421) with the target list as an argument; prep_fit, leak_check, run_parallel, _task, gate_eval, cost, GRID
(d1_train.grid_cells('XGB'), 12 cells), ESTIMATOR are imported from train_xgb and called unchanged. Per target: inner
GroupKFold(4, core) mean MAE (kcal) chooses the cell (ties grid order), final fit on all outer training pairs, one fit
(random_state 42). use_targets() sets train_xgb.TARGETS / NT / H_IDX / H_TARGETS (read by prep_fit at call time; arm S:
H_IDX = []). The data object PackData is built from the XP pack and replicates train_xgb.XData.split (L862-922) verbatim
for split T1 (inner GroupKFold on the unfiltered outer-train list, fold 0 = split file inner lists, then the usable
filter).
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
import ch3m_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "analysis" / "d1_ladder"))
import train_xgb as TX  # noqa: E402

CODE_VERSION = "ch3m-xgb-1"
MODEL, ARM = "XGB", "S"
PRED_COLS = list(TX.PRED_COLS)
T10, T3 = list(C.TARGETS10), list(C.TARGETS3)
L_XGB_REPORT = C.LAD_RES / "L_xgb_report.json"
L_PRED = C.LAD_RES / "L_predictions.parquet"
CR_TOL = 1e-6
CXB_TOL = 1e-9
CODE_FILES = ("xgb_m.py", "ch3m_common.py")
EXT_FILES = ("d1_ladder/train_xgb.py", "d1_pair/d1_train.py", "d1_pair/d1pair_common.py",
             "espley_xtb_repro/train_ml_single.py")
CODE_SHA = {**{f: C.sha256(HERE / f) for f in CODE_FILES},
            **{f: C.sha256(C.REPO / "analysis" / f) for f in EXT_FILES}}
TRAINER_FILES = CODE_FILES
_GATE: dict = {}
NF = C.NF


def norm(o):
    return json.loads(json.dumps(o, default=C._json_default))


def use_targets(targets):
    """spec 3: train_xgb TARGETS / NT / H_IDX (+ H_TARGETS) = this run's target list (arm S: no H columns)."""
    targets = list(targets)
    if targets not in (T10, T3):
        C.stop(f"unknown target list {targets}")
    TX.TARGETS = list(targets)
    TX.NT = len(targets)
    TX.H_IDX = []
    TX.H_TARGETS = []
    return mode_now()


def mode_now():
    return dict(TARGETS=list(TX.TARGETS), NT=int(TX.NT), H_IDX=list(TX.H_IDX))


def targets_of(mode):
    return T10 if int(mode) == 10 else T3


def uname(u):
    return C.uname(u["rung"], MODEL, u["fold"])


# ================================================================ data (XP pack)
class PackData:
    """the XP pack of a source as the train_xgb.XData attributes that run_unit / prep_fit / leak_check read."""

    def __init__(self, source, mode):
        p, jp = C.XPACK / f"{source}.npz", C.XPACK / f"{source}.json"
        if not (p.is_file() and jp.is_file()):
            C.not_ready(f"{p} / {jp} missing (xgb_pack.py)")
        info = C.read_json(jp)
        if not info.get("ok") or C.sha256(p) != info.get("npz_sha256"):
            C.stop(f"{p}: pack not ok or changed after its json")
        z = np.load(p, allow_pickle=False)
        self.source, self.mode, self.arm = source, int(mode), ARM
        self.rung = "g2" if source == "g2" else C.RUNGS[int(mode)]
        self.X = z["X"]
        self.ra, self.rb = z["ra"].astype(int), z["rb"].astype(int)
        self.usable_pair, self.has_tok = z["usable_pair"].astype(bool), z["has_tok"].astype(bool)
        self.pid = z["pair_id"].astype(str)
        self.P = pd.DataFrame(dict(ref_kind=z["ref_kind"].astype(str), pos=z["pos"].astype(str),
                                   a_rxn=z["a_rxn"], b_rxn=z["b_rxn"]), index=pd.Index(self.pid, name="pair_id"))
        self.y = (z["y10"] if int(mode) == 10 else z["y4"]).astype(float)
        self.dH = np.zeros_like(self.y)                                    # arm S: never used beyond slicing
        self.core = z["core"].astype(str)
        self.pid_row = {q: k for k, q in enumerate(self.pid)}
        self.folds = {k: {key: [str(q) for q in z[f"f{k}_{key}"]] for key in TX.FOLD_KEYS} for k in range(NF)}
        self.feat_cols = list(info["glob_cols"])
        self.feat_cols_sha = TX.C.list_sha([(c,) for c in self.feat_cols])
        self.pair_list_sha = info["pair_list_sha"]
        self.sha = dict(pack_npz=info["npz_sha256"], pack_json=C.sha256(jp), **info["sources_sha256"])
        self.inputs, self.hinfo, self.linfo, self.opened = info.get("inputs"), None, None, []
        self.split_name = TX.SPLIT
        self._splits = {}

    def _ids(self, rows):
        return [str(q) for q in self.pid[np.asarray(rows, dtype=int)]]

    def split(self, k):
        """= train_xgb.XData.split (L862-922) for split T1 (no D0 part)."""
        k = int(k)
        if k in self._splits:
            return self._splits[k]
        rec = self.folds[k]
        out, dropped = {}, {}
        for key in TX.FOLD_KEYS:
            rows = np.array([self.pid_row[q] for q in rec[key]], dtype=int)
            keep = self.usable_pair[rows]
            out[key] = rows[keep]
            dropped[key] = dict(no_tokens=int((~self.has_tok[rows]).sum()), not_usable=int((~keep).sum()))
        st = {key: set(out[key].tolist()) for key in TX.FOLD_KEYS}
        if st["test"] & st["train"]:
            C.stop(f"{self.rung} fold {k}: test pair in train")
        if st["inner_train"] | st["inner_val"] != st["train"] or st["inner_train"] & st["inner_val"]:
            C.stop(f"{self.rung} fold {k}: inner folds do not partition train")
        rx = lambda r: set(self.ra[r].tolist()) | set(self.rb[r].tolist())          # noqa: E731
        if rx(out["test"]) & rx(out["train"]):
            C.stop(f"{self.rung} fold {k}: test reaction in train")
        from sklearn.model_selection import GroupKFold
        tr_all = np.array([self.pid_row[q] for q in rec["train"]], dtype=int)
        inner = []
        for j, (ia, ib) in enumerate(GroupKFold(TX.C.N_INNER).split(tr_all, groups=self.core[tr_all])):
            itr_all, iva_all = tr_all[ia], tr_all[ib]
            if j == 0 and (self._ids(itr_all) != list(rec["inner_train"]) or self._ids(iva_all) != list(rec["inner_val"])):
                C.stop(f"{self.rung} fold {k}: GroupKFold(4) fold 0 differs from the split file inner_train / inner_val "
                       "(train_xgb note: resubmit on the rev 9 node type)")
            if set(self.core[itr_all].tolist()) & set(self.core[iva_all].tolist()):
                C.stop(f"{self.rung} fold {k} inner {j}: a core on both sides")
            itr, iva = itr_all[self.usable_pair[itr_all]], iva_all[self.usable_pair[iva_all]]
            if not len(itr) or not len(iva) or set(itr.tolist()) & set(iva.tolist()):
                C.stop(f"{self.rung} fold {k} inner {j}: empty or overlapping inner sides")
            inner.append(dict(train=itr, val=iva, dropped=dict(train=int(len(itr_all) - len(itr)),
                                                              val=int(len(iva_all) - len(iva)))))
        if not (np.array_equal(inner[0]["train"], out["inner_train"]) and np.array_equal(inner[0]["val"], out["inner_val"])):
            C.stop(f"{self.rung} fold {k}: filtered inner fold 0 differs from the filtered split file lists (L-9)")
        ids = {key: self._ids(out[key]) for key in TX.FOLD_KEYS}
        ls = TX.C.list_sha
        out.update(inner=inner, dropped=dropped,
                   pairs_sha=ls([(q,) for q in ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"] + ["|"]
                                 + ids["inner_val"]]),
                   val_sha=ls([(q,) for q in ids["inner_val"]]), test_sha=ls([(q,) for q in ids["test"]]),
                   inner_sha=ls([(q,) for f in inner for q in self._ids(f["train"]) + ["|"] + self._ids(f["val"]) + ["||"]]))
        self._splits[k] = out
        return out


# ================================================================ unit
def _check_root(root, smoke):
    r = C.assert_write_ok(Path(root) / "probe").parent
    p = Path(C.UNITS).resolve()
    if r == p or p in r.parents:
        if smoke:
            C.stop(f"smoke / test runs never write {p}")
        if not _GATE.get("prereg"):
            C.stop(f"writing {p} needs the PREREG gate (xgb_m.py run)")
    return r


def fingerprint(D, u, sp, idx, targets, smoke=False):
    return norm(dict(code=CODE_VERSION, code_sha256=CODE_SHA, rung=u["rung"], arm=ARM, model=MODEL, targets=list(targets),
                     conversion=C.M_ROWS if list(targets) == T3 else None, source=D.source, inputs_sha256=D.sha,
                     pair_list=D.pair_list_sha, split_pairs=sp["pairs_sha"], inner_folds=sp["inner_sha"],
                     prereg=None if smoke else _GATE.get("prereg"), grid=[TX.GRID[i] for i in idx],
                     estimator=TX.ESTIMATOR, feat_cols=D.feat_cols_sha, tx_mode=mode_now(),
                     rules=dict(inner=f"GroupKFold({TX.C.N_INNER}) by core on the unfiltered outer-train list, then the "
                                      "usable filter; fold 0 = split file inner lists",
                                select="per target lowest mean inner MAE (kcal), ties grid order; final fit on all training pairs",
                                seeds="one fit, GRIDS estimator random_state 42",
                                antisym=dict(final_tol=TX.F.ANTISYM_TOL, model_tol=TX.C.ANTISYM_TOL, n=TX.C.ANTISYM_N)),
                     versions=TX.versions()))


def run_unit(D, u, targets, root, cells=None, smoke=False):
    """= train_xgb.run_unit (L1301-1421) for arm S with the target list as an argument."""
    targets = list(targets)
    use_targets(targets)
    nt = len(targets)
    if D.y.shape[1] != nt:
        C.stop(f"data y has {D.y.shape[1]} columns, targets {targets}")
    root = _check_root(root, smoke)
    root.mkdir(parents=True, exist_ok=True)
    if (u["rung"], u.get("arm")) != (D.rung, D.arm) or u.get("model") != MODEL:
        C.stop(f"unit {u} does not match the data ({D.rung}, {D.arm}, {MODEL})")
    GRID = TX.GRID
    name = uname(u)
    idx = list(range(len(GRID))) if cells is None else sorted({int(c) for c in cells})
    sp = D.split(u["fold"])
    fp = fingerprint(D, u, sp, idx, targets, smoke)
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
    if min(len(tr), len(te), *[len(f["train"]) for f in inner], *[len(f["val"]) for f in inner]) == 0:
        C.stop(f"{name}: empty split list")
    rng = np.random.default_rng(TX.C.SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10 ** 6)
    gp = te[rng.choice(len(te), size=min(TX.C.ANTISYM_N, len(te)), replace=False)]
    fits = []
    for f in inner:
        pf = TX.prep_fit(D, f["train"], dict(val=f["val"]), gp)
        pf["leak"] = TX.leak_check(D, pf, (f["val"], te), te)
        fits.append(pf)
    jobs = sorted([(j, ti, ci) for ti in range(nt) for j in range(len(fits)) for ci in idx], key=lambda x: -TX.cost(GRID[x[2]]))
    print(f"  {name}: {len(jobs)} tuning fits on {TX.N_JOBS} workers ({len(D.feat_cols)} features)", flush=True)
    res = TX.run_parallel([(fits[j]["zd"], fits[j]["zm"], fits[j]["ys"][ti], fits[j]["sy"][ti], GRID[ci], fits[j]["evals"])
                           for j, ti, ci in jobs])
    mae = np.full((nt, len(fits), len(idx)), np.nan)
    gmax = {t: dict(anti_max=0.0, self_max=0.0, model_anti_max=0.0, model_self_max=0.0) for t in targets}
    tune_s = []
    for (j, ti, ci), (kc, sec) in zip(jobs, res):
        f, c = fits[j], idx.index(ci)
        yv = D.y[inner[j]["val"], ti]
        mae[ti, j, c] = float(np.mean(np.abs(f["base"]["val"][:, ti] + kc["val"] - yv)))
        g = TX.gate_eval(kc, f["base"], ti)
        if not g["ok"]:
            C.stop(f"gate C-4 failed ({name}, {targets[ti]}, inner {j}, cell {GRID[ci]}): {g}")
        for k in gmax[targets[ti]]:
            gmax[targets[ti]][k] = max(gmax[targets[ti]][k], g[k])
        tune_s.append(sec)
    if not np.isfinite(mae).all():
        C.stop(f"{name}: non-finite inner MAE")
    tune, best = {}, {}
    for ti, t in enumerate(targets):
        means = mae[ti].mean(axis=0)
        b = int(np.argmin(means))
        best[t] = idx[b]
        tune[t] = dict(grid_index=idx, grid_mae=means.tolist(), fold_mae=mae[ti].tolist(),
                       best=dict(cell=GRID[idx[b]], grid_index=idx[b]), inner_cv_mae=float(means[b]),
                       inner_fold_mae=mae[ti][:, b].tolist(), val_kcal_mae=float(mae[ti, 0, b]),
                       sy_inner=[float(f["sy"][ti]) for f in fits], gate_max=gmax[t])
    fin = TX.prep_fit(D, tr, dict(test=te, train=tr), gp)
    fin["leak"] = TX.leak_check(D, fin, (te,), te)
    fres = TX.run_parallel([(fin["zd"], fin["zm"], fin["ys"][ti], fin["sy"][ti], GRID[best[t]], fin["evals"])
                            for ti, t in enumerate(targets)])
    yhat = np.zeros((len(te), nt))
    final = {}
    for ti, (t, (kc, sec)) in enumerate(zip(targets, fres)):
        g = TX.gate_eval(kc, fin["base"], ti)
        if not g["ok"]:
            C.stop(f"gate C-4 failed ({name}, {t}, final): {g}")
        yhat[:, ti] = fin["base"]["test"][:, ti] + kc["test"]
        ptr = fin["base"]["train"][:, ti] + kc["train"]
        final[t] = dict(cell=GRID[best[t]], grid_index=best[t], sy=float(fin["sy"][ti]),
                        train_mae=float(np.mean(np.abs(ptr - D.y[tr, ti]))), gate_M1=g, seconds=float(sec))
    if not np.isfinite(yhat).all():
        C.stop(f"{name}: non-finite test predictions")
    base = fin["base"]["test"]
    pid = D.pid[te].astype(str)
    rk, ps = D.P.ref_kind.to_numpy()[te], D.P.pos.to_numpy()[te]
    parts = [pd.DataFrame(dict(pair_id=pid, ref_kind=rk, pos=ps, fold=int(u["fold"]), rung=u["rung"], arm=ARM, model=MODEL,
                               target=t, y=D.y[te, j], yhat=yhat[:, j], base=base[:, j], beta=0.0, yhat_seed_sd=0.0))
             for j, t in enumerate(targets)]
    pred = pd.concat(parts, ignore_index=True)[PRED_COLS]
    pq = root / f"{name}.parquet"
    C.write_parquet(pq, pred)
    rec = dict(unit=u, name=name, model=MODEL, targets=targets, fingerprint=fp, smoke=bool(smoke), gates_ok=True, n_seeds=1,
               chosen={t: dict(cell=GRID[best[t]], grid_index=best[t]) for t in targets}, tune=tune, final=final,
               val_kcal_mae={t: tune[t]["val_kcal_mae"] for t in targets}, val_pairs_sha256=sp["val_sha"],
               test_pairs_sha256=sp["test_sha"], n_val=int(len(sp["inner_val"])), n_train=int(len(tr)), n_test=int(len(te)),
               n_inner_train=int(len(sp["inner_train"])), n_inner_val=int(len(sp["inner_val"])),
               inner_folds=[dict(n_train=int(len(f["train"])), n_val=int(len(f["val"])), dropped=f["dropped"],
                                 scale=pf["scale"], leak=pf["leak"]) for f, pf in zip(inner, fits)],
               final_scale=fin["scale"], final_leak=fin["leak"], dropped=sp["dropped"], n_gate_pairs=int(len(gp)),
               n_features=len(D.feat_cols), parquet_sha256=C.sha256(pq), inputs=D.inputs,
               fit_seconds=dict(tune=tune_s, final=[final[t]["seconds"] for t in targets]),
               cpu_seconds=float(sum(tune_s) + sum(final[t]["seconds"] for t in targets)), n_jobs=TX.N_JOBS,
               wall_seconds=time.time() - t_unit, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    C.write_json(js, rec)
    print(f"  {name}: done ({rec['wall_seconds']:.0f} s wall, {rec['cpu_seconds'] / 3600:.2f} CPU-h) val "
          f"{json.dumps({t: round(v, 4) for t, v in rec['val_kcal_mae'].items()})}", flush=True)
    return rec


# ================================================================ gates
def gate_prereg():
    import prereg_m as PR
    rec = PR.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    p = C.RES / "C_tests.json"
    c = C.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        C.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K)")
    rec = (c.get("trainer_code_sha256") or {}).get("xgb_m") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA[f]]
    if diff:
        C.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}")
    return dict(c_tests_sha256=C.sha256(p))


# ================================================================ commands
def cmd_smoke():
    root = C.SMOKE / "xgb"
    _check_root(root, True)
    if root.is_dir():
        for p in root.glob("D__*"):
            C.assert_write_ok(p)
            p.unlink()
    D = PackData("lf", 3)
    u = dict(rung=D.rung, arm=ARM, model=MODEL, fold=0)
    t0 = time.time()
    rec = run_unit(D, u, T3, root, cells=[0, len(TX.GRID) - 1], smoke=True)
    P = pd.read_parquet(root / f"{uname(u)}.parquet")
    ct = dict(tx_targets_is_TARGETS3=list(TX.TARGETS) == T3, tx_NT_4=int(TX.NT) == 4, tx_H_IDX_empty=list(TX.H_IDX) == [],
              record_targets_4=list(rec["targets"]) == T3 and sorted(rec["chosen"]) == sorted(T3),
              prediction_targets_4=sorted(P.target.unique()) == sorted(T3))
    gate_max = max(max(v["anti_max"], v["self_max"]) for v in (t["gate_max"] for t in rec["tune"].values()))
    leak_ok = all(f["leak"]["ok"] for f in rec["inner_folds"]) and rec["final_leak"]["ok"]
    out = dict(check="C-8 (XGB) + C-T (XGB)", ok=bool(all(ct.values()) and rec["gates_ok"] and leak_ok), C_T=ct,
               gate_C4_max=gate_max, leak_C5=leak_ok, seconds=time.time() - t0, cpu_seconds=rec["cpu_seconds"],
               chosen=rec["chosen"], n_train=rec["n_train"], n_test=rec["n_test"], versions=TX.versions(), code_sha256=CODE_SHA)
    C.write_json(C.RES / "C8_smoke_xgb.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "C_T", "gate_C4_max", "seconds")}, indent=1))
    return 0 if out["ok"] else 3


def cmd_cr():
    rep = C.read_json(L_XGB_REPORT)
    key = "L__g2__S__XGB__f0"
    ref = (rep.get("units") or {}).get(key)
    if ref is None:
        C.stop(f"{L_XGB_REPORT}: no unit {key}")
    D = PackData("g2", 10)
    u = dict(rung="g2", arm=ARM, model=MODEL, fold=0)
    root = C.CR / "xgb"
    rec = run_unit(D, u, T10, root)
    mine = pd.read_parquet(root / f"{uname(u)}.parquet")
    L = pd.read_parquet(L_PRED)
    theirs = L[(L.rung == "g2") & (L.arm == "S") & (L.model == "XGB") & (L.fold == 0)]
    m = mine.merge(theirs[["pair_id", "target", "y", "yhat"]], on=["pair_id", "target"], suffixes=("", "_lad"), how="outer",
                   indicator=True)
    same = bool((m._merge == "both").all())
    dyh = {t: float(np.abs(g.yhat - g.yhat_lad).max()) for t, g in m.groupby("target")} if same else {}
    cells = {t: dict(mine=rec["chosen"][t]["grid_index"], ladder=(ref.get("chosen") or {}).get(t, {}).get("grid_index"))
             for t in T10}
    dcv = {t: abs(rec["tune"][t]["inner_cv_mae"] - float((ref.get("inner_cv_mae") or {}).get(t, np.nan))) for t in T10}
    checks = dict(same_cell_per_target=all(v["mine"] == v["ladder"] for v in cells.values()),
                  yhat_within_1e6=same and max(dyh.values()) <= CR_TOL, same_test_pairs=same,
                  same_y=same and float(np.abs(m.y - m.y_lad).max()) <= 1e-12, gates=rec["gates_ok"])
    out = dict(check="C-RX", ok=bool(all(checks.values())), checks=checks, cells=cells, max_abs_dyhat=dyh,
               inner_cv_mae_abs_diff=dcv, val_kcal_mae=rec["val_kcal_mae"], val_kcal_mae_ladder=ref.get("val_kcal_mae"),
               cpu_seconds=rec["cpu_seconds"], cpu_seconds_ladder=ref.get("cpu_seconds"), wall_seconds=rec["wall_seconds"],
               n_jobs=TX.N_JOBS, host=socket.gethostname(), versions=TX.versions(), code_sha256=CODE_SHA, ladder_unit=key)
    C.write_json(C.RES / "CR_xgb.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "checks", "max_abs_dyhat")}, indent=1))
    return 0 if out["ok"] else 3


def unit_list(mode=None):
    modes = (10, 3) if mode is None else (int(mode),)
    return [dict(rung=C.RUNGS[m], arm=ARM, model=MODEL, fold=k, mode=m) for m in modes for k in range(NF)]


def cmd_run(mode, i, n):
    mode = int(mode)
    i, n = C.array_task(i), int(n)
    us = unit_list(mode)[i::n]
    if not us:
        return 0
    pre = gate_prereg()
    gate_c_tests()
    D = PackData("lf", mode)
    print(f"[K] PREREG ok (git {pre.get('git_head')}), C_tests ok, {TX.versions()}, N_JOBS {TX.N_JOBS}, XGB {mode} slice "
          f"{i}/{n}: {len(us)} units", flush=True)
    for u in us:
        run_unit(D, dict(rung=u["rung"], arm=ARM, model=MODEL, fold=u["fold"]), targets_of(mode), C.UNITS)
    return 0


def load_units(mode):
    names = [uname(u) for u in unit_list(mode)]
    missing = [x for x in names if not ((C.UNITS / f"{x}.json").is_file() and (C.UNITS / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: C.read_json(C.UNITS / f"{x}.json") for x in names}
    for x, r in recs.items():
        if C.sha256(C.UNITS / f"{x}.parquet") != r.get("parquet_sha256") or r.get("smoke") or not r.get("gates_ok"):
            C.stop(f"{x}: parquet changed, smoke or gate failure")
        if not (all(f["leak"]["ok"] for f in r["inner_folds"]) and r["final_leak"]["ok"]):
            C.stop(f"{x}: C-5 leak check failed")
        if list(r.get("targets") or []) != targets_of(mode):
            C.stop(f"{x}: targets {r.get('targets')}")
    return recs, []


def cmd_collect():
    import prereg_m as PR
    _GATE["prereg"] = C.sha256(PR.MD) if PR.MD.is_file() else None
    nn_p, nn_r = C.CH3M / "collect" / "nn.parquet", C.CH3M / "collect" / "nn_report.json"
    if not (nn_p.is_file() and nn_r.is_file()):
        C.not_ready(f"{nn_p} missing (train_m.py collect)")
    nnr = C.read_json(nn_r)
    if C.sha256(nn_p) != nnr.get("parquet_sha256"):
        C.stop(f"{nn_p} changed after its report")
    parts, units, missing_all = [pd.read_parquet(nn_p)[PRED_COLS]], {}, []
    for mode in (10, 3):
        recs, missing = load_units(mode)
        if missing:
            missing_all += missing
            continue
        D = PackData("lf", mode)
        use_targets(targets_of(mode))                     # the fingerprint records the train_xgb target mode
        for x, r in recs.items():
            u = r["unit"]
            now = fingerprint(D, u, D.split(u["fold"]), list(range(len(TX.GRID))), targets_of(mode))
            if r["fingerprint"] != now:
                C.stop(f"{x} was computed with other code / inputs: keys "
                       f"{sorted(k for k in set(now) | set(r['fingerprint']) if r['fingerprint'].get(k) != now.get(k))}")
            parts.append(pd.read_parquet(C.UNITS / f"{x}.parquet")[PRED_COLS])
            units[x] = dict(chosen=r["chosen"], val_kcal_mae=r["val_kcal_mae"], n_train=r["n_train"], n_test=r["n_test"],
                            cpu_seconds=r["cpu_seconds"], wall_seconds=r["wall_seconds"], host=r["host"], n_jobs=r["n_jobs"])
    if missing_all:
        print(f"GATE: {len(missing_all)} XGB units missing: {missing_all}", file=sys.stderr)
        return 2
    pred = pd.concat(parts, ignore_index=True)
    if pred.duplicated(["pair_id", "rung", "model", "target"]).any():
        C.stop("duplicated prediction rows")
    # C-XB: XGB 3-mode barrier = XGB 10-mode barrier (same data and rules)
    b10 = pred[(pred.model == MODEL) & (pred.rung == "lf10") & (pred.target == "barrier")].set_index("pair_id").yhat
    b3 = pred[(pred.model == MODEL) & (pred.rung == "lf3") & (pred.target == "barrier")].set_index("pair_id").yhat
    same_ids = set(b10.index) == set(b3.index)
    dxb = float(np.abs(b10 - b3.reindex(b10.index)).max()) if same_ids else float("inf")
    cell_eq = all(units[C.uname("lf10", MODEL, k)]["chosen"]["barrier"] == units[C.uname("lf3", MODEL, k)]["chosen"]["barrier"]
                  for k in range(NF))
    cxb = dict(ok=bool(same_ids and dxb <= CXB_TOL and cell_eq), max_abs_diff=dxb, same_pairs=same_ids, same_cells=cell_eq,
               tol=CXB_TOL)
    out = C.RES / "KM_predictions.parquet"
    C.write_parquet(out, pred)
    vrows = []
    for x, r in {**nnr["units"], **units}.items():
        rung, model = x.split("__")[1], x.split("__")[3]
        for t, v in r["val_kcal_mae"].items():
            vrows.append(dict(rung=rung, model=model, fold=int(x.split("__f")[-1]), target=t, val_kcal_mae=float(v)))
    C.write_csv(C.RES / "KM_val.csv", pd.DataFrame(vrows))
    rep = dict(spec="docs/specs/CH3M_MODELS.md 7", written=C.now(), n_rows=int(len(pred)),
               n_units=len(nnr["units"]) + len(units), C_XB=cxb, units_nn=nnr["units"], units_xgb=units,
               gpu_hours=nnr.get("gpu_hours"), xgb_cpu_hours=float(sum(u["cpu_seconds"] for u in units.values()) / 3600),
               nn_report_sha256=C.sha256(nn_r), predictions=str(out), predictions_sha256=C.sha256(out),
               val_csv_sha256=C.sha256(C.RES / "KM_val.csv"), code_sha256=CODE_SHA)
    C.write_json(C.RES / "KM_report.json", rep)
    print(json.dumps(dict(n_rows=rep["n_rows"], n_units=rep["n_units"], C_XB=cxb, gpu_hours=rep["gpu_hours"],
                          xgb_cpu_hours=rep["xgb_cpu_hours"]), indent=1))
    if not cxb["ok"]:
        C.stop(f"C-XB failed (spec 13 stage 7): {cxb}")
    return 0


def main(argv):
    a = argv[1:]
    if a == ["list"]:
        print("\n".join(uname(u) for u in unit_list()))
        return 0
    if a == ["smoke"]:
        return cmd_smoke()
    if a == ["cr"]:
        return cmd_cr()
    if a[:1] == ["run"] and len(a) == 4:
        return cmd_run(a[1], a[2], a[3])
    if a == ["collect"]:
        return cmd_collect()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
