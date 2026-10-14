#!/usr/bin/env python3
"""d1_models.py — SPEC_D1_ALL Phase D: M0–M4 exactly as pre-registered in analysis/d1_stats/PREREG_D1.md.

  python d1_models.py check                  fast_grid vs sklearn GridSearchCV on a subsample (must pass first)
  python d1_models.py m0                     rev 5 EXT_SEL KRR_rbf refit on D0 dev (sklearn, as final_eval.fit_unit),
                                             reproduction check vs the rev 5 unit records, predictions for every D1 row
  python d1_models.py split <task> <ntask>   M1 + M2 (+ M3 for LOCO) for the splits task, task+ntask, … of LOCO+LODO+LOXO
  python d1_models.py m4                     final M1 model -> the D0 lockbox, once (PREREG §6)
  python d1_models.py report                 metrics, core bootstrap, MODEL_REPORT.md, results_models/*.csv

Every command stops unless PREREG_D1.md carries an approval line ("- 승인: YYYY-MM-DD …"), is committed and equals HEAD
(its sha256 is part of every cache fingerprint), and runs inside SLURM only. Caches (idempotent, atomic) live in
$D1_MODEL_ROOT (default /gpfs/tmp_cpu2/yeseo1ee/d1_models): check.json, m0.npz/.json, split/<scheme>__<name>.npz/.json,
m4/{pred.npz, LOCKBOX_OPENED.json, result.json}. A cache made from other inputs stops the run (move it away).

fast_grid = GridSearchCV(_make_pipe(KernelRidge(rbf)), GRIDS["KRR_rbf"]) with the same arithmetic: per fold StandardScaler
on X and on each target (TransformedTargetRegressor), sklearn rbf_kernel once per (fold, gamma), scipy.linalg.solve(...,
assume_a="pos") once per (fold, gamma, alpha) for the 9 targets together, mean fold MAE, first minimum in ParameterGrid
order (alpha outer, gamma inner). `check` verifies it against sklearn before anything else is fitted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
XR = HERE.parent / "espley_xtb_repro"
sys.path.insert(0, str(XR))
sys.path.insert(0, str(HERE))
import rev5_common as R5  # noqa: E402
import train_ml_single as T  # noqa: E402
from d1s_prep import load_labels, pairs_vs_H, sibling_pairs  # noqa: E402

CODE_VERSION = "d1-models-1"
MODEL_ROOT = Path(os.environ.get("D1_MODEL_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/d1_models"))
D1R = REPO / "D1_build/d1_run" if (REPO / "D1_build/d1_run").is_dir() else Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run")
LABELS_D1 = Path(os.environ.get("D1_LABELS_FINAL", D1R / "results/labels_d1.json"))
GATE_C_REASONS = Path(os.environ.get("D1_GATE_C_REASONS", HERE / "results_final/gate_c_fail_reasons.csv"))
REV5_UNITS = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/lockbox/units")
REV5_LOCK_PRED = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/lockbox/lockbox_predictions.parquet")
D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")   # = g1_geom.SMILES_CSV
PREREG = HERE / "PREREG_D1.md"
RES = HERE / "results_models"
TGT = list(R5.TARGETS9)
SEED = R5.SEED                       # 20261001: inner CV, M3 permutations, lockbox bootstrap
BOOT_SEED, N_BOOT_CORE = 20261014, 1000
M3_K, M3_SEEDS = [0, 5, 10, 20, 40], 5
MIN_CTX = 5
EST, GRID = T.GRIDS["KRR_rbf"]
AK, GK = "regressor__kernelridge__alpha", "regressor__kernelridge__gamma"
ALPHAS, GAMMAS = list(GRID[AK]), list(GRID[GK])
CY = "carbonyl ylide"


def die(msg, code=2):
    print(f"STOP: {msg}", flush=True)
    sys.exit(code)


def sha(p):
    return R5.sha256(p)


def jhash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ================================================================ gates
def gate_prereg():
    if not os.environ.get("SLURM_JOB_ID"):
        die("runs inside a SLURM job only (sbatch analysis/d1_stats/jobs/phase_d.slurm …), never on the login node")
    txt = PREREG.read_text()
    m = re.search(r"^- 승인: (\d{4}-\d{2}-\d{2}) \S", txt, re.M)
    if not m:
        die(f"{PREREG} has no approval line ('- 승인: YYYY-MM-DD 이름') — PREREG_D1 is a draft")
    rel = str(PREREG.relative_to(REPO))
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    if not (tracked and clean):
        die(f"{rel} is not committed or differs from HEAD — commit the approved pre-registration first")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return dict(prereg_sha256=sha(PREREG), approved=m.group(1), git_head=head)


# ================================================================ data
class Data:
    pass


def load_data(keep_lockbox=False):
    D = Data()
    cols = list(T.resolve_arm("EXT_SEL"))
    if len(cols) != 133:
        die(f"EXT_SEL has {len(cols)} columns, expected 133")
    ml, info = T.load_ml(feat_path=R5.FEAT_G1, rows_path=str(R5.ROWS5), targets=TGT, geom="g1", ext=True, ext_path=R5.FEAT_EXT)
    rid = ml.rxn_id.astype(int).to_numpy()
    pos = pd.Series(np.arange(len(rid)), index=rid)
    dev = pd.read_csv(R5.RES5 / "dev_ids.csv").rxn_id.astype(int).tolist()
    lock = pd.read_csv(R5.LOCKBOX).rxn_id.astype(int).tolist()
    if set(dev) & set(lock) or len(dev) != 4112 or len(lock) != 726:
        die("dev / lockbox partition is not the rev 5 one")
    i0 = pos.loc[dev].to_numpy()
    D.ids0 = rid[i0]
    D.X0 = ml[cols].to_numpy(float)[i0]
    D.Y0 = ml[TGT].to_numpy(float)[i0]
    if keep_lockbox:                                    # M4 only: lockbox features now, targets after LOCKBOX_OPENED
        il = pos.loc[lock].to_numpy()
        D.idsL, D.XL = rid[il], ml[cols].to_numpy(float)[il]
    del ml
    # D1
    lab = load_labels(str(LABELS_D1))
    base = pd.read_parquet(R5.D1_FEAT)
    ext = pd.read_parquet(R5.D1_FEAT_EXT).drop(columns=["job_id"])
    F = base.merge(ext, on="rxn_id", how="inner", suffixes=("", "_ext"), validate="1:1")
    st = pd.read_csv(GATE_C_REASONS)[["job_id", "ext_sel_status"]]
    F = F.merge(st, on="job_id", how="left", validate="1:1")
    meta = ["job_id", "core", "dipole", "skeleton", "side", "X", "attach", "is_parent", "cell_id", "orient", "face",
            "stereo_index", "q_bad", "barrier_kcal", "series", "R", "R2p", "regio"]
    F = F.merge(lab[meta], on="job_id", how="inner", validate="1:1")
    n_all = len(F)
    F = F[(F.ext_sel_status == "ok") & ~F.q_bad].reset_index(drop=True)
    if not np.allclose(F.dft_barrier_kcal, F.barrier_kcal, rtol=0, atol=1e-6):
        die("D1 dft_barrier_kcal != label barrier_kcal (|Δ| > 1e-6 kcal/mol)")
    D.d1 = F[meta].copy()
    D.X1 = F[cols].to_numpy(float)
    D.Y1 = F[TGT].to_numpy(float)
    if not (np.isfinite(D.X1).all() and np.isfinite(D.Y1).all() and np.isfinite(D.X0).all()):
        die("non-finite feature / target values")
    D.hygiene_would_drop = int((~T.hygiene(F)).sum())
    D.n_d1_all, D.cols = n_all, cols
    # the same reaction in D1 and D0 (atom maps removed, canonical; PREREG §1): D1 rows equal to a lockbox reaction are
    # left out of the M4 training; equal to a dev reaction: counted (M0 / M1 test rows that D0 training already holds)
    s0 = pd.read_csv(D0_SMILES, usecols=["rxn_id", "rxn_smiles"]).set_index("rxn_id").rxn_smiles
    k1 = [rxn_keys(s) for s in lab.set_index("job_id").rxn_smiles.loc[D.d1.job_id]]
    kd = [set(rxn_keys(s0.loc[r])[i] for r in D.ids0) for i in (0, 1)]
    kl = [set(rxn_keys(s0.loc[r])[i] for r in lock) for i in (0, 1)]
    D.ov_lock = np.array([k[1] in kl[1] for k in k1])                # flat key (stereo ignored): conservative
    D.ov_lock_iso = int(sum(k[0] in kl[0] for k in k1))
    D.ov_dev, D.ov_dev_iso = int(sum(k[1] in kd[1] for k in k1)), int(sum(k[0] in kd[0] for k in k1))
    # pairs and siblings among the model rows
    L = lab[lab.job_id.isin(set(D.d1.job_id))].copy()
    jpos = pd.Series(np.arange(len(D.d1)), index=D.d1.job_id)
    P = pairs_vs_H(L)
    D.pairs = P[["job_id", "parent_job", "core", "dipole", "side", "X", "attach", "ctx"]].reset_index(drop=True)
    D.ic, D.ip = jpos.loc[D.pairs.job_id].to_numpy(), jpos.loc[D.pairs.parent_job].to_numpy()
    R = sibling_pairs(L[L.orient.isin(["a", "b"])], ["cell_id", "face", "stereo_index"], "orient", "a", "b")
    E = sibling_pairs(L[L.face.isin(["exo", "endo"])], ["cell_id", "orient"], "face", "exo", "endo")
    D.sibs = {k: (jpos.loc[S.job_a].to_numpy(), jpos.loc[S.job_b].to_numpy(), S.dipole.to_numpy())
              for k, S in (("regio", R), ("exoendo", E)) if len(S)}
    D.fp = dict(code=CODE_VERSION, columns=jhash(cols), grid=jhash(GRID), targets=TGT,
                d0_features=info["features_sha256"], d0_ext=info["ext_features_sha256"], rows=info["rows_sha256"],
                dev=sha(R5.RES5 / "dev_ids.csv"), lockbox=sha(R5.LOCKBOX), d1_labels=sha(LABELS_D1),
                d1_features=sha(R5.D1_FEAT), d1_ext=sha(R5.D1_FEAT_EXT), gate_c=sha(GATE_C_REASONS), d0_smiles=sha(D0_SMILES))
    print(f"data: D0 dev {len(D.ids0)}, D1 rows {len(D.d1)} of {n_all} (q_bad / EXT_SEL missing removed), pairs "
          f"{len(D.pairs)}, siblings {[(k, len(v[0])) for k, v in D.sibs.items()]}, rev5 hygiene would drop "
          f"{D.hygiene_would_drop} D1 rows (not applied); same reaction as D0 lockbox {int(D.ov_lock.sum())} "
          f"(with stereo {D.ov_lock_iso}), as D0 dev {D.ov_dev} (with stereo {D.ov_dev_iso})", flush=True)
    return D


def rxn_keys(smi):
    """(isomeric key, stereo-free key) of a reaction SMILES: atom maps removed, canonical, reactants sorted."""
    from rdkit import Chem
    r, p = str(smi).split(">>")

    def can(s, iso):
        m = Chem.MolFromSmiles(s)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(m, isomericSmiles=iso)
    return tuple(".".join(sorted(can(x, iso) for x in r.split("."))) + ">>" + can(p, iso) for iso in (True, False))


# ================================================================ fast exact KRR grid
def _solve(K, Y, alpha):
    K2 = K.copy()
    K2.flat[::K2.shape[0] + 1] += alpha
    return linalg.solve(K2, Y, assume_a="pos", overwrite_a=True)


def fold_scores(X, Y, folds):
    """MAE[fold, alpha, gamma, target] of the KRR_rbf grid (same arithmetic as GridSearchCV on _make_pipe)."""
    S = np.zeros((len(folds), len(ALPHAS), len(GAMMAS), Y.shape[1]))
    for f, (a, b) in enumerate(folds):
        xs = StandardScaler().fit(X[a]); A, Bv = xs.transform(X[a]), xs.transform(X[b])
        ys = StandardScaler().fit(Y[a]); Ys = ys.transform(Y[a])
        for j, g in enumerate(GAMMAS):
            K, Kv = rbf_kernel(A, A, gamma=g), rbf_kernel(Bv, A, gamma=g)
            for i, al in enumerate(ALPHAS):
                Pv = ys.inverse_transform(Kv @ _solve(K, Ys, al))
                S[f, i, j] = np.abs(Pv - Y[b]).mean(axis=0)
    return S


def best_of(S):
    """Per target: (alpha, gamma) of the first minimum of the fold-mean MAE in ParameterGrid order, and the cv MAE."""
    M = S.mean(axis=0).reshape(-1, S.shape[-1])          # (alpha * gamma, target), alpha-major = ParameterGrid order
    k = M.argmin(axis=0)
    return [(ALPHAS[q // len(GAMMAS)], GAMMAS[q % len(GAMMAS)]) for q in k], M[k, np.arange(M.shape[1])]


def fit_predict(X, Y, Xp, params):
    """Refit on (X, Y) with per-target (alpha, gamma) and predict Xp (n_p, targets)."""
    out = np.empty((len(Xp), Y.shape[1]))
    xs = StandardScaler().fit(X); A, Bp = xs.transform(X), xs.transform(Xp)
    ys = StandardScaler().fit(Y); Ys = ys.transform(Y)
    for g in sorted({p[1] for p in params}):
        K, Kp = rbf_kernel(A, A, gamma=g), rbf_kernel(Bp, A, gamma=g)
        for al in sorted({p[0] for p in params if p[1] == g}):
            c = [t for t, p in enumerate(params) if p == (al, g)]
            out[:, c] = (Kp @ _solve(K, Ys[:, c], al)) * ys.scale_[c] + ys.mean_[c]
    return out


def group_folds(groups):
    return list(GroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(groups)), groups=groups))


def train_groups(D, m1):
    return np.concatenate([np.array([f"d0:{r}" for r in D.ids0]), np.array(["d1:" + c for c in D.d1.core.to_numpy()[m1]])])


# ================================================================ cache helpers
def save_cache(stem: Path, arrays: dict, rec: dict):
    stem.parent.mkdir(parents=True, exist_ok=True)

    def _w(t):
        with open(t, "wb") as f:
            np.savez_compressed(f, **arrays)
    R5.write_atomic(stem.with_suffix(".npz"), _w)
    R5.write_json(stem.with_suffix(".json"), rec)           # JSON last = done marker


def load_cache(stem: Path, fp):
    if not stem.with_suffix(".json").is_file() or not stem.with_suffix(".npz").is_file():
        return None
    rec = json.loads(stem.with_suffix(".json").read_text())
    if rec.get("fp") != fp:
        die(f"{stem}.json was made from other inputs — move {MODEL_ROOT} away to recompute")
    return rec, dict(np.load(stem.with_suffix(".npz"), allow_pickle=False))


def full_fp(D, pre):
    return dict(D.fp, prereg=pre["prereg_sha256"])


# ================================================================ commands
def cmd_check(D, pre):
    out = MODEL_ROOT / "check.json"
    rng = np.random.default_rng(0)
    dp = sorted(D.d1.dipole.unique())[0]
    te = D.d1.dipole.to_numpy() == dp
    i0 = rng.choice(len(D.ids0), 600, replace=False); i1 = rng.choice(np.where(~te)[0], 600, replace=False)
    X = np.vstack([D.X0[i0], D.X1[i1]]); Y = np.vstack([D.Y0[i0], D.Y1[i1]])
    g = np.concatenate([[f"d0:{r}" for r in D.ids0[i0]], ["d1:" + c for c in D.d1.core.to_numpy()[i1]]])
    folds = group_folds(g)
    Xt = D.X1[rng.choice(np.where(te)[0], 300, replace=False)]
    S = fold_scores(X, Y, folds); params, cvm = best_of(S)
    mine = fit_predict(X, Y, Xt, params)
    rows, ok = [], True
    for t in (0, 3):                                           # barrier, elst
        gs = GridSearchCV(T._make_pipe(EST), GRID, cv=folds, scoring="neg_mean_absolute_error", n_jobs=1, error_score="raise")
        gs.fit(X, Y[:, t])
        bp = (gs.best_params_[AK], gs.best_params_[GK])
        d = float(np.abs(gs.best_estimator_.predict(Xt) - mine[:, t]).max())
        r = dict(target=TGT[t], sklearn=bp, fast=params[t], same_params=bp == params[t], max_abs_pred_diff=d,
                 cv_mae_sklearn=float(-gs.best_score_), cv_mae_fast=float(cvm[t]))
        ok &= r["same_params"] and d <= 1e-6 and abs(r["cv_mae_sklearn"] - r["cv_mae_fast"]) <= 1e-9 * max(1, r["cv_mae_fast"])
        rows.append(r)
    rec = dict(passed=bool(ok), rows=rows, fp=full_fp(D, pre), host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"))
    R5.write_json(out, rec)
    print(json.dumps(rec, indent=1, default=str))
    if not ok:
        die("fast_grid differs from sklearn GridSearchCV — Phase D stops", 3)


def need_check(D, pre):
    f = MODEL_ROOT / "check.json"
    rec = json.loads(f.read_text()) if f.is_file() else None
    if not rec or not rec.get("passed") or rec.get("fp") != full_fp(D, pre):
        die("run `d1_models.py check` first (missing, failed or made from other inputs)")


def need_m0(D, pre):
    got = load_cache(MODEL_ROOT / "m0", full_fp(D, pre))
    if not got or not got[0].get("reproduced"):
        die("M0 has not reproduced the rev 5 fits (run `d1_models.py m0`; PREREG §4 STOP if it failed)")


def cmd_m0(D, pre):
    stem = MODEL_ROOT / "m0"
    fp = full_fp(D, pre)
    got = load_cache(stem, fp)
    if got:
        if not got[0].get("reproduced"):
            die("M0 cache: the rev 5 EXT_SEL KRR_rbf fits were NOT reproduced (PREREG §4) — Phase D stays stopped", 3)
        print("m0 cached (reproduced)"); return
    nj = int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    P1, P0, recs, ok = np.empty_like(D.Y1), np.empty_like(D.Y0), {}, True
    for t, tg in enumerate(TGT):
        gs = GridSearchCV(T._make_pipe(EST), GRID, cv=KFold(R5.N_INNER, shuffle=True, random_state=R5.SEED),
                          scoring="neg_mean_absolute_error", n_jobs=nj, error_score="raise")
        gs.fit(D.X0, D.Y0[:, t])
        P1[:, t] = gs.best_estimator_.predict(D.X1); P0[:, t] = gs.best_estimator_.predict(D.X0)
        u = json.loads((REV5_UNITS / f"EXT_SEL__KRR_rbf__{tg}.json").read_text())
        cv = float(-gs.best_score_); tr = float(np.abs(P0[:, t] - D.Y0[:, t]).mean())
        r = dict(best_params=dict(gs.best_params_), rev5_best_params=u["best_params"], cv_mae=cv, rev5_cv_mae=u["cv_mae"],
                 train_mae=tr, rev5_train_mae=u["train"]["mae"], rev5_columns_equal=u["fingerprint"]["columns"] == D.cols,
                 rev5_n_train=u["n_train"])
        r["reproduced"] = bool(r["best_params"] == r["rev5_best_params"] and abs(cv - u["cv_mae"]) <= 1e-6 * u["cv_mae"]
                               and abs(tr - u["train"]["mae"]) <= 1e-6 * u["train"]["mae"] and r["rev5_columns_equal"]
                               and u["n_train"] == len(D.ids0))
        ok &= r["reproduced"]; recs[tg] = r
        print(tg, json.dumps(r, default=str), flush=True)
    save_cache(stem, dict(pred1=P1, pred0=P0), dict(fp=fp, units=recs, reproduced=bool(ok), job=os.environ.get("SLURM_JOB_ID")))
    if not ok:
        die("M0 does not reproduce the rev 5 EXT_SEL KRR_rbf fits (PREREG §4) — Phase D stops", 3)


def split_list(D):
    d = D.d1
    out = [("LOCO", c, d.core.to_numpy() == c) for c in sorted(d.core.unique())]
    out += [("LODO", x, d.dipole.to_numpy() == x) for x in sorted(d.dipole.unique())]
    out += [("LOXO", x, (d.X.to_numpy() == x) & ~d.is_parent.to_numpy()) for x in sorted(set(d.X) - {"H"})]
    return [s for s in out if s[2].any()]


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def cmd_split(D, pre, task, ntask):
    need_check(D, pre); need_m0(D, pre)
    fp = full_fp(D, pre)
    S = split_list(D)
    Z = np.hstack([D.X1[D.ic] - D.X1[D.ip], D.X1[D.ip]])
    DY = D.Y1[D.ic] - D.Y1[D.ip]
    for k in range(task, len(S), ntask):
        scheme, name, te = S[k]
        stem = MODEL_ROOT / "split" / f"{scheme}__{safe(name)}"
        if load_cache(stem, fp):
            print(f"{scheme} {name}: cached", flush=True); continue
        t0 = time.time()
        tr1 = ~te
        Xtr, Ytr = np.vstack([D.X0, D.X1[tr1]]), np.vstack([D.Y0, D.Y1[tr1]])
        S1 = fold_scores(Xtr, Ytr, group_folds(train_groups(D, tr1)))
        p1, cv1 = best_of(S1)
        pr = fit_predict(Xtr, Ytr, np.vstack([D.X0, D.X1]), p1)
        pred0, pred1 = pr[:len(D.X0)], pr[len(D.X0):]
        # M2: pair-difference model on train pairs (child and parent in train)
        mtr = tr1[D.ic] & tr1[D.ip]
        g2 = D.pairs.core.to_numpy()[mtr]
        if len(set(g2)) < 5:
            die(f"{scheme} {name}: {len(set(g2))} train-pair cores < 5 folds")
        S2 = fold_scores(Z[mtr], DY[mtr], group_folds(g2))
        p2, cv2 = best_of(S2)
        dhat = fit_predict(Z[mtr], DY[mtr], Z, p2)
        arrays = dict(test=te, pred0=pred0, pred1=pred1, dhat=dhat, cv1=cv1, cv2=cv2,
                      p1=np.array(p1), p2=np.array(p2), pair_train=mtr)
        # M3 (LOCO): learning curve with the M1 hyper-parameters of this split
        if scheme == "LOCO":
            others = np.array(sorted(set(D.d1.core) - {name}))
            m3 = np.full((M3_SEEDS, len(M3_K), int(te.sum()), len(TGT)), np.nan)
            for s in range(M3_SEEDS):
                perm = np.random.default_rng(SEED + s).permutation(others)
                for j, kk in enumerate(M3_K):
                    sel = np.isin(D.d1.core.to_numpy(), perm[:kk]) & tr1
                    m3[s, j] = fit_predict(np.vstack([D.X0, D.X1[sel]]), np.vstack([D.Y0, D.Y1[sel]]), D.X1[te], p1)
            arrays["m3"] = m3
        rec = dict(fp=fp, scheme=scheme, name=name, n_test=int(te.sum()), n_train=len(Xtr), n_pair_train=int(mtr.sum()),
                   params_m1=[list(p) for p in p1], params_m2=[list(p) for p in p2], seconds=round(time.time() - t0, 1),
                   host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"))
        save_cache(stem, arrays, rec)
        print(f"{scheme} {name}: n_test {rec['n_test']}, {rec['seconds']} s", flush=True)


def cmd_m4(D, pre):
    need_check(D, pre); need_m0(D, pre)
    fp = full_fp(D, pre)
    root = MODEL_ROOT / "m4"; root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").is_file():
        die("M4 result exists — the lockbox was already opened once (PREREG §6); nothing is recomputed", 3)
    keep1 = ~D.ov_lock                                  # D1 rows that are a lockbox reaction stay out (PREREG §1)
    got = load_cache(root / "pred", fp)
    if got is None:
        if (root / "LOCKBOX_OPENED.json").is_file():
            die("LOCKBOX_OPENED.json exists without a prediction cache — inspect by hand", 3)
        X, Y = np.vstack([D.X0, D.X1[keep1]]), np.vstack([D.Y0, D.Y1[keep1]])
        S4 = fold_scores(X, Y, group_folds(train_groups(D, keep1)))
        p4, cv4 = best_of(S4)
        pr = fit_predict(X, Y, np.vstack([D.XL, X]), p4)
        save_cache(root / "pred", dict(predL=pr[:len(D.XL)], pred_train=pr[len(D.XL):], cv4=cv4, p4=np.array(p4)),
                   dict(fp=fp, params=[list(p) for p in p4], lockbox_ids_sha=jhash(D.idsL.tolist()),
                        n_d1_left_out_same_as_lockbox=int((~keep1).sum())))
        got = load_cache(root / "pred", fp)
    rec, arr = got
    opened = root / "LOCKBOX_OPENED.json"
    stamp = dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"))
    if opened.is_file():                                # resume of step (2) only: keep the first opening, log the resume
        o = json.loads(opened.read_text()); o.setdefault("resumed", []).append(stamp)
    else:
        o = dict(opened=stamp, fp=fp, note="PREREG_D1 §6: the single lockbox evaluation")
    R5.write_json(opened, o)
    ml, _ = T.load_ml(feat_path=R5.FEAT_G1, rows_path=str(R5.ROWS5), targets=TGT, geom="g1", ext=True, ext_path=R5.FEAT_EXT)
    Y = ml.set_index(ml.rxn_id.astype(int)).loc[D.idsL, TGT].to_numpy(float)
    del ml
    P = arr["predL"]
    r5 = pd.read_parquet(REV5_LOCK_PRED)
    r5 = r5[(r5.arm == "EXT_SEL") & (r5.model == "KRR_rbf")]
    idx = np.random.default_rng(SEED).integers(0, len(Y), size=(R5.N_BOOT, len(Y)))
    rows, worse = [], []
    for t, tg in enumerate(TGT):
        q = r5[r5.target == tg].set_index("rxn_id").loc[D.idsL]
        if not np.allclose(q.y.to_numpy(float), Y[:, t]):
            die(f"rev 5 lockbox targets differ for {tg}")
        e4, e5 = np.abs(P[:, t] - Y[:, t]), np.abs(q.yhat.to_numpy(float) - Y[:, t])
        d = e4 - e5
        b = d[idx].mean(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        r = dict(target=tg, n=len(Y), mae_m4=float(e4.mean()), mae_rev5_ext_sel=float(e5.mean()), nmae_m4=R5.nmae(Y[:, t], P[:, t]),
                 diff=float(d.mean()), diff_ci_lo=float(lo), diff_ci_hi=float(hi), worse=bool(lo > 0),
                 params=rec["params"][t], train_mae=float(np.abs(arr["pred_train"][:, t] - np.vstack([D.Y0, D.Y1[keep1]])[:, t]).mean()),
                 n_d1_left_out_same_as_lockbox=int((~keep1).sum()))
        rows.append(r); worse += [tg] if r["worse"] else []
    res = dict(fp=fp, rows=rows, worse_targets=worse, stop=bool(worse), n_boot=R5.N_BOOT, seed=SEED)
    R5.write_json(root / "result.json", res)
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "M4_lockbox.csv", index=False)
    print(json.dumps(res, indent=1, default=str))
    if worse:
        die(f"M4: lockbox MAE worse than rev 5 EXT_SEL (CI > 0) for {worse} — report and stop (PREREG §5.2)", 3)


# ================================================================ report
def pair_metrics(dd_hat, dd, ctx):
    out = dict(n=len(dd), mae=float(np.abs(dd_hat - dd).mean()) if len(dd) else np.nan,
               sign_agree=float((np.sign(dd_hat) == np.sign(dd)).mean()) if len(dd) else np.nan)
    # Spearman inside each context with >= MIN_CTX pairs = Pearson of the within-context average ranks (scipy spearmanr
    # definition, ties averaged), vectorised over contexts; a context with no spread in either variable is skipped
    rho = np.array([])
    if len(dd):
        d = pd.DataFrame(dict(c=ctx, h=dd_hat, y=dd))
        d = d[d.groupby("c").c.transform("size") >= MIN_CTX]
        if len(d):
            g = d.groupby("c")
            rh, ry = g.h.rank(), g.y.rank()
            xh, xr = rh - rh.groupby(d.c).transform("mean"), ry - ry.groupby(d.c).transform("mean")
            s = pd.DataFrame(dict(c=d.c, xy=xh * xr, xx=xh * xh, yy=xr * xr)).groupby("c").sum()
            s = s[(s.xx > 1e-12) & (s.yy > 1e-12)]
            rho = (s.xy / np.sqrt(s.xx * s.yy)).to_numpy()
    out.update(ctx_spearman_median=float(np.median(rho)) if len(rho) else np.nan, n_ctx=len(rho))
    return out


def sel_acc(a, b, yhat, y):
    if not len(a):
        return np.nan
    return float((np.sign(yhat[a] - yhat[b]) == np.sign(y[a] - y[b])).mean())


def metric_rows(D, model, scheme, part, pred1, rows_m, pairs_m, dd_hat, sib_ok=None):
    """Metric rows for one (model, scheme, part): carbonyl ylide incl / excl x dipole (all + each) x target.
    Absolute (MAE, NMAE) on rows_m, pair metrics on pairs_m (ΔΔ̂ = dd_hat), selectivity (ΔE‡) on siblings in rows_m."""
    out = []
    dip, pdip = D.d1.dipole.to_numpy(), D.pairs.dipole.to_numpy()
    ctx = D.pairs.ctx.to_numpy()
    for cyv in ("incl", "excl"):
        for dsel in ["all"] + sorted(D.d1.dipole.unique()):
            if cyv == "excl" and dsel == CY:
                continue
            keep = rows_m & ((dip != CY) if cyv == "excl" else True) & ((dip == dsel) if dsel != "all" else True)
            pk = pairs_m & ((pdip != CY) if cyv == "excl" else True) & ((pdip == dsel) if dsel != "all" else True)
            for t, tg in enumerate(TGT):
                y, p = D.Y1[keep, t], pred1[keep, t]
                dd = D.Y1[D.ic[pk], t] - D.Y1[D.ip[pk], t]
                pm = pair_metrics(dd_hat[pk, t], dd, ctx[pk])
                r = dict(model=model, scheme=scheme, part=part, carbonyl_ylide=cyv, dipole=dsel, target=tg, n=int(keep.sum()),
                         mae=float(np.abs(y - p).mean()) if len(y) else np.nan, nmae=R5.nmae(y, p) if len(y) > 1 else np.nan,
                         pair_n=pm["n"], pair_mae=pm["mae"], pair_sign=pm["sign_agree"], pair_ctx_rho=pm["ctx_spearman_median"],
                         n_ctx=pm["n_ctx"])
                if t == 0:
                    for k, (a, b, _) in D.sibs.items():
                        m = keep[a] & keep[b] & (sib_ok[a] & sib_ok[b] if sib_ok is not None else True)
                        r[f"sel_{k}_acc"], r[f"sel_{k}_n"] = sel_acc(a[m], b[m], pred1[:, 0], D.Y1[:, 0]), int(m.sum())
                out.append(r)
    return out


KEYS = ["model", "scheme", "part", "carbonyl_ylide", "dipole", "target"]


def m2_rows(D, a, rows_m, pairs_m):
    """ŷ₂ for the rows of one split: children of pairs_m = ŷ₁(parent) + Δ̂, parents in rows_m = ŷ₁; NaN elsewhere."""
    P2 = np.full_like(D.Y1, np.nan)
    P2[D.ic[pairs_m]] = a["pred1"][D.ip[pairs_m]] + a["dhat"][pairs_m]
    par = rows_m & D.d1.is_parent.to_numpy()
    P2[par] = a["pred1"][par]
    return P2


def cmd_report(D, pre):
    fp = full_fp(D, pre)
    RES.mkdir(exist_ok=True)
    rows, boot, m3rows = [], [], []
    allp = np.ones(len(D.pairs), bool)
    m0 = load_cache(MODEL_ROOT / "m0", fp)
    if m0:
        rec, a = m0
        rows += metric_rows(D, "M0", "D0dev->D1", "test", a["pred1"], np.ones(len(D.d1), bool), allp, a["pred1"][D.ic] - a["pred1"][D.ip])
        for t, tg in enumerate(TGT):
            rows.append(dict(model="M0", scheme="D0dev->D1", part="train (D0 dev, in-sample)", carbonyl_ylide="-", dipole="-",
                             target=tg, n=len(D.Y0), mae=float(np.abs(a["pred0"][:, t] - D.Y0[:, t]).mean()),
                             nmae=R5.nmae(D.Y0[:, t], a["pred0"][:, t])))
    for scheme in ("LOCO", "LODO", "LOXO"):
        rng = np.random.default_rng(BOOT_SEED)                  # one stream per scheme: CIs do not depend on other schemes
        mine = [(n, te) for s, n, te in split_list(D) if s == scheme]
        caches = [(c[0], c[1], te) for n, te in mine
                  if (c := load_cache(MODEL_ROOT / "split" / f"{scheme}__{safe(n)}", fp)) is not None]
        if len(caches) != len(mine):
            print(f"{scheme}: {len(caches)}/{len(mine)} splits cached — skipped in the report", flush=True); continue
        P1 = np.full_like(D.Y1, np.nan); P2 = np.full_like(D.Y1, np.nan)
        DD1 = np.full((len(D.pairs), len(TGT)), np.nan); DD2 = DD1.copy()
        tested = np.zeros(len(D.d1), bool); ptest = np.zeros(len(D.pairs), bool)
        tr_rows = []
        for rec, a, te in caches:
            P1[te] = a["pred1"][te]; tested |= te
            pt = te[D.ic]                                   # test pairs: child in the test group (parent: same model)
            dd1 = a["pred1"][D.ic] - a["pred1"][D.ip]
            DD1[pt] = dd1[pt]; DD2[pt] = a["dhat"][pt]; ptest |= pt
            p2 = m2_rows(D, a, te, pt); P2[te] = p2[te]
            # train part, in-sample on this split's training side (D1 train rows, train pairs), averaged over splits below
            tr, mtr = ~te, a["pair_train"]
            p2t = m2_rows(D, a, tr, mtr); ok2t = ~np.isnan(p2t[:, 0])
            tr_rows += metric_rows(D, "M1", scheme, "train", a["pred1"], tr, mtr, dd1)
            tr_rows += metric_rows(D, "M2", scheme, "train", np.nan_to_num(p2t), tr & ok2t, mtr, a["dhat"], sib_ok=ok2t)
            for t, tg in enumerate(TGT):
                tr_rows.append(dict(model="M1", scheme=scheme, part="train (D0 dev, in-sample)", carbonyl_ylide="-", dipole="-",
                                    target=tg, n=len(D.Y0), mae=float(np.abs(a["pred0"][:, t] - D.Y0[:, t]).mean()),
                                    nmae=R5.nmae(D.Y0[:, t], a["pred0"][:, t])))
            if scheme == "LOCO" and "m3" in a:
                tei = np.where(te)[0]; loc = pd.Series(np.arange(len(tei)), index=tei)
                pp = pt & np.isin(D.ip, tei)
                dd = D.Y1[D.ic[pp]] - D.Y1[D.ip[pp]]
                for s in range(M3_SEEDS):
                    for j, kk in enumerate(M3_K):
                        pk = a["m3"][s, j]
                        dd_hat = pk[loc.loc[D.ic[pp]].to_numpy()] - pk[loc.loc[D.ip[pp]].to_numpy()]
                        m3rows.append(dict(core=rec["name"], seed=s, k=kk, n=len(tei), abs_err_sum=np.abs(pk - D.Y1[te]).sum(axis=0).tolist(),
                                           pair_err_sum=np.abs(dd_hat - dd).sum(axis=0).tolist(), n_pairs=int(pp.sum())))
                    m3rows.append(dict(core=rec["name"], seed=s, k="all (M1)", n=len(tei), abs_err_sum=np.abs(a["pred1"][te] - D.Y1[te]).sum(axis=0).tolist(),
                                       pair_err_sum=np.abs(dd1[pp] - dd).sum(axis=0).tolist(), n_pairs=int(pp.sum())))
        ok2 = ~np.isnan(P2[:, 0])
        rows += metric_rows(D, "M1", scheme, "test", P1, tested, ptest, np.nan_to_num(DD1))
        rows += metric_rows(D, "M1 (M2 rows)", scheme, "test", P1, tested & ok2, ptest, np.nan_to_num(DD1), sib_ok=ok2)
        rows += metric_rows(D, "M2", scheme, "test", np.nan_to_num(P2), tested & ok2, ptest, np.nan_to_num(DD2), sib_ok=ok2)
        Tr = pd.DataFrame(tr_rows)
        Tr = Tr.groupby(KEYS, sort=False).mean(numeric_only=True).reset_index()
        Tr["part"] = Tr["part"].where(Tr["part"] != "train", "train (in-sample, mean over splits)")
        rows += Tr.to_dict("records")
        # M2 vs M1: core bootstrap of the pair-MAE difference (PREREG §5.1)
        cores = D.pairs.core.to_numpy()[ptest]
        uc = np.unique(cores); byc = {c: np.where(cores == c)[0] for c in uc}
        ddt = D.Y1[D.ic[ptest]] - D.Y1[D.ip[ptest]]
        e1, e2 = np.abs(DD1[ptest] - ddt), np.abs(DD2[ptest] - ddt)
        bs = []
        for _ in range(N_BOOT_CORE):
            ii = np.concatenate([byc[c] for c in rng.choice(uc, len(uc))])
            bs.append(e2[ii].mean(axis=0) - e1[ii].mean(axis=0))
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        for t, tg in enumerate(TGT):
            boot.append(dict(scheme=scheme, target=tg, n_pairs=int(ptest.sum()), n_cores=len(uc), pair_mae_m1=float(e1[:, t].mean()),
                             pair_mae_m2=float(e2[:, t].mean()), diff=float(e2[:, t].mean() - e1[:, t].mean()), ci_lo=float(lo[t]),
                             ci_hi=float(hi[t]), m2_better=bool(hi[t] < 0)))
    if not rows:
        die("nothing cached yet (m0 / split) — no report")
    M = pd.DataFrame(rows); M.to_csv(RES / "metrics.csv", index=False)
    Bt = pd.DataFrame(boot); Bt.to_csv(RES / "M2_vs_M1_bootstrap.csv", index=False)
    lc = None
    if m3rows:
        L3 = pd.DataFrame(m3rows)
        agg = []
        for (s, k), g in L3.groupby(["seed", "k"], sort=False):
            ae = np.sum(np.array(g.abs_err_sum.tolist()), axis=0) / g.n.sum()
            pe = np.sum(np.array(g.pair_err_sum.tolist()), axis=0) / max(1, g.n_pairs.sum())
            agg.append(dict(k=str(k), **{f"mae_{tg}": ae[t] for t, tg in enumerate(TGT)}, **{f"pair_mae_{tg}": pe[t] for t, tg in enumerate(TGT)}))
        lc = pd.DataFrame(agg).groupby("k", sort=False).agg(["mean", "std"])
        lc.to_csv(RES / "M3_learning_curve.csv")
    m4 = MODEL_ROOT / "m4" / "result.json"
    write_md(D, pre, M, Bt, lc, json.loads(m4.read_text()) if m4.is_file() else None, m0[0] if m0 else None)


def write_md(D, pre, M, Bt, lc, m4, m0rec):
    def tab(df, cols):
        if df is None or not len(df):
            return ["(없음)"]
        d = df[[c for c in cols if c in df.columns]]
        f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v))
        return ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    L = ["# MODEL_REPORT — D1 Phase D (PREREG_D1.md)", "",
         f"- PREREG sha256 `{pre['prereg_sha256'][:16]}…`, 승인 {pre['approved']}, HEAD `{pre['git_head'][:10]}`",
         f"- D0 dev {len(D.ids0)}행, D1 {len(D.d1)}행 (전체 {D.n_d1_all} 중 q_bad · EXT_SEL 결측 제외), 짝 {len(D.pairs)}, rev5 위생 필터에 걸리는 D1 행 {D.hygiene_would_drop} (적용 안 함)",
         f"- D0 와 같은 반응인 D1 행: lockbox {int(D.ov_lock.sum())} (입체 포함 {D.ov_lock_iso}; M4 학습에서 뺌), dev {D.ov_dev} (입체 포함 {D.ov_dev_iso}; 보고만)",
         "- 다른 점: LOXO 에서 H 부모는 train 쪽 (부모 예측 = 학습 내 예측). M3 는 그 LOCO 분할의 M1 하이퍼파라미터 고정.",
         "- 'M1 (M2 rows)' = M1 을 M2 가 예측한 행(부모가 있는 자식 + 부모)에서만 잰 값 — M2 와 같은 행끼리 비교용.",
         f"- 전체 표 (dipole 별 포함): `results_models/metrics.csv`", ""]
    if m0rec:
        L += ["## M0 재현 확인 (rev5 EXT_SEL KRR_rbf 단위 기록)", "", f"- 9 타깃 모두 재현: {m0rec['reproduced']}", ""]
    cols = ["model", "scheme", "part", "carbonyl_ylide", "target", "n", "mae", "nmae", "pair_n", "pair_mae", "pair_sign", "pair_ctx_rho",
            "sel_regio_acc", "sel_regio_n", "sel_exoendo_acc", "sel_exoendo_n"]
    overall = M.dipole.isin(["all", "-"])
    for mdl in ("M0", "M1", "M1 (M2 rows)", "M2"):
        L += [f"## {mdl}", "", *tab(M[(M.model == mdl) & overall], cols), ""]
    byd = M[(M.target == "dft_barrier_kcal") & (M.part == "test") & (M.carbonyl_ylide == "incl") & ~overall]
    L += ["## dipole 별 (ΔE‡, test, carbonyl ylide 포함)", "",
          *tab(byd, ["model", "scheme", "dipole", "n", "mae", "nmae", "pair_n", "pair_mae", "pair_sign", "pair_ctx_rho", "sel_regio_acc", "sel_exoendo_acc"]), ""]
    L += ["## M2 vs M1: 짝 MAE 차이 (코어 부트스트랩 95 % CI; 주 판정 LOCO · ΔE‡)", "",
          *tab(Bt, ["scheme", "target", "n_pairs", "n_cores", "pair_mae_m1", "pair_mae_m2", "diff", "ci_lo", "ci_hi", "m2_better"]), ""]
    if len(Bt):
        pr = Bt[(Bt.scheme == "LOCO") & (Bt.target == "dft_barrier_kcal")]
        L += [f"- 주 판정: {'M2 채택' if len(pr) and bool(pr.m2_better.iloc[0]) else 'M2 채택 아님'}", ""]
    L += ["## M3 학습 곡선 (LOCO, seed 5개 평균 ± sd)", "", "```", lc.round(3).to_string() if lc is not None else "(없음)", "```", ""]
    L += ["## M4 lockbox (1회)", ""]
    if m4:
        L += tab(pd.DataFrame(m4["rows"]), ["target", "n", "mae_m4", "mae_rev5_ext_sel", "diff", "diff_ci_lo", "diff_ci_hi", "worse", "train_mae"])
        L += ["", f"- 악화(CI 하한 > 0): {m4['worse_targets'] or '없음'} → {'STOP' if m4['stop'] else '통과'}", ""]
    else:
        L += ["(M4 미실행)", ""]
    out = HERE / "MODEL_REPORT.md"
    R5.write_atomic(out, "\n".join(L) + "\n")
    print(f"wrote {out}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("check", "m0", "split", "m4", "report"):
        raise SystemExit(__doc__)
    pre = gate_prereg()
    D = load_data(keep_lockbox=cmd == "m4")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if cmd == "check":
        cmd_check(D, pre)
    elif cmd == "m0":
        cmd_m0(D, pre)
    elif cmd == "split":
        cmd_split(D, pre, int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "m4":
        cmd_m4(D, pre)
    else:
        cmd_report(D, pre)


if __name__ == "__main__":
    main()
