#!/usr/bin/env python3
"""espley_rev5.py — rev 5 Phase D-3(a), D-3(b), D-4 (docs/specs/REV5_FEATURES_FIGURES.md): our xTB models vs Espley
et al. 2024 on their ds3 reactions. Run by r5_espley.sh inside one SLURM allocation; refuses to run outside SLURM.

  python espley_rev5.py d3a|d3b|d4|all [--gates-only]

Common inputs (compare_espley helpers and the rev 4 Espley-side files; none of them is modified)
  Espley rows    compare_espley.load_esp() = manual_tt_solvent.pkl in its row order ("ds3 row order"; reaction_number
                 = our rxn_id; whether it is ascending is recorded as inputs.ds3_sorted_by_rxn_id); ids, label mask
                 and d1/d2 swap vector from $R4_SCRATCH/compare/espley/prep.pkl
                 (compare_espley.load_common; its labels sha256 must be the one of labels_all.json now)
  targets (5)    role dipole / dipolarophile distortion (Espley's distortion_energy_1/2 re-assigned by the swap vector,
                 compare_espley.target_values) + interaction_energies_dft, e_barrier_dft, q_barrier_dft as Espley
                 stores them (interaction: positive = stabilising, the opposite sign of our eint_spe). Every side is
                 trained and scored on these numbers.
  our features   train_ml_single.load_ml(G1 parquet + blocks B1..B6, rows = results_rev5/rows_rev5.csv, every row
                 gated), arms ESPLEY46 / ESPLEY73 / EXT_SEL (train_ml_single.resolve_arm; EXT_SEL = prereg_rev5b.json),
                 model KRR_rbf (train_ml_single.GRIDS / _make_pipe)
  Espley feat.   compare_espley.espley_X: the 46 ML columns (DROP_ML); tuning columns TUNE_DROPS[0] (47, their
                 hyp_tuning.py); on the role targets both AM1 distortions are re-assigned by the swap vector (as rev 4)
  row set        labelled (prep have_lab) AND in rows_rev5.csv (which implies G1-usable: xtb ok, NaN-free). Asserted
                 to be a subset of the rev 4 scored rows (labelled AND G1-usable, recomputed from the G1 parquet, =
                 compare_espley.N_SCORED); the rev 4 rows left out are listed with the reason (D3a.json)
Gates before any data is read, for every stage (all three use lockbox rows: D-3(a) / D-4 through the row set, D-3(b)
as test), in the order of final_eval.py (D-1): final_eval.gate_prereg (PREREG_REV5b.md, prereg_rev5b.json,
rows_rev5.csv, dev_ids.csv, lockbox_ids.csv tracked and identical to HEAD, lockbox sha256 written in a committed
PREREG_REV5* file); final_eval.gate_partition (dev ∩ lockbox = ∅, dev ∪ lockbox = rows_rev5, lockbox size);
final_eval.audit_select (first use of the lockbox: the Phase C caches under $R5_SCRATCH/select hold exactly the dev
ids, no lockbox id), so the lockbox is audited whichever Phase D job runs first. Recorded in every output JSON.

D-3(a) Espley split (compare_espley.espley_split: test = the FIRST half of the 20 % hold-out), seeds PROTO_A_SEEDS.
  Every side is scored on exactly the row-set rows of each seed's test split (gated), against the same y (gated).
  Ours     ESPLEY46 (re-run of the rev 4 pre-registered main comparison), ESPLEY73, EXT_SEL; KRR only;
           compare_espley.our_pipeline (per-seed GridSearchCV, KFold(5, shuffle, rs = seed), X and y scaled) on the
           row-set rows of the seed's training split
  Espley   stored test predictions of their 5 models (Ridge, KRR, SVR, 2-/4-layer NN) for interaction / ΔE‡ / ΔG‡
           (rev 4 espley_stored_preds.parquet; trained by Espley on their whole training split);
           role targets: their-protocol SVR and KRR (compare_espley.their_protocol, trained on their labelled rows),
           reused from rev 4 role_theirs_preds.parquet + role_theirs.json when these cover every scored test row with
           the same y, else recomputed here (units);
           their 46 AM1 features x our pipeline (Ridge, KRR, SVR, XGB; compare_espley.our_pipeline on their rows with
           a target = all rows for interaction / ΔE‡ / ΔG‡, labelled rows for the role targets, as rev 4 role_ours)
           for all 5 targets; the role targets are cross-checked against rev 4 role_ours_preds.parquet (reported)
  "Espley SVR" = stored SVR (interaction / ΔE‡ / ΔG‡) or their-protocol SVR (role). "Espley strongest" = the
  Espley-side model with the lowest seed-mean MAE on the scored rows, per target (a best-of on test: in Espley's favour).
  Test: per seed d = MAE(Espley) − MAE(ours); rev5_common.nadeau_bengio(d, mean n_test, mean n_train of ours) ->
  mean, t, two-sided p, 95 % CI, for every our arm vs (i) Espley SVR and (ii) Espley strongest.
D-3(b) test = lockbox ∩ row set, train = dev ∩ row set. Espley: their protocol on this train — THEIR_TUNE grid,
  GridSearchCV(cv=5 = unshuffled KFold, error_score as theirs, neg MAE) on StandardScaler(X_train) of the TUNE_DROPS[0]
  columns, y raw, with the train rows in the order of their hyp_tuning.py X_train (rank in train_test_split(range(n),
  test_size=0.2, random_state=TUNE_SEED): its training part, then its hold-out part), so the unshuffled folds are random
  as in their code (ds3 row order would give contiguous folds); then THEIR_ML (KRR tuned with the poly
  kernel and run with rbf, as compare_espley.their_protocol) with the tuned params on StandardScaler(the 46 ML columns
  of the train rows), y raw. SVR (primary) and KRR. Ours: EXT_SEL KRR, GridSearchCV(_make_pipe, GRIDS['KRR_rbf'],
  KFold(N_INNER, shuffle, rs SEED)) on the train rows in ds3 row order.
  Paired reaction-level bootstrap: ONE index matrix default_rng(SEED).integers(0, n_test, (N_BOOT, n_test)) over the
  test rows in ds3 row order, shared by every target and side; 95 % percentile CI of each MAE and of
  MAE(Espley) − MAE(ours).
D-4 learning curves on the D-3(a) splits and seeds: training frame of seed s = the row-set rows of
  espley_split(n, s)[0] in that order (the order of the X_train frame their ml_analysis.py samples from), indexed by
  rxn_id; subset = frame.sample(frac=f, random_state=s) for f in LC_FRACS, the same subset for both sides; test rows =
  the D-3(a) scored rows of seed s, fixed. Sensitivity (reported separately, D4_learning_curves_sensitivity.csv): the
  same with the frame in ds3 row order. The primary order (LC_PRIMARY) is a design choice fixed in this file before any
  D-4 result exists; every output JSON records this file's sha256 and whether it was committed and unchanged at run
  time (code_file). Hyperparameters fixed to the full-train tuned values:
    Espley SVR (THEIR_ML): hps.pkl b_hps (interaction / ΔE‡ / ΔG‡, as their code), the their-protocol params of D-3(a)
      (role); StandardScaler fit on the whole training frame, then subsets of the scaled rows (as ml_analysis.py)
    EXT_SEL KRR: the per-seed best params of the D-3(a) EXT_SEL units; _make_pipe refit on the subset
  Check (reported, not a gate): their ml_analysis.py learning curve re-run on all their rows (SVR, hps.pkl, fracs
  0.1..0.9, seeds of ml_results.pkl) vs the '<frac>_y_test_pred_values' stored in ml_results.pkl.
  Gate: EXT_SEL at frac 1.0 reproduces the D-3(a) EXT_SEL per-seed MAE within FRAC1_TOL (both frame orders).

Parallelism: GridSearchCV fits run one unit at a time with n_jobs = R5_NJOBS (default SLURM_CPUS_PER_TASK) workers
(compare_espley.N_JOBS is set to the same); the D-4 fits run in a spawn ProcessPoolExecutor of that size. One BLAS
thread per process (OMP/OPENBLAS/MKL_NUM_THREADS default 1).
Idempotent: one cache per unit under $R5_SCRATCH/espley/{d3a,d3b,d4}/units (parquet, then JSON = done marker; D-4:
JSON only; atomic). A finished unit is reused; one made from other inputs (fingerprint: data sha256, columns, grid,
seed, params, package versions) stops the run with exit 3: move it away. Reports are rebuilt from the caches each run.
Outputs (atomic)
  results_rev5/D3a_rows.csv, D3a_per_seed.csv, D3a_summary.csv, D3a_tests.csv, D3a_predictions.csv (per-row
    predictions of the four figure series Espley / EXT_SEL / ESPLEY73 / ESPLEY46 on the scored rows), D3a.json
  results_rev5/D3b_summary.csv, D3b_paired.csv, D3b_predictions.csv, D3b.json
  results_rev5/D4_learning_curves.csv (target, side, frac, n_train, seed, mae), D4_learning_curves_summary.csv,
    D4_learning_curves_reach.csv, D4_learning_curves_sensitivity.csv (ds3 row order), D4_learning_curves.json
  $R5_SCRATCH/espley/D3a_predictions.parquet, D3b_predictions.parquet: per-row predictions of every side
exit 0 ok; 1 a unit / check failed after fitting started; 2 missing / invalid input or gate (including the
train_ml_single / compare_espley gates met while loading the data); 3 a cache made from other inputs (move it away);
4 LOCKBOX LEAK, a lockbox id in the Phase C caches (final_eval.audit_select; final_eval.py itself exits 3 for it,
remapped here so that 3 keeps one meaning).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before numpy: one BLAS thread each
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev5_common as R5  # noqa: E402
import train_ml_single as T  # noqa: E402
import compare_espley as CE  # noqa: E402
import final_eval as FE  # noqa: E402

# ------------------------------------------------------------------------------------------ paths
OUT = R5.SCRATCH / "espley"
UNITS = {"d3a": OUT / "d3a" / "units", "d3b": OUT / "d3b" / "units", "d4": OUT / "d4" / "units"}
LC_REP = OUT / "d4" / "their_lc_replication.json"
PREDS_A, PREDS_B = OUT / "D3a_predictions.parquet", OUT / "D3b_predictions.parquet"
REV4 = CE.OUT / "espley"                                     # rev 4 Espley-side outputs (read only)
PREP = REV4 / "prep.pkl"
STORED = REV4 / "espley_stored_preds.parquet"
ROLE_THEIRS, ROLE_THEIRS_JSON = REV4 / "role_theirs_preds.parquet", REV4 / "role_theirs.json"
ROLE_OURS = REV4 / "role_ours_preds.parquet"
REV4_PER_SEED = R5.RES4 / "espley_compare_per_seed.csv"      # rev 4 reference numbers (other rows; reported only)
RES = R5.RES5

# ------------------------------------------------------------------------------------------ design
ROLE = list(CE.ROLE_TARGETS)                                 # dipole_distortion_role, dipolarophile_distortion_role
RAW3 = ["interaction_energies_dft", "e_barrier_dft", "q_barrier_dft"]
TARGETS5 = ROLE + RAW3
assert all(t in CE.TARGETS for t in RAW3), RAW3
LABEL = {**{t: CE.ROLE_TARGETS[t][0] for t in ROLE}, **{t: CE.TARGETS[t][0] for t in RAW3}}
SEEDS = [int(s) for s in R5.PROTO_A_SEEDS]
OUR_ARMS = ("ESPLEY46", "ESPLEY73", "EXT_SEL")               # ESPLEY73 before EXT_SEL: EXT_SEL = ESPLEY73 is copied
OUR_MODEL = "KRR_rbf"
MODEL_NAME = {m: m.replace("_rbf", "") for m in T.GRIDS}     # Ridge, KRR, SVR, XGB
ESP, OURS, ESP_ARM = "Espley", "Ours", "AM1-46"
P_STORED, P_THEIRS, P_OURS = "stored", "their_protocol", "our_pipeline"
KEYS = ["side", "protocol", "arm", "model"]
SERIES = list(R5.COLORS)                                     # figure series = colour keys
assert set(SERIES) == {ESP, *OUR_ARMS}, SERIES
SCORING = "neg_mean_absolute_error"
Y_TOL = 1e-6                                                 # max |Δy| between sides / vs the target vector
FRAC1_TOL = 1e-2                                             # D-4 EXT_SEL frac 1.0 vs D-3(a), kcal/mol per seed
LC_REP_TOL = 1e-6                                            # their learning-curve re-run vs ml_results.pkl
# D-4 training frame per seed (rxn_id-indexed, then DataFrame.sample, which draws positions, so the order changes the
# subset): "split" = their ml_analysis.py X_train order (espley_split(n, seed)[0], i.e. the train_test_split order,
# which their :400-401 X_train.sample draws from); "ds3" = ds3 row order (manual_tt_solvent.pkl). Both are computed.
# LC_PRIMARY picks the one reported as D4_learning_curves.csv (the other -> D4_learning_curves_sensitivity.csv). It is
# a pre-result design choice: fix it here and commit this file before d4 runs (the outputs record code_file).
LC_ORDERS = {"split": "row-set rows of espley_split(n, seed)[0] in that order (the X_train order their "
                      "ml_analysis.py samples from), indexed by rxn_id",
             "ds3": "row-set rows of espley_split(n, seed)[0] in ds3 row order (manual_tt_solvent.pkl), indexed "
                    "by rxn_id"}
LC_PRIMARY = "split"
assert LC_PRIMARY in LC_ORDERS, LC_PRIMARY
NJOBS = 1                                                    # set in main()

NOTES_A = [
    "Targets are Espley's DFT numbers (Gaussian B3LYP-D3(BJ)/def2-TZVP SMD(water)); interaction_energies_dft keeps "
    "Espley's sign (positive = stabilising, the opposite of our eint_spe).",
    "Role targets: Espley's distortion_energy_1/2 re-assigned to dipole / dipolarophile by the rev 4 swap vector; their "
    "AM1 distortion features re-assigned by the same vector.",
    "Every side is scored on the same row-set test rows of each seed; ours train on the row-set rows of the training "
    "split only, while Espley's stored models were trained on every ds3 row of their training split and the "
    "Espley-side retrains on every row of it with a target (more training rows than ours).",
    "EXT_SEL: its blocks were selected in Phase C on dev rows, which overlap these test rows; its selection-unbiased "
    "Espley comparison is D-3(b) (lockbox).",
    "'Espley strongest' is chosen on these test rows (a best-of in Espley's favour).",
    "Nadeau-Bengio: J = 5 seeds, n_test = mean scored test rows per seed, n_train = mean training rows of ours.",
]


# ------------------------------------------------------------------------------------------ helpers
def die(msg, code=2):
    FE.die(msg, code)


def log(msg):
    print(msg, flush=True)


def _native(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return str(x)


def jsonable(x):
    return json.loads(json.dumps(x, default=_native))


def data_sha(*arrays):
    """sha256 over shapes and little-endian bytes (float -> <f8, int -> <i8) of the arrays."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        a = a.astype("<f8") if a.dtype.kind == "f" else a.astype("<i8")
        h.update(repr(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def pred_frame(D, rows, y, p):
    rows = np.asarray(rows, dtype=np.int64)
    p = np.asarray(p, dtype=float).ravel()
    if len(p) != len(rows) or not np.isfinite(p).all():
        raise ValueError(f"{len(p)} predictions for {len(rows)} rows, finite: {bool(np.isfinite(p).all())}")
    return pd.DataFrame(dict(row=rows, rxn_id=D["ids"][rows], y=np.asarray(y, float)[rows], yhat=p))


def run_unit(stage, stem, fp, fit, allow_fit=True):
    """(record, predictions, how) of one cached unit: reused when its JSON + parquet exist with the same fingerprint
    (exit 3 if they were made from other inputs), else fit() -> (record, predictions), saved atomically (JSON last)."""
    d = UNITS[stage]
    pp, pj = d / f"{stem}.parquet", d / f"{stem}.json"
    fp = jsonable(fp)
    if pj.exists() and pp.exists():
        rec = json.loads(pj.read_text())
        old = rec.get("fingerprint") or {}
        if old != fp:
            diff = sorted(k for k in set(fp) | set(old) if old.get(k) != fp.get(k))
            die(f"{pj} exists but was made with other inputs (keys {diff}) — move {d} away to recompute", 3)
        return rec, pd.read_parquet(pp), "cached"
    if not allow_fit:
        die(f"{pj} missing: run `sbatch ... r5_espley.sh {stage}` first")
    t0 = time.time()
    rec, preds = fit()
    rec = jsonable(dict(rec, fingerprint=fp, seconds=round(time.time() - t0, 1), host=socket.gethostname(),
                        job=os.environ.get("SLURM_JOB_ID", "local"), finished=FE.now()))
    R5.write_atomic(pp, lambda f: preds.to_parquet(f, index=False))
    R5.write_json(pj, rec)                                   # JSON last: its presence marks a finished unit
    return rec, preds, f"fitted in {rec['seconds']:.0f} s"


def pool_map(fn, tasks, what):
    """Yield (key, fn(*args)) for every (key, args) of tasks, as they finish, from a spawn ProcessPoolExecutor of
    NJOBS workers (one BLAS thread each); a failing task stops the run (exit 1)."""
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=max(1, min(NJOBS, len(tasks))), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *args): key for key, args in tasks}
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception as e:                                           # noqa: BLE001
                for g in futs:
                    g.cancel()
                die(f"{what} {futs[f]} failed: {type(e).__name__}: {e}", 1)
            yield futs[f], res


def write_csv(path, df):
    R5.write_atomic(path, lambda f: df.to_csv(f, index=False))


def order_frame(df, cols=("target",)):
    """Stable order: TARGETS5, figure series, seeds."""
    key = {"target": TARGETS5, "series": SERIES + [""], "seed": SEEDS}
    out = df.copy()
    tmp = []
    for c in cols:
        out[f"_o_{c}"] = out[c].map({v: i for i, v in enumerate(key[c])}).fillna(len(key[c]))
        tmp.append(f"_o_{c}")
    return out.sort_values(tmp, kind="mergesort").drop(columns=tmp).reset_index(drop=True)


def xkind(t):
    return "role" if t in ROLE else "raw"


# ------------------------------------------------------------------------------------------ data
def load_data():
    """Espley frame + prep, the row set, targets, our and Espley feature matrices (rows in Espley order)."""
    for f in (PREP, CE.ESP_DS):
        if not f.exists():
            die(f"{f} missing (rev 4 compare_espley prep / Espley data)")
    C, esp = CE.load_common(), CE.load_esp()
    ids = np.asarray(C["ids"]).astype(np.int64)
    n = int(C["n"])
    if len(esp) != n or len(ids) != n or not np.array_equal(esp["reaction_number"].astype(np.int64).to_numpy(), ids):
        die(f"{PREP} ids differ from {CE.ESP_DS} reaction_number (their order)")
    lab_sha = R5.sha256(R5.LABELS)
    if C.get("labels_sha256") != lab_sha:
        die(f"{PREP} was made from labels sha256 {C.get('labels_sha256')}, {R5.LABELS} is {lab_sha} now")
    have_lab = np.asarray(C["have_lab"], dtype=bool)
    sw = np.asarray(C["swap"], dtype=float) == 1             # NaN (no label) -> not swapped; its role target is NaN

    # our rows and features: exactly rows_rev5.csv, gated by load_ml (ok, NaN-free, hygiene; nothing dropped)
    rows5, rows4 = FE.read_ids(R5.ROWS5), FE.read_ids(R5.ROWS4)
    ml, rows_info = T.load_ml(feat_path=R5.FEAT_G1, rows_path=str(R5.ROWS5), targets=R5.TARGETS9, geom="g1",
                              ext=True, ext_path=R5.FEAT_EXT)
    rid = ml["rxn_id"].astype(np.int64).to_numpy()
    if rid.tolist() != rows5:
        die("load_ml returned rows in another order than rows_rev5.csv")
    cols = {a: list(T.resolve_arm(a)) for a in OUR_ARMS}
    absent = sorted({c for a in OUR_ARMS for c in cols[a]} - set(ml.columns))
    if absent:
        die(f"arm columns absent from the ML rows: {absent[:10]}")
    F = ml.set_index(pd.Index(rid, name="rxn_id"))
    in5 = np.isin(ids, rid)
    usable = have_lab & in5

    # rev 4 scored rows (labelled AND G1-usable = xtb ok, NaN-free ESPLEY46/73), recomputed from the G1 parquet
    f4673 = sorted(set(T.FEATURE_SETS["ESPLEY46"]) | set(T.FEATURE_SETS["ESPLEY73"]))
    g1 = pd.read_parquet(R5.FEAT_G1, columns=["rxn_id", "xtb_status"] + f4673)
    g1 = g1.assign(rxn_id=g1["rxn_id"].astype(np.int64))
    if g1["rxn_id"].duplicated().any():
        die(f"duplicate rxn_id in {R5.FEAT_G1}")
    G = g1.set_index("rxn_id").reindex(ids)
    g1_ok = ((G["xtb_status"].fillna("absent").astype(str) == "ok").to_numpy()
             & ~G[f4673].isna().any(axis=1).to_numpy())
    rev4 = have_lab & g1_ok
    n_scored4 = getattr(CE, "N_SCORED", None)
    if n_scored4 is not None and int(rev4.sum()) != int(n_scored4):
        die(f"rev 4 scored rows recomputed = {int(rev4.sum())}, compare_espley.N_SCORED = {n_scored4}")
    if (usable & ~rev4).any():
        die(f"{int((usable & ~rev4).sum())} row-set rows are not rev 4 scored rows: {ids[usable & ~rev4][:10].tolist()}")
    s4 = set(rows4)
    dropped = [dict(rxn_id=int(ids[i]), reason="not in rows_rev4.csv (rev 4 row rule: hygiene / NaN target)"
                    if int(ids[i]) not in s4 else "in rows_rev4.csv, not in rows_rev5.csv (rev 5 block features)")
               for i in np.flatnonzero(rev4 & ~usable)]
    if (in5 & ~have_lab).any():
        die(f"{int((in5 & ~have_lab).sum())} rows_rev5 rows in ds3 without an ok label in prep.pkl: "
            f"{ids[in5 & ~have_lab][:10].tolist()}")
    if not usable.any():
        die("empty row set")

    # targets (Espley's numbers; role targets through the swap vector)
    Y = {t: np.asarray(CE.target_values(t, esp, C), dtype=float) for t in TARGETS5}
    for t in TARGETS5:
        bad = usable & ~np.isfinite(Y[t])
        if bad.any():
            die(f"target {t}: {int(bad.sum())} row-set rows without a finite value: {ids[bad][:10].tolist()}")

    X_our = {a: F[cols[a]].reindex(ids).to_numpy(dtype=float) for a in OUR_ARMS}
    for a in OUR_ARMS:
        if not np.isfinite(X_our[a][usable]).all():
            die(f"arm {a}: non-finite features on row-set rows")

    esp46 = list(CE.espley_X(esp).columns)
    tune = list(CE.espley_X(esp, drop=CE.TUNE_DROPS[0]).columns)
    if len(esp46) != 46 or esp46 != list(C["esp46"]) or tune != list(C["esp_tune"]):
        die(f"Espley columns differ from prep.pkl: ML {len(esp46)} (prep {len(C['esp46'])}), tuning {len(tune)} "
            f"(prep {len(C['esp_tune'])})")
    Xe = {"raw": CE.espley_X(esp).to_numpy(dtype=float), "role": CE.espley_X(esp, sw).to_numpy(dtype=float)}
    Xt = {"raw": CE.espley_X(esp, None, CE.TUNE_DROPS[0]).to_numpy(dtype=float),
          "role": CE.espley_X(esp, sw, CE.TUNE_DROPS[0]).to_numpy(dtype=float)}
    for t in TARGETS5:
        ok = np.isfinite(Y[t])
        for what, X in (("ML", Xe[xkind(t)]), ("tuning", Xt[xkind(t)])):
            if not np.isfinite(X[ok]).all():
                die(f"Espley {what} features: non-finite values on rows with a {t} target")

    split = {s: CE.espley_split(n, s) for s in SEEDS}
    want = {s: np.sort(te[usable[te]]) for s, (tr, te) in split.items()}
    D = dict(C=C, esp=esp, ids=ids, n=n, have_lab=have_lab, usable=usable, rev4=rev4, dropped=dropped,
             rows5=rows5, cols=cols, Y=Y, X_our=X_our, Xe=Xe, Xt=Xt, esp46=esp46, tune=tune, split=split, want=want,
             versions=FE.versions())
    D["sha_ours"] = {a: {t: data_sha(X_our[a][usable], Y[t][usable], ids[usable]) for t in TARGETS5} for a in OUR_ARMS}
    D["sha_esp"] = {t: data_sha(Xe[xkind(t)][np.isfinite(Y[t])], Y[t][np.isfinite(Y[t])], ids[np.isfinite(Y[t])])
                    for t in TARGETS5}
    D["inputs"] = dict(espley_ds=str(CE.ESP_DS), espley_ds_sha256=R5.sha256(CE.ESP_DS),
                       ds3_sorted_by_rxn_id=bool(np.all(np.diff(ids) > 0)), prep=str(PREP),
                       prep_sha256=R5.sha256(PREP), labels=str(R5.LABELS), labels_sha256=lab_sha, **rows_info,
                       rows_rev4=str(R5.ROWS4), rows_rev4_sha256=R5.sha256(R5.ROWS4),
                       prereg_rev5b=str(R5.PREREG5B_JSON), prereg_rev5b_sha256=R5.sha256(R5.PREREG5B_JSON),
                       ext_sel_blocks=R5.ext_sel_blocks(), arm_n_features={a: len(cols[a]) for a in OUR_ARMS},
                       espley_ml_columns=esp46, espley_tuning_columns=tune)
    log(f"rows: Espley ds3 {n}, labelled {int(have_lab.sum())}, rev 4 scored {int(rev4.sum())}, rows_rev5 "
        f"{len(rows5)}, row set = labelled ∩ rows_rev5 {int(usable.sum())} ({len(dropped)} rev 4 rows left out); "
        f"arms {', '.join(f'{a} ({len(cols[a])})' for a in OUR_ARMS)}, EXT_SEL blocks {D['inputs']['ext_sel_blocks']}")
    return D


# ------------------------------------------------------------------------------------------ D-3(a) units
def ours_unit(D, arm, t, s, allow_fit=True):
    """Our arm x KRR on target t, seed s (compare_espley.our_pipeline on the row set)."""
    est, grid = T.GRIDS[OUR_MODEL]
    fp = dict(stage="D-3(a)", kind="ours", arm=arm, columns=D["cols"][arm], model=OUR_MODEL, est=repr(est), grid=grid,
              target=t, seed=s, rows="row set (labelled AND rows_rev5) of espley_split(n, seed)",
              pipeline="compare_espley.our_pipeline: GridSearchCV(_make_pipe, KFold(5, shuffle, rs=seed), neg MAE)",
              n_espley=D["n"], data_sha256=D["sha_ours"][arm][t], versions=D["versions"])

    def fit():
        if arm == "EXT_SEL" and D["cols"]["EXT_SEL"] == D["cols"]["ESPLEY73"]:
            rec0, p0, _ = ours_unit(D, "ESPLEY73", t, s, allow_fit)     # identical columns -> identical fit
            return dict(kind="ours", arm=arm, model=OUR_MODEL, target=t, seed=s, n_train=rec0["n_train"],
                        n_test=rec0["n_test"], best_params=rec0["best_params"], same_columns_as="ESPLEY73"), p0.copy()
        te, p, best = CE.our_pipeline(D["X_our"][arm], D["Y"][t], D["usable"], D["n"], s, est, grid)
        tr = D["split"][s][0]
        return (dict(kind="ours", arm=arm, model=OUR_MODEL, target=t, seed=s, n_train=int(D["usable"][tr].sum()),
                     n_test=int(len(te)), best_params=best, same_columns_as=None),
                pred_frame(D, te, D["Y"][t], p))
    return run_unit("d3a", f"ours__{arm}__{MODEL_NAME[OUR_MODEL]}__{t}__s{s}", fp, fit, allow_fit)


def espfeat_unit(D, model, t, s, allow_fit=True):
    """Espley's 46 AM1 features x our pipeline (GRIDS model) on target t, seed s, their rows with a target."""
    est, grid = T.GRIDS[model]
    y = D["Y"][t]
    ok = np.isfinite(y)
    fp = dict(stage="D-3(a)", kind="espley_features_our_pipeline", model=model, est=repr(est), grid=grid, target=t,
              seed=s, columns=D["esp46"], am1_distortions_swapped=t in ROLE,
              rows="Espley rows with a finite target of espley_split(n, seed) (rev 4 role_ours convention)",
              pipeline="compare_espley.our_pipeline", n_espley=D["n"], data_sha256=D["sha_esp"][t],
              versions=D["versions"])

    def fit():
        te, p, best = CE.our_pipeline(D["Xe"][xkind(t)], y, ok, D["n"], s, est, grid)
        tr = D["split"][s][0]
        return (dict(kind="espley_features_our_pipeline", model=model, target=t, seed=s, n_train=int(ok[tr].sum()),
                     n_test=int(len(te)), best_params=best), pred_frame(D, te, y, p))
    return run_unit("d3a", f"espfeat__{MODEL_NAME[model]}__{t}__s{s}", fp, fit, allow_fit)


def theirs_role(D, allow_fit=True):
    """Their-protocol SVR / KRR on the role targets -> (predictions [row, rxn_id, y, yhat, model, target, seed,
    n_train], params {target: {model: params}}, info). Rev 4 files are reused when they cover every scored test row."""
    n, Y, want, split = D["n"], D["Y"], D["want"], D["split"]
    ntr = {t: {s: int(np.isfinite(Y[t])[split[s][0]].sum()) for s in SEEDS} for t in ROLE}
    problems = []
    if ROLE_THEIRS.exists() and ROLE_THEIRS_JSON.exists():
        rp = pd.read_parquet(ROLE_THEIRS)
        rj = json.loads(ROLE_THEIRS_JSON.read_text())
        rp = rp[rp["target"].isin(ROLE)].copy()
        if len(set(rp["protocol"].astype(str))) != 1:
            problems.append(f"protocols {sorted(set(rp['protocol'].astype(str)))}")
        # rev 4 tuned the role targets on whichever TUNE_DROPS set replicated hps.pkl; the recompute path, D-3(b) and
        # hence D-4 (role SVR params) tune on TUNE_DROPS[0]: reuse only files made on exactly those columns
        rev4_tune = list(rj.get("tuning_features") or [])
        if rev4_tune != D["tune"]:
            problems.append(f"rev 4 tuning columns ({len(rev4_tune)}) != TUNE_DROPS[0] ({len(D['tune'])}): "
                            f"only in rev 4 {sorted(set(rev4_tune) - set(D['tune']))[:5]}, only in TUNE_DROPS[0] "
                            f"{sorted(set(D['tune']) - set(rev4_tune))[:5]}")
        for t in ROLE:
            for m in CE.THEIR_TUNE:
                if not isinstance((((rj.get("tuned") or {}).get(t) or {}).get(m) or {}).get("best_params"), dict):
                    problems.append(f"{t} {m}: no tuned params in {ROLE_THEIRS_JSON.name}")
                for s in SEEDS:
                    g = rp[(rp["target"] == t) & (rp["model"] == m) & (rp["seed"] == s)]
                    rows = g["row"].to_numpy(np.int64)
                    if len(set(rows.tolist())) != len(rows):
                        problems.append(f"{t} {m} seed {s}: duplicate rows")
                    miss = np.setdiff1d(want[s], rows)
                    if miss.size:
                        problems.append(f"{t} {m} seed {s}: {miss.size} scored test rows not covered")
                    elif np.max(np.abs(g["y"].to_numpy(float) - Y[t][rows])) > Y_TOL:
                        problems.append(f"{t} {m} seed {s}: y differs from the role target")
        if not problems:
            preds = rp[["row", "rxn_id", "y", "yhat", "model", "target", "seed"]].copy()
            preds["row"] = preds["row"].astype(np.int64)
            preds["n_train"] = [ntr[t][int(s)] for t, s in zip(preds["target"], preds["seed"])]
            params = {t: {m: dict(rj["tuned"][t][m]["best_params"]) for m in CE.THEIR_TUNE} for t in ROLE}
            info = dict(source="reused: rev 4 compare_espley.py espley_role theirs", preds=str(ROLE_THEIRS),
                        preds_sha256=R5.sha256(ROLE_THEIRS), json=str(ROLE_THEIRS_JSON),
                        json_sha256=R5.sha256(ROLE_THEIRS_JSON), rev4_replicated=rj.get("replicated"),
                        rev4_tuning_columns=len(rev4_tune),
                        rev4_tuning_columns_equal_TUNE_DROPS0=rev4_tune == D["tune"],     # gated above: True
                        params=params, n_train=ntr)
            return preds, params, info
    else:
        problems.append(f"{ROLE_THEIRS} / {ROLE_THEIRS_JSON} missing")
    log(f"their-protocol role predictions of rev 4 not reusable ({problems[:5]}): recomputing (units)")
    frames, params, recs = [], {}, {}
    for t in ROLE:
        for m in CE.THEIR_TUNE:
            est, grid = CE.THEIR_TUNE[m]
            ok = np.isfinite(Y[t])
            fp = dict(stage="D-3(a)", kind="their_protocol_role", model=m, target=t, tune_est=repr(est),
                      tune_grid=grid, ml_est=repr(CE.THEIR_ML[m]()), tune_columns=D["tune"], ml_columns=D["esp46"],
                      protocol="compare_espley.their_protocol (tuned once on the seed-TUNE_SEED training split)",
                      data_sha256=data_sha(D["Xt"]["role"][ok], D["Xe"]["role"][ok], Y[t][ok], D["ids"][ok]),
                      versions=D["versions"])

            def fit(t=t, m=m):
                best, cv, per = CE.their_protocol(D["Xt"]["role"], D["Xe"]["role"], Y[t], n, m)
                if sorted(per) != sorted(SEEDS):
                    raise ValueError(f"their_protocol seeds {sorted(per)} != {sorted(SEEDS)}")
                fr = [pred_frame(D, te, Y[t], p).assign(seed=int(s)) for s, (te, p) in per.items()]
                return dict(kind="their_protocol_role", model=m, target=t, best_params=best, cv_mae=cv), \
                    pd.concat(fr, ignore_index=True)
            rec, pr, how = run_unit("d3a", f"theirs__{m}__{t}", fp, fit, allow_fit)
            log(f"  their protocol {m} {t}: {how}")
            frames.append(pr.assign(model=m, target=t))
            params.setdefault(t, {})[m] = dict(rec["best_params"])
            recs[f"{m}__{t}"] = dict(best_params=rec["best_params"], cv_mae=rec["cv_mae"], seconds=rec["seconds"])
    preds = pd.concat(frames, ignore_index=True)
    preds["n_train"] = [ntr[t][int(s)] for t, s in zip(preds["target"], preds["seed"])]
    return preds, params, dict(source="recomputed here (compare_espley.their_protocol)", problems=problems[:20],
                               units=recs, params=params, n_train=ntr)


def stored_preds(D):
    """Espley's stored test predictions (rev 4 prep) for interaction / ΔE‡ / ΔG‡, all 5 models."""
    if not STORED.exists():
        die(f"{STORED} missing (rev 4 compare_espley.py prep)")
    sp = pd.read_parquet(STORED)
    sp = sp[sp["target"].isin(RAW3)].copy()
    sp["row"] = sp["row"].astype(np.int64)
    sp["seed"] = sp["seed"].astype(int)
    models = sorted(set(sp["model"].astype(str)))
    if models != sorted(CE.ESP_MODELS.values()):
        die(f"{STORED}: models {models}, expected {sorted(CE.ESP_MODELS.values())}")
    for m in models:
        for t in RAW3:
            g = sp[(sp["model"] == m) & (sp["target"] == t)]
            if sorted(set(g["seed"])) != sorted(SEEDS):
                die(f"{STORED}: {m} {t} seeds {sorted(set(g['seed']))} != {sorted(SEEDS)}")
            for s in SEEDS:
                gs = g[g["seed"] == s]
                rows = gs["row"].to_numpy(np.int64)
                if np.setdiff1d(D["want"][s], rows).size or len(set(rows.tolist())) != len(rows):
                    die(f"{STORED}: {m} {t} seed {s} does not cover the scored test rows once each")
                if np.max(np.abs(gs["y"].to_numpy(float) - D["Y"][t][rows])) > Y_TOL:
                    die(f"{STORED}: {m} {t} seed {s}: stored y differs from Espley's target column")
    sp["n_train"] = sp["seed"].map({s: int(len(D["split"][s][0])) for s in SEEDS})
    return sp[["row", "rxn_id", "y", "yhat", "model", "target", "seed", "n_train"]]


def espley_svr_key(t):
    return (ESP, P_STORED if t in RAW3 else P_THEIRS, ESP_ARM, "SVR")


def per_seed_table(sc):
    rows = []
    for key, g in sc.groupby(KEYS + ["target", "seed"], sort=False):
        y, p = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
        e = np.abs(p - y)
        m = T.mets(y, p)
        rows.append(dict(zip(KEYS + ["target", "seed"], key), series=g["series"].iloc[0], label=LABEL[key[4]],
                         mae=m["mae"], se=float(np.std(e) / np.sqrt(len(e))), nmae=m["nmae"], r2=m["r2"],
                         rmse=m["rmse"], pearson_r=m["pearson_r"], n_test=int(len(e)),
                         n_train=int(g["n_train"].iloc[0])))
    return pd.DataFrame(rows)


def seed_mae(per, k, t):
    g = per[(per["side"] == k[0]) & (per["protocol"] == k[1]) & (per["arm"] == k[2]) & (per["model"] == k[3])
            & (per["target"] == t)]
    s = g.set_index("seed")["mae"].reindex(SEEDS)
    if s.isna().any():
        die(f"{k} {t}: per-seed MAE missing for seeds {s.index[s.isna()].tolist()}", 1)
    return s.to_numpy(float)


def rev4_reference():
    """Rev 4 seed-mean MAE of G1 · ESPLEY46 · KRR per target (3,327 rows; reported for continuity only)."""
    if not REV4_PER_SEED.exists():
        return None
    r = pd.read_csv(REV4_PER_SEED)
    need = {"side", "protocol", "arm", "model", "target", "mae"}
    if not need <= set(r.columns):
        return None
    g = r[r["side"].astype(str).str.startswith("Ours G1") & (r["arm"] == "ESPLEY46") & (r["model"] == "KRR")
          & r["target"].isin(TARGETS5)]
    return dict(file=str(REV4_PER_SEED), g1_espley46_krr_mae=g.groupby("target")["mae"].mean().to_dict())


def d3a(D):
    t_start = time.time()
    frames, timing, best = [], {}, {}
    n_units = len(OUR_ARMS) * len(TARGETS5) * len(SEEDS)
    k = 0
    # 1. ours: ESPLEY46 / ESPLEY73 / EXT_SEL x KRR
    for a in OUR_ARMS:
        for t in TARGETS5:
            for s in SEEDS:
                k += 1
                rec, p, how = ours_unit(D, a, t, s)
                frames.append(p.assign(side=OURS, protocol=P_OURS, arm=a, model=MODEL_NAME[OUR_MODEL], target=t,
                                       seed=s, n_train=rec["n_train"]))
                best.setdefault(a, {}).setdefault(t, {})[str(s)] = rec["best_params"]
                timing[f"ours__{a}__{t}__s{s}"] = rec["seconds"]
                log(f"[ours {k:3d}/{n_units}] {a:8s} {t:30s} seed {s:2d}: MAE "
                    f"{np.mean(np.abs(p['yhat'] - p['y'])):6.3f} (n={len(p)}) {how}")
    # 2. Espley stored predictions (interaction / ΔE‡ / ΔG‡)
    frames.append(stored_preds(D).assign(side=ESP, protocol=P_STORED, arm=ESP_ARM))
    # 3. Espley their protocol on the role targets
    tp, tparams, tinfo = theirs_role(D)
    frames.append(tp.assign(side=ESP, protocol=P_THEIRS, arm=ESP_ARM))
    log(f"their-protocol role: {tinfo['source']}")
    # 4. Espley 46 AM1 features x our pipeline, all GRIDS models, all 5 targets
    esp_best = {}
    n_units = len(T.GRIDS) * len(TARGETS5) * len(SEEDS)
    k = 0
    for m in T.GRIDS:
        for t in TARGETS5:
            for s in SEEDS:
                k += 1
                rec, p, how = espfeat_unit(D, m, t, s)
                frames.append(p.assign(side=ESP, protocol=P_OURS, arm=ESP_ARM, model=MODEL_NAME[m], target=t,
                                       seed=s, n_train=rec["n_train"]))
                esp_best.setdefault(MODEL_NAME[m], {}).setdefault(t, {})[str(s)] = rec["best_params"]
                timing[f"espfeat__{MODEL_NAME[m]}__{t}__s{s}"] = rec["seconds"]
                log(f"[espley feat. x our pipeline {k:3d}/{n_units}] {MODEL_NAME[m]:5s} {t:30s} seed {s:2d}: "
                    f"{how}")

    allp = pd.concat(frames, ignore_index=True)
    allp["row"] = allp["row"].astype(np.int64)
    allp["seed"] = allp["seed"].astype(int)
    allp["series"] = ""
    allp.loc[allp["side"] == OURS, "series"] = allp.loc[allp["side"] == OURS, "arm"]
    for t in TARGETS5:
        k_svr = espley_svr_key(t)
        m = ((allp["target"] == t) & (allp["side"] == k_svr[0]) & (allp["protocol"] == k_svr[1])
             & (allp["arm"] == k_svr[2]) & (allp["model"] == k_svr[3]))
        allp.loc[m, "series"] = ESP
    allp["scored"] = D["usable"][allp["row"].to_numpy()]
    if allp.duplicated(KEYS + ["target", "seed", "row"]).any():
        die("duplicate (side, protocol, arm, model, target, seed, row) prediction", 1)

    # cross-check: Espley features x our pipeline on the role targets vs rev 4 role_ours (same function and rows)
    xcheck = None
    if ROLE_OURS.exists():
        ro = pd.read_parquet(ROLE_OURS)
        ro = ro[ro["target"].isin(ROLE)][["row", "model", "target", "seed", "yhat"]]
        mine = allp[(allp["side"] == ESP) & (allp["protocol"] == P_OURS) & allp["target"].isin(ROLE)]
        mm = mine.merge(ro.assign(row=ro["row"].astype(np.int64), seed=ro["seed"].astype(int)),
                        on=["row", "model", "target", "seed"], how="inner", suffixes=("", "_rev4"))
        dmax = mm.assign(d=(mm["yhat"] - mm["yhat_rev4"]).abs()).groupby(["model", "target"])["d"].max()
        xcheck = dict(file=str(ROLE_OURS), n_mine=int(len(mine)), n_matched=int(len(mm)),
                      max_abs_yhat_diff=float(dmax.max()) if len(dmax) else None,
                      per_model_target={f"{a}|{b}": float(v) for (a, b), v in dmax.items()})
        log(f"cross-check vs rev 4 role_ours: matched {len(mm)}/{len(mine)} rows, max |Δyhat| "
            f"{xcheck['max_abs_yhat_diff']}")

    # gates: every entity scored on exactly the row-set test rows of each seed, same y everywhere
    sc = allp[allp["scored"]].copy()
    for key, g in sc.groupby(KEYS + ["target", "seed"], sort=False):
        if not np.array_equal(np.sort(g["row"].to_numpy()), D["want"][key[-1]]):
            die(f"GATE: test rows of {key} differ from the row-set test rows ({len(g)} vs "
                f"{len(D['want'][key[-1]])})", 1)
    yy = sc.groupby(["target", "seed", "row"])["y"].agg(["min", "max"])
    ydev = float((yy["max"] - yy["min"]).max())
    ydev_t = max(float(np.max(np.abs(sc.loc[sc["target"] == t, "y"].to_numpy(float)
                                     - D["Y"][t][sc.loc[sc["target"] == t, "row"].to_numpy()]))) for t in TARGETS5)
    if max(ydev, ydev_t) > Y_TOL:
        die(f"GATE: sides disagree on the target values (max |Δy| {ydev:.2e}, vs target vector {ydev_t:.2e})", 1)

    per = per_seed_table(sc)
    ents = per.groupby(KEYS + ["target"], sort=False)["seed"].nunique()
    if (ents != len(SEEDS)).any():
        die(f"entities without all {len(SEEDS)} seeds: {ents[ents != len(SEEDS)].index.tolist()[:5]}", 1)
    n_ent = {tgt: int((ents.index.get_level_values("target") == tgt).sum()) for tgt in TARGETS5}
    exp_ent = {t: len(OUR_ARMS) + len(T.GRIDS) + (len(CE.ESP_MODELS) if t in RAW3 else len(CE.THEIR_TUNE))
               for t in TARGETS5}
    if n_ent != exp_ent:
        die(f"entities per target {n_ent}, expected {exp_ent}", 1)

    summ = (per.groupby(KEYS + ["target"], sort=False)
            .agg(series=("series", "first"), label=("label", "first"), n_seeds=("seed", "nunique"),
                 mae=("mae", "mean"), mae_sd=("mae", "std"), se_espley_def=("se", "mean"),
                 n_test=("n_test", "mean"), n_train=("n_train", "mean"), r2=("r2", "mean"), nmae=("nmae", "mean"))
            .reset_index())
    summ["se_seeds"] = summ["mae_sd"] / np.sqrt(summ["n_seeds"])
    summ["espley_svr"] = [tuple(r) == espley_svr_key(r_t) for r, r_t in
                          zip(summ[KEYS].itertuples(index=False, name=None), summ["target"])]
    strongest = {}
    for t in TARGETS5:
        cand = summ[(summ["side"] == ESP) & (summ["target"] == t)].sort_values(["mae", "protocol", "model"],
                                                                                kind="mergesort")
        r = cand.iloc[0]
        strongest[t] = (r["side"], r["protocol"], r["arm"], r["model"])
    summ["espley_strongest"] = [r_t in strongest and tuple(r) == strongest[r_t] for r, r_t in
                                zip(summ[KEYS].itertuples(index=False, name=None), summ["target"])]

    # tests: every our arm vs Espley SVR and vs the strongest Espley-side model
    tests = []
    for a in OUR_ARMS:
        ko = (OURS, P_OURS, a, MODEL_NAME[OUR_MODEL])
        for t in TARGETS5:
            b = seed_mae(per, ko, t)
            po = per[(per["side"] == OURS) & (per["arm"] == a) & (per["target"] == t)]
            n_te, n_tr = float(po["n_test"].mean()), float(po["n_train"].mean())
            for comp, ke in (("Espley SVR", espley_svr_key(t)), ("Espley strongest", strongest[t])):
                e = seed_mae(per, ke, t)
                pe = per[(per["side"] == ke[0]) & (per["protocol"] == ke[1]) & (per["arm"] == ke[2])
                         & (per["model"] == ke[3]) & (per["target"] == t)]
                d = e - b
                nb = R5.nadeau_bengio(d, n_te, n_tr)
                tests.append(dict(our_arm=a, our_model=MODEL_NAME[OUR_MODEL], target=t, label=LABEL[t],
                                  comparator=comp, esp_protocol=ke[1], esp_model=ke[3], espley_mae=float(e.mean()),
                                  ours_mae=float(b.mean()), diff_mean=nb["mean"], diff_sd=float(np.std(d, ddof=1)),
                                  t=nb["t"], p=nb["p"], ci95_lo=nb["ci95"][0], ci95_hi=nb["ci95"][1], J=nb["J"],
                                  n_test=n_te, n_train_ours=n_tr, n_train_espley=float(pe["n_train"].mean()),
                                  mae_ratio_mean=float(np.mean(e / b)), mae_ratio_of_means=float(e.mean() / b.mean()),
                                  seeds_ours_lower=int((b < e).sum()),
                                  prereg_rev4_main=bool(a == "ESPLEY46" and comp == "Espley SVR"),
                                  note=("EXT_SEL selected on dev rows overlapping these test rows; unbiased: D-3(b)"
                                        if a == "EXT_SEL" else ""),
                                  **{f"d_seed{s}": float(v) for s, v in zip(SEEDS, d)}))
    tests = pd.DataFrame(tests)

    # outputs
    per_o = order_frame(per, ("target", "series", "seed"))
    summ_o = order_frame(summ, ("target", "series"))
    tests_o = order_frame(tests, ("target",))
    fig = sc[sc["series"] != ""][["series", "target", "seed", "rxn_id", "y", "yhat"]]
    fig = order_frame(fig, ("series", "target", "seed")).round(6)
    rows_df = pd.DataFrame(dict(rxn_id=D["ids"][D["usable"]], espley_row=np.flatnonzero(D["usable"])))
    R5.write_atomic(PREDS_A, lambda f: allp.to_parquet(f, index=False))
    write_csv(RES / "D3a_rows.csv", rows_df)
    write_csv(RES / "D3a_per_seed.csv", per_o)
    write_csv(RES / "D3a_summary.csv", summ_o)
    write_csv(RES / "D3a_tests.csv", tests_o)
    write_csv(RES / "D3a_predictions.csv", fig)
    R5.write_json(RES / "D3a.json", dict(
        phase="D-3(a) Espley comparison on their split (docs/specs/REV5_FEATURES_FIGURES.md)", created=FE.now(),
        job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), code=str(HERE), code_file=D["code_file"],
        prereg=D["prereg"], partition=D["partition"], select_audit=D["select_audit"],
        inputs=dict(D["inputs"], stored_preds=str(STORED), stored_preds_sha256=R5.sha256(STORED)),
        rows=dict(n_espley=D["n"], n_labelled=int(D["have_lab"].sum()), n_rev4_scored=int(D["rev4"].sum()),
                  n_row_set=int(D["usable"].sum()), n_left_out_vs_rev4=len(D["dropped"]),
                  left_out_vs_rev4=D["dropped"],
                  per_seed={str(s): dict(n_test_scored=int(len(D["want"][s])),
                                         n_train_ours=int(D["usable"][D["split"][s][0]].sum()),
                                         n_train_espley_split=int(len(D["split"][s][0])),
                                         n_test_espley_split=int(len(D["split"][s][1]))) for s in SEEDS}),
        design=dict(targets=TARGETS5, seeds=SEEDS, split="compare_espley.espley_split (test = first half of the 20 %)",
                    ours=dict(arms=list(OUR_ARMS), model=OUR_MODEL, pipeline="compare_espley.our_pipeline",
                              grid=T.GRIDS[OUR_MODEL][1]),
                    espley_svr={t: "|".join(espley_svr_key(t)) for t in TARGETS5},
                    espley_candidates="stored Ridge/KRR/SVR/2-layer NN/4-layer NN (interaction, ΔE‡, ΔG‡); "
                                      "their-protocol SVR/KRR (role); our pipeline Ridge/KRR/SVR/XGB on Espley's 46 "
                                      "features (all 5)",
                    test="rev5_common.nadeau_bengio(d = MAE Espley − MAE ours per seed, mean n_test, mean n_train "
                         "ours)"),
        their_protocol_role=tinfo, espley_features_role_crosscheck_vs_rev4=xcheck,
        strongest={t: "|".join(strongest[t]) for t in TARGETS5},
        checks=dict(test_rows_identical_across_sides=True, max_abs_y_diff_across_sides=ydev,
                    max_abs_y_diff_vs_target=ydev_t, entities_per_target=n_ent),
        tests=tests_o.to_dict(orient="records"),
        ours_best_params=best, espley_features_our_pipeline_best_params=esp_best,
        rev4_reference=rev4_reference(), notes=NOTES_A, unit_seconds=timing,
        outputs=dict(rows=str(RES / "D3a_rows.csv"), per_seed=str(RES / "D3a_per_seed.csv"),
                     summary=str(RES / "D3a_summary.csv"), tests=str(RES / "D3a_tests.csv"),
                     figure_predictions=str(RES / "D3a_predictions.csv"), all_predictions=str(PREDS_A)),
        wall_seconds=round(time.time() - t_start, 1)))

    pd.set_option("display.width", 250)
    log("\nD-3(a) seed-mean test MAE (kcal/mol), scored rows:")
    log(summ_o[summ_o["series"] != ""][["target", "series", "protocol", "model", "mae", "se_espley_def",
                                        "n_test"]].round(3).to_string(index=False))
    log("\nD-3(a) tests (Espley − ours, Nadeau–Bengio):")
    log(tests_o[["our_arm", "target", "comparator", "esp_protocol", "esp_model", "espley_mae", "ours_mae",
                 "diff_mean", "ci95_lo", "ci95_hi", "p", "seeds_ours_lower"]].round(4).to_string(index=False))
    log(f"saved -> {RES}/D3a_*.csv, D3a.json, {PREDS_A}")


# ------------------------------------------------------------------------------------------ D-3(b)
def their_order(n, rows):
    """rows in the order of their hyp_tuning.py X_train: rank in train_test_split(range(n), test_size=0.2,
    random_state=TUNE_SEED) = its training part, then its hold-out part."""
    tr_t, rest_t = train_test_split(np.arange(n), test_size=0.2, random_state=CE.TUNE_SEED)
    rank = np.empty(n, dtype=np.int64)
    rank[np.concatenate([tr_t, rest_t])] = np.arange(n)
    return np.asarray(rows)[np.argsort(rank[np.asarray(rows)], kind="stable")]


def pct_ci(b):
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def d3b(D):
    t_start = time.time()
    dev, lock, part = D["dev"], D["lock"], D["partition"]            # gated + select-audited in main()
    ids, n, usable, Y = D["ids"], D["n"], D["usable"], D["Y"]
    tr = np.flatnonzero(usable & np.isin(ids, dev))                  # ds3 row order (positions ascending)
    te = np.flatnonzero(usable & np.isin(ids, lock))
    if not len(tr) or not len(te):
        die(f"D-3(b): empty train ({len(tr)}) or test ({len(te)})")
    if np.isin(ids[tr], lock).any() or np.isin(ids[te], dev).any() or np.intersect1d(tr, te).size:
        die("D-3(b): train and test overlap")
    tr_their = their_order(n, tr)
    if sorted(tr_their.tolist()) != tr.tolist():
        die("their_order changed the train rows", 1)
    Xo = D["X_our"]["EXT_SEL"]
    est, grid = T.GRIDS[OUR_MODEL]
    cv_ours = dict(kind="KFold", n_splits=R5.N_INNER, shuffle=True, random_state=R5.SEED)
    units = {}
    for t in TARGETS5:
        y = Y[t]
        fp = dict(stage="D-3(b)", kind="ours", arm="EXT_SEL", columns=D["cols"]["EXT_SEL"], model=OUR_MODEL,
                  est=repr(est), grid=grid, cv=cv_ours, target=t, train_order="ds3 row order (manual_tt_solvent.pkl)",
                  data_sha256=data_sha(Xo[tr], y[tr], ids[tr], Xo[te], y[te], ids[te]), versions=D["versions"])

        def fit_ours(y=y):
            gs = GridSearchCV(T._make_pipe(clone(est)), grid,
                              cv=KFold(R5.N_INNER, shuffle=True, random_state=R5.SEED), scoring=SCORING,
                              n_jobs=NJOBS, error_score="raise")
            gs.fit(Xo[tr], y[tr])
            best = dict(gs.best_params_)
            return (dict(kind="ours", n_train=int(len(tr)), n_test=int(len(te)), best_params=best,
                         edge_hits={k: T._edge_hit(k, v, grid[k]) for k, v in best.items()},
                         cv_mae=float(-gs.best_score_)),
                    pred_frame(D, te, y, gs.best_estimator_.predict(Xo[te])))
        rec, p, how = run_unit("d3b", f"ours__EXT_SEL__KRR__{t}", fp, fit_ours)
        units[(OURS, P_OURS, "EXT_SEL", MODEL_NAME[OUR_MODEL], t)] = (rec, p)
        log(f"[D-3(b)] ours EXT_SEL KRR {t:30s}: lockbox MAE {np.mean(np.abs(p['yhat'] - p['y'])):6.3f} {how}")
        Xt, Xm = D["Xt"][xkind(t)], D["Xe"][xkind(t)]
        for m in CE.THEIR_TUNE:
            est_m, grid_m = CE.THEIR_TUNE[m]
            fp = dict(stage="D-3(b)", kind="their_protocol", model=m, tune_est=repr(est_m), tune_grid=grid_m,
                      ml_est=repr(CE.THEIR_ML[m]()), tune_columns=D["tune"], ml_columns=D["esp46"],
                      am1_distortions_swapped=t in ROLE, cv="GridSearchCV(cv=5): unshuffled KFold(5)",
                      train_order=f"their hyp_tuning.py X_train order (train_test_split rank, rs={CE.TUNE_SEED})",
                      scaling="StandardScaler on X only (fit on the train rows), y raw", target=t,
                      data_sha256=data_sha(Xt[tr_their], Xm[tr_their], y[tr_their], ids[tr_their], Xm[te], y[te],
                                           ids[te]), versions=D["versions"])

            def fit_their(y=y, Xt=Xt, Xm=Xm, m=m, est_m=est_m, grid_m=grid_m):
                gs = GridSearchCV(est_m, grid_m, cv=5, scoring=SCORING, n_jobs=NJOBS)     # error_score as theirs
                gs.fit(StandardScaler().fit_transform(Xt[tr_their]), y[tr_their])
                sc = StandardScaler().fit(Xm[tr_their])
                mdl = CE.THEIR_ML[m]().set_params(**gs.best_params_).fit(sc.transform(Xm[tr_their]), y[tr_their])
                n_nan = int(np.isnan(gs.cv_results_["mean_test_score"]).sum())
                return (dict(kind="their_protocol", n_train=int(len(tr_their)), n_test=int(len(te)),
                             best_params=dict(gs.best_params_), cv_mae=float(-gs.best_score_),
                             n_grid_points=int(len(gs.cv_results_["params"])), n_nan_cv_scores=n_nan),
                        pred_frame(D, te, y, mdl.predict(sc.transform(Xm[te]))))
            rec, p, how = run_unit("d3b", f"theirs__{m}__{t}", fp, fit_their)
            units[(ESP, P_THEIRS, ESP_ARM, m, t)] = (rec, p)
            log(f"[D-3(b)] Espley {m} their protocol {t:30s}: lockbox MAE "
                f"{np.mean(np.abs(p['yhat'] - p['y'])):6.3f} {how}")

    # bootstrap: one index matrix over the test rows (ds3 row order), shared by every target and side
    for key, (rec, p) in units.items():
        if not np.array_equal(p["row"].to_numpy(np.int64), te):
            die(f"D-3(b) {key}: prediction rows differ from the test rows — move {UNITS['d3b']} away", 1)
        if np.max(np.abs(p["y"].to_numpy(float) - Y[key[-1]][te])) > Y_TOL:
            die(f"D-3(b) {key}: y differs from the target vector", 1)
    B = np.random.default_rng(R5.SEED).integers(0, len(te), size=(R5.N_BOOT, len(te)))
    boot_sha = hashlib.sha256(B.astype("<i8").tobytes()).hexdigest()
    summ, paired, bm = [], [], {}
    for key, (rec, p) in units.items():
        y, yhat = p["y"].to_numpy(float), p["yhat"].to_numpy(float)
        e = np.abs(yhat - y)
        bm[key] = e[B].mean(axis=1)
        lo, hi = pct_ci(bm[key])
        m = T.mets(y, yhat)
        series = "EXT_SEL" if key[0] == OURS else (ESP if key[3] == "SVR" else "")
        summ.append(dict(zip(KEYS + ["target"], key), series=series, label=LABEL[key[-1]], n_train=rec["n_train"],
                         n_test=rec["n_test"], mae=m["mae"], mae_ci_lo=lo, mae_ci_hi=hi, nmae=R5.nmae(y, yhat),
                         r2=m["r2"], pearson_r=m["pearson_r"], cv_mae=rec["cv_mae"],
                         best_params=json.dumps(rec["best_params"], sort_keys=True)))
    for t in TARGETS5:
        ko = (OURS, P_OURS, "EXT_SEL", MODEL_NAME[OUR_MODEL], t)
        mo = float(np.mean(np.abs(units[ko][1]["yhat"] - units[ko][1]["y"])))
        for m in CE.THEIR_TUNE:
            ke = (ESP, P_THEIRS, ESP_ARM, m, t)
            me = float(np.mean(np.abs(units[ke][1]["yhat"] - units[ke][1]["y"])))
            d = bm[ke] - bm[ko]                                          # paired: same resampled reactions
            lo, hi = pct_ci(d)
            paired.append(dict(target=t, label=LABEL[t], espley_model=m, espley_protocol=P_THEIRS,
                               primary=m == "SVR", ours="EXT_SEL KRR", n_train=int(len(tr)), n_test=int(len(te)),
                               espley_mae=me, ours_mae=mo, diff_espley_minus_ours=me - mo, diff_ci_lo=lo,
                               diff_ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0),
                               frac_boot_diff_le0=float(np.mean(d <= 0)), mae_ratio=me / mo, n_boot=R5.N_BOOT))
    summ = order_frame(pd.DataFrame(summ), ("target", "series"))
    paired = order_frame(pd.DataFrame(paired), ("target",))
    preds = pd.concat([p.assign(**dict(zip(KEYS + ["target"], key))) for key, (rec, p) in units.items()],
                      ignore_index=True)
    preds["series"] = np.where(preds["side"] == OURS, "EXT_SEL", np.where(preds["model"] == "SVR", ESP, ""))
    R5.write_atomic(PREDS_B, lambda f: preds.to_parquet(f, index=False))
    write_csv(RES / "D3b_summary.csv", summ)
    write_csv(RES / "D3b_paired.csv", paired)
    write_csv(RES / "D3b_predictions.csv",
              order_frame(preds[KEYS + ["series", "target", "rxn_id", "y", "yhat"]], ("target", "series")).round(6))
    R5.write_json(RES / "D3b.json", dict(
        phase="D-3(b) lockbox head-to-head, EXT_SEL KRR vs Espley their protocol (docs/specs/REV5_FEATURES_FIGURES.md)",
        created=FE.now(), job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), code=str(HERE),
        code_file=D["code_file"], prereg=D["prereg"],
        inputs=dict(D["inputs"], dev_ids=str(FE.DEV_IDS), dev_sha256=R5.sha256(FE.DEV_IDS),
                    lockbox_ids=str(R5.LOCKBOX), lockbox_sha256=R5.sha256(R5.LOCKBOX)),
        partition=part, select_audit=D["select_audit"],
        rows=dict(n_train=int(len(tr)), n_test=int(len(te)), n_lockbox=len(lock),
                  n_lockbox_in_espley=int(np.isin(ids, lock).sum()), n_dev=len(dev),
                  n_dev_in_espley=int(np.isin(ids, dev).sum()),
                  rule="train = dev ∩ row set, test = lockbox ∩ row set (row set = labelled ∩ rows_rev5)"),
        espley_protocol=dict(tune="THEIR_TUNE grid, GridSearchCV(cv=5 unshuffled, neg MAE, error_score default) on "
                                  "StandardScaler(X_train[TUNE_DROPS[0] columns]), y raw",
                             train_order=f"rank in train_test_split(range(n), test_size=0.2, random_state="
                                         f"{CE.TUNE_SEED}) (their hyp_tuning.py X_train order)",
                             ml="THEIR_ML with the tuned params (KRR: tuned poly, run rbf) on StandardScaler(46 "
                                "ML columns of the train rows), y raw",
                             role_targets="AM1 distortion features re-assigned by the swap vector"),
        ours=dict(arm="EXT_SEL", columns=len(D["cols"]["EXT_SEL"]), model=OUR_MODEL, cv=cv_ours,
                  train_order="ds3 row order (manual_tt_solvent.pkl)"),
        bootstrap=dict(n_boot=R5.N_BOOT, seed=R5.SEED, n_test=int(len(te)), ci="95 % percentile",
                       indices="np.random.default_rng(SEED).integers(0, n_test, size=(N_BOOT, n_test)) over the test "
                               "rows in ds3 row order; shared by every target and side", index_sha256=boot_sha),
        units={"|".join(map(str, k)): {kk: v for kk, v in rec.items() if kk != "fingerprint"}
               for k, (rec, p) in units.items()},
        summary=summ.to_dict(orient="records"), paired=paired.to_dict(orient="records"),
        notes=["EXT_SEL was fixed (PREREG_REV5b) before the lockbox was used; this is its primary Espley comparison.",
               "interaction_energies_dft in Espley's sign (positive = stabilising).",
               "Both sides train on exactly the same rows (dev ∩ row set); Espley with their protocol, ours nested."],
        wall_seconds=round(time.time() - t_start, 1)))
    log("\nD-3(b) lockbox (bootstrap 95 % CI):")
    log(paired[["target", "espley_model", "espley_mae", "ours_mae", "diff_espley_minus_ours", "diff_ci_lo",
                "diff_ci_hi", "frac_boot_diff_le0"]].round(4).to_string(index=False))
    log(f"saved -> {RES}/D3b_*.csv, D3b.json, {PREDS_B}")


# ------------------------------------------------------------------------------------------ D-4
def job_lc(t, s, tr_u, te_u, rid_tr, Xe, Xo, y, esp_params, our_params, fracs):
    """One (target, seed) learning curve: both sides on the same subsets of the training frame (worker process)."""
    sc = StandardScaler().fit(Xe[tr_u])                  # ml_analysis.py: X_train scaled first, subsets drawn after
    frame = pd.DataFrame({"pos": np.asarray(tr_u, dtype=np.int64)}, index=pd.Index(rid_tr, name="rxn_id"))
    ze_te, xo_te, y_te = sc.transform(Xe[te_u]), Xo[te_u], y[te_u]
    rows = []
    for f in fracs:
        sub = frame.sample(frac=f, random_state=s)["pos"].to_numpy()
        me = CE.THEIR_ML["SVR"]().set_params(**esp_params).fit(sc.transform(Xe[sub]), y[sub])
        mo = T._make_pipe(clone(T.GRIDS[OUR_MODEL][0])).set_params(**our_params).fit(Xo[sub], y[sub])
        for side, p in ((ESP, me.predict(ze_te)), ("EXT_SEL", mo.predict(xo_te))):
            p = np.asarray(p, dtype=float)
            if not np.isfinite(p).all():
                raise ValueError(f"{t} seed {s} frac {f} {side}: non-finite predictions")
            rows.append(dict(target=t, side=side, frac=float(f), n_train=int(len(sub)), seed=int(s),
                             mae=float(np.mean(np.abs(p - y_te))), n_test=int(len(te_u))))
    return rows


def job_their_lc(t, s, Xm, y, n, params, stored, fracs):
    """Their ml_analysis.py learning curve on all their rows (split of seed s) vs their stored predictions."""
    tr, te = CE.espley_split(n, s)
    sc = StandardScaler().fit(Xm[tr])
    Xtr, ytr = pd.DataFrame(sc.transform(Xm[tr])), pd.Series(y[tr])
    zte = sc.transform(Xm[te])
    out = {}
    for f in fracs:
        if f not in stored:
            out[str(f)] = dict(stored=False)
            continue
        m = CE.THEIR_ML["SVR"]().set_params(**params).fit(Xtr.sample(frac=f, random_state=s).to_numpy(),
                                                          ytr.sample(frac=f, random_state=s).to_numpy())
        p, ref = m.predict(zte), stored[f]
        same_n = len(ref) == len(p)
        out[str(f)] = dict(stored=True, n=int(len(ref)),
                           max_abs_pred_diff=float(np.max(np.abs(p - ref))) if same_n else None,
                           mae_rerun=float(np.mean(np.abs(p - y[te]))),
                           mae_stored=float(np.mean(np.abs(ref - y[te]))) if len(ref) == len(te) else None)
    return out


def their_lc_replication(D):
    fracs = [f for f in R5.LC_FRACS if f < 1.0]
    for f in (CE.ESP_RES, CE.ESP_HPS):
        if not f.exists():
            die(f"{f} missing (Espley data)")
    fp = jsonable(dict(stage="D-4 check", ml_results_sha256=R5.sha256(CE.ESP_RES), hps_sha256=R5.sha256(CE.ESP_HPS),
                       espley_ds_sha256=D["inputs"]["espley_ds_sha256"], fracs=fracs, targets=RAW3,
                       columns=D["esp46"], versions=D["versions"]))
    if LC_REP.exists():
        rec = json.loads(LC_REP.read_text())
        if rec.get("fingerprint") != fp:
            die(f"{LC_REP} was made with other inputs — move it away", 3)
        return rec
    res = pd.read_pickle(CE.ESP_RES).set_index("model_target")
    hps = pd.read_pickle(CE.ESP_HPS)
    if not res.index.is_unique:
        die(f"{CE.ESP_RES}: duplicate model_target")
    tasks, n_missing = [], 0
    for t in RAW3:
        mt = f"svr_{t}"
        if mt not in res.index or mt not in hps:
            die(f"{mt} missing in {CE.ESP_RES} / {CE.ESP_HPS}")
        row = res.loc[mt]
        for s_i, s in enumerate(row["random_state"]):
            stored = {}
            for f in fracs:
                col = f"{f}_y_test_pred_values"
                if col in row.index:
                    stored[f] = np.asarray(row[col][s_i], dtype=float).ravel()
                else:
                    n_missing += 1
            tasks.append(((t, int(s)), (t, int(s), D["Xe"]["raw"], D["Y"][t], D["n"], dict(hps[mt]["b_hps"]),
                                        stored, fracs)))
    out = {}
    for key, r in pool_map(job_their_lc, tasks, "D-4 their learning-curve re-run"):
        out[f"{key[0]}__s{key[1]}"] = r
    diffs = [v["max_abs_pred_diff"] for r in out.values() for v in r.values() if v.get("max_abs_pred_diff") is not None]
    rec = dict(fingerprint=fp,
               what="their ml_analysis.py learning curve re-run on all their rows (SVR, hps.pkl b_hps, X_train "
                    "scaled on the whole training split, X_train.sample(frac, random_state=seed)) vs the "
                    "'<frac>_y_test_pred_values' of ml_results.pkl",
               per_target_seed=out, n_compared=len(diffs), n_not_stored=n_missing,
               max_abs_pred_diff=max(diffs) if diffs else None, tolerance=LC_REP_TOL,
               replicated=bool(diffs) and max(diffs) <= LC_REP_TOL and n_missing == 0, created=FE.now())
    R5.write_json(LC_REP, rec)
    return rec


def reach(curve, level):
    """curve: rows sorted by frac with n_train_mean, mae_mean. Smallest (linearly interpolated) training-row count at
    which the mean MAE reaches level."""
    x, yv, fr = (curve["n_train_mean"].to_numpy(float), curve["mae_mean"].to_numpy(float),
                 curve["frac"].to_numpy(float))
    below = np.flatnonzero(yv <= level)
    if not below.size:
        return dict(reached=False, n_train=None, frac=None, interpolated=False, at_first_point=False)
    k = int(below[0])
    if k == 0:
        return dict(reached=True, n_train=float(x[0]), frac=float(fr[0]), interpolated=False, at_first_point=True)
    w = (level - yv[k - 1]) / (yv[k] - yv[k - 1])
    return dict(reached=True, n_train=float(x[k - 1] + w * (x[k] - x[k - 1])),
                frac=float(fr[k - 1] + w * (fr[k] - fr[k - 1])), interpolated=True, at_first_point=False)


def d4(D):
    t_start = time.time()
    usable, Y = D["usable"], D["Y"]
    fracs = [float(f) for f in R5.LC_FRACS]
    if fracs[-1] != 1.0 or fracs != sorted(fracs):
        die(f"LC_FRACS {fracs} must be ascending and end at 1.0")
    _, tparams, tinfo = theirs_role(D, allow_fit=False)
    hps = pd.read_pickle(CE.ESP_HPS)
    esp_params, esp_src = {}, {}
    for t in TARGETS5:
        if t in RAW3:
            if f"svr_{t}" not in hps:
                die(f"svr_{t} missing in {CE.ESP_HPS}")
            esp_params[t], esp_src[t] = dict(hps[f"svr_{t}"]["b_hps"]), f"hps.pkl svr_{t} b_hps"
        else:
            esp_params[t], esp_src[t] = dict(tparams[t]["SVR"]), f"their protocol, D-3(a) ({tinfo['source']})"
    our, d3a_mae = {}, {}
    for t in TARGETS5:
        for s in SEEDS:
            rec, p, _ = ours_unit(D, "EXT_SEL", t, s, allow_fit=False)
            if not np.array_equal(np.sort(p["row"].to_numpy(np.int64)), D["want"][s]):
                die(f"D-3(a) EXT_SEL unit {t} seed {s}: rows differ from the scored test rows", 1)
            our[(t, s)] = dict(rec["best_params"])
            d3a_mae[(t, s)] = float(np.mean(np.abs(p["yhat"].to_numpy(float) - p["y"].to_numpy(float))))

    Xo = D["X_our"]["EXT_SEL"]
    tasks, fps, results = [], {}, {}
    for order in LC_ORDERS:
        for t in TARGETS5:
            Xe = D["Xe"][xkind(t)]
            for s in SEEDS:
                tr, te = D["split"][s]
                tr_u, te_u = tr[usable[tr]], te[usable[te]]
                if order == "ds3":
                    tr_u = np.sort(tr_u)                 # ds3 row order (positions ascending)
                fp = jsonable(dict(stage="D-4", order=order, frame=LC_ORDERS[order], target=t, seed=s, fracs=fracs,
                                   esp_model=repr(CE.THEIR_ML["SVR"]()), esp_params=esp_params[t],
                                   esp_params_source=esp_src[t], esp_columns=D["esp46"],
                                   am1_distortions_swapped=t in ROLE, our_model=OUR_MODEL, our_params=our[(t, s)],
                                   our_columns=D["cols"]["EXT_SEL"],
                                   subset="DataFrame.sample(frac=f, random_state=seed), same subset for both sides",
                                   espley_scaling="StandardScaler fit on the whole training frame",
                                   data_sha256=data_sha(Xe[tr_u], Xo[tr_u], Y[t][tr_u], tr_u, Xe[te_u], Xo[te_u],
                                                        Y[t][te_u], te_u), versions=D["versions"]))
                path = UNITS["d4"] / f"{order}__{t}__s{s}.json"
                if path.exists():
                    rec = json.loads(path.read_text())
                    if rec.get("fingerprint") != fp:
                        diff = sorted(k for k in set(fp) | set(rec.get("fingerprint") or {})
                                      if (rec.get("fingerprint") or {}).get(k) != fp.get(k))
                        die(f"{path} exists but was made with other inputs (keys {diff}) — move {UNITS['d4']} away",
                            3)
                    results[(order, t, s)] = rec["rows"]
                    continue
                fps[(order, t, s)] = fp
                tasks.append(((order, t, s), (t, s, tr_u, te_u, D["ids"][tr_u], Xe, Xo, Y[t], esp_params[t],
                                              our[(t, s)], fracs)))
    log(f"D-4: {len(results)} (order, target, seed) curves cached, {len(tasks)} to fit ({len(fracs)} fractions x 2 "
        f"sides; frame orders {list(LC_ORDERS)}, primary {LC_PRIMARY})")
    for key, rows in pool_map(job_lc, tasks, "D-4 learning curve"):
        R5.write_json(UNITS["d4"] / f"{key[0]}__{key[1]}__s{key[2]}.json",
                      dict(fingerprint=fps[key], rows=rows, finished=FE.now(), host=socket.gethostname(),
                           job=os.environ.get("SLURM_JOB_ID", "local")))
        results[key] = rows
        log(f"  D-4 {key[0]} {key[1]} seed {key[2]} done")
    rep = their_lc_replication(D)
    log(f"their learning-curve re-run vs ml_results.pkl: {rep['n_compared']} comparisons, max |Δpred| "
        f"{rep['max_abs_pred_diff']}, replicated {rep['replicated']}")

    lc_all = pd.DataFrame([dict(r, order=key[0]) for key in results for r in results[key]])
    lc_all = order_frame(lc_all.sort_values(["order", "side", "frac", "seed"], kind="mergesort"), ("target",))
    # gate: EXT_SEL at frac 1.0 = the D-3(a) EXT_SEL fit (same rows, params and pipeline), in both frame orders
    f1 = lc_all[(lc_all["side"] == "EXT_SEL") & (lc_all["frac"] == 1.0)]
    dev1 = {f"{o}|{t}|{s}": float(abs(f1[(f1["order"] == o) & (f1["target"] == t) & (f1["seed"] == s)]["mae"].iloc[0]
                                      - d3a_mae[(t, s)]))
            for o in LC_ORDERS for t in TARGETS5 for s in SEEDS}
    if max(dev1.values()) > FRAC1_TOL:
        die(f"D-4 EXT_SEL at frac 1.0 differs from D-3(a) by up to {max(dev1.values()):.3g} kcal/mol "
            f"(> {FRAC1_TOL}): {dict(sorted(dev1.items(), key=lambda kv: -kv[1])[:5])}", 1)
    lc = lc_all[lc_all["order"] == LC_PRIMARY].drop(columns="order").reset_index(drop=True)
    lc_sens = lc_all[lc_all["order"] != LC_PRIMARY].reset_index(drop=True)
    sens = (lc_sens.groupby(["order", "target", "side", "frac"], sort=False)
            .agg(n_train_mean=("n_train", "mean"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                 n_seeds=("seed", "nunique"))
            .reset_index())
    summ = (lc.groupby(["target", "side", "frac"], sort=False)
            .agg(n_train_mean=("n_train", "mean"), n_train_min=("n_train", "min"), n_train_max=("n_train", "max"),
                 mae_mean=("mae", "mean"), mae_sd=("mae", "std"), n_seeds=("seed", "nunique"),
                 n_test_mean=("n_test", "mean"))
            .reset_index())
    if (summ["n_seeds"] != len(SEEDS)).any():
        die("D-4: a (target, side, frac) without all seeds", 1)
    summ["mae_se"] = summ["mae_sd"] / np.sqrt(summ["n_seeds"])
    d3a_svr = None
    if (RES / "D3a_summary.csv").exists():
        a = pd.read_csv(RES / "D3a_summary.csv")
        d3a_svr = {t: float(a[(a["series"] == ESP) & (a["target"] == t)]["mae"].iloc[0]) for t in TARGETS5
                   if ((a["series"] == ESP) & (a["target"] == t)).any()}
    reach_rows = []
    for t in TARGETS5:
        cur = summ[(summ["target"] == t) & (summ["side"] == "EXT_SEL")].sort_values("frac")
        e1 = summ[(summ["target"] == t) & (summ["side"] == ESP) & (summ["frac"] == 1.0)].iloc[0]
        refs = [("Espley SVR, D-4 frac 1.0 (row-set training rows)", float(e1["mae_mean"]))]
        if d3a_svr and t in d3a_svr:
            refs.append(("Espley SVR, D-3(a) (stored / their protocol, their training rows)", d3a_svr[t]))
        for name, level in refs:
            r = reach(cur, level)
            reach_rows.append(dict(target=t, label=LABEL[t], reference=name, reference_mae=level,
                                   espley_n_train_frac1=float(e1["n_train_mean"]),
                                   ext_sel_mae_frac1=float(cur[cur["frac"] == 1.0]["mae_mean"].iloc[0]), **r))
    reach_df = pd.DataFrame(reach_rows)
    write_csv(RES / "D4_learning_curves.csv", lc[["target", "side", "frac", "n_train", "seed", "mae"]])
    write_csv(RES / "D4_learning_curves_summary.csv", summ)
    write_csv(RES / "D4_learning_curves_reach.csv", reach_df)
    write_csv(RES / "D4_learning_curves_sensitivity.csv",
              lc_sens[["order", "target", "side", "frac", "n_train", "seed", "mae"]])
    R5.write_json(RES / "D4_learning_curves.json", dict(
        phase="D-4 learning curves, Espley SVR vs EXT_SEL KRR (docs/specs/REV5_FEATURES_FIGURES.md)",
        created=FE.now(), job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), code=str(HERE),
        code_file=D["code_file"], prereg=D["prereg"], partition=D["partition"], select_audit=D["select_audit"],
        inputs=dict(D["inputs"], hps=str(CE.ESP_HPS), hps_sha256=R5.sha256(CE.ESP_HPS)),
        design=dict(fracs=fracs, seeds=SEEDS, targets=TARGETS5, frame=LC_ORDERS[LC_PRIMARY],
                    frame_orders=LC_ORDERS, primary_order=LC_PRIMARY,
                    primary_order_fixed_in=f"espley_rev5.py LC_PRIMARY (code_file: sha256 "
                                           f"{D['code_file']['sha256'][:12]}, committed and unchanged at run time: "
                                           f"{D['code_file']['committed_unchanged']})",
                    subset="frame.sample(frac=f, random_state=seed); the same subset for both sides",
                    test_rows="the D-3(a) scored test rows of the seed (fixed)",
                    espley="SVR (THEIR_ML) with fixed params; StandardScaler fit on the whole training frame, then "
                           "the subset of the scaled rows (ml_analysis.py); y raw",
                    ours="EXT_SEL KRR, _make_pipe with the D-3(a) per-seed best params, refit on the subset"),
        espley_params=esp_params, espley_params_source=esp_src,
        ext_sel_params={f"{t}|{s}": our[(t, s)] for t in TARGETS5 for s in SEEDS},
        checks=dict(frac1_ext_sel_vs_d3a_max_abs_mae_diff=max(dev1.values()), tolerance=FRAC1_TOL,
                    their_learning_curve_replication={k: v for k, v in rep.items() if k != "per_target_seed"},
                    their_learning_curve_replication_file=str(LC_REP)),
        reach=reach_df.to_dict(orient="records"), summary=summ.to_dict(orient="records"),
        sensitivity_frame_order=dict(file=str(RES / "D4_learning_curves_sensitivity.csv"),
                                     summary=sens.to_dict(orient="records")),
        wall_seconds=round(time.time() - t_start, 1)))
    log("\nD-4 mean test MAE by training fraction:")
    log(summ.pivot_table(index=["target", "frac"], columns="side", values="mae_mean").round(3).to_string())
    log(reach_df[["target", "reference", "reference_mae", "reached", "n_train", "frac"]].round(3).to_string(index=False))
    log(f"saved -> {RES}/D4_learning_curves*.csv / .json")


# ------------------------------------------------------------------------------------------ main
def select_audit(dev, lock):
    """final_eval.audit_select (first use of the lockbox), with its leak exit 3 remapped to 4 (3 = stale cache here)."""
    try:
        return FE.audit_select(dev, lock)
    except SystemExit as e:
        if e.code == 3:
            die("LOCKBOX LEAK (select audit, message above): exit 4 — no unit cache is at fault, do not move them", 4)
        raise


def code_file():
    """sha256 of this file and whether it is committed and unchanged (records the pre-result choice LC_PRIMARY)."""
    me = Path(__file__).resolve()
    why = FE.committed(me)
    return dict(file=str(me), sha256=R5.sha256(me), committed_unchanged=not why, reason=why or None,
                lc_primary=LC_PRIMARY)


def gates_only(D):
    dev, lock, part, audit = D["dev"], D["lock"], D["partition"], D["select_audit"]
    ids, usable = D["ids"], D["usable"]
    n_tr_b, n_te_b = int((usable & np.isin(ids, dev)).sum()), int((usable & np.isin(ids, lock)).sum())
    cached = {k: len(list(d.glob("*.json"))) if d.is_dir() else 0 for k, d in UNITS.items()}
    log(f"--gates-only: partition {part}\n  select audit: {audit['n_files']} files, {audit['n_distinct_ids']} "
        f"distinct ids = dev {audit['ids_equal_dev']}, lockbox ids found {audit['lockbox_ids_found']}\n"
        f"  code {D['code_file']}\n  D-3(a) scored test rows per seed "
        f"{ {s: len(D['want'][s]) for s in SEEDS} }, training rows of ours "
        f"{ {s: int(usable[D['split'][s][0]].sum()) for s in SEEDS} }\n  D-3(b) train {n_tr_b}, test {n_te_b}\n"
        f"  units planned: D-3(a) {len(OUR_ARMS) * 5 * 5} ours + {len(T.GRIDS) * 5 * 5} Espley-feature + their-protocol "
        f"role (reused or 4); D-3(b) {5 * (1 + len(CE.THEIR_TUNE))}; D-4 {len(LC_ORDERS) * 5 * 5} curves; cached "
        f"JSONs {cached}\n"
        f"  rev 4 files: stored {STORED.exists()}, role_theirs {ROLE_THEIRS.exists() and ROLE_THEIRS_JSON.exists()}, "
        f"role_ours {ROLE_OURS.exists()}; nothing fitted or written")


def main():
    global NJOBS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["d3a", "d3b", "d4", "all"])
    ap.add_argument("--gates-only", action="store_true", help="run every gate and the data load, fit nothing")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        die("espley_rev5.py runs inside a SLURM job only (sbatch r5_espley.sh), never on the login node")
    NJOBS = int(os.environ.get("R5_NJOBS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    CE.N_JOBS = NJOBS                          # compare_espley.our_pipeline / their_protocol read it at call time
    log(f"[espley_rev5 {args.stage}] {FE.now()} host {socket.gethostname()} job {os.environ.get('SLURM_JOB_ID')} "
        f"n_jobs {NJOBS}\n  code {HERE}\n  scratch {OUT}\n  rev 4 Espley side {REV4}")
    # gates before any data is read, as final_eval.py (D-1): pre-registration, partition, first use of the lockbox
    prereg = FE.gate_prereg()
    rows5, dev, lock = FE.read_ids(R5.ROWS5), FE.read_ids(FE.DEV_IDS), FE.read_ids(R5.LOCKBOX)
    part = FE.gate_partition(rows5, dev, lock)
    audit = select_audit(dev, lock)
    log(f"gates ok: HEAD {prereg['git_head'][:10]}, rows_rev5 {part['n_rows']} = dev {part['n_dev']} + lockbox "
        f"{part['n_lockbox']}, select caches hold exactly the dev ids")
    try:
        D = load_data()
    except SystemExit as e:
        if isinstance(e.code, str):          # train_ml_single / compare_espley gates: sys.exit(message) -> exit 1
            die(e.code)                      # an input / gate failure: exit 2
        raise                                # our own die(): code already set
    except Exception as e:                   # noqa: BLE001 — nothing is fitted in load_data: an input failure
        traceback.print_exc()
        die(f"loading the inputs failed: {type(e).__name__}: {e}")
    D.update(prereg=prereg, dev=dev, lock=lock, partition=part, select_audit=audit, code_file=code_file())
    if args.gates_only:
        gates_only(D)
        return
    stages = ["d3a", "d3b", "d4"] if args.stage == "all" else [args.stage]
    for st in stages:
        log(f"\n===== {st} =====")
        {"d3a": d3a, "d3b": d3b, "d4": d4}[st](D)


if __name__ == "__main__":
    main()
