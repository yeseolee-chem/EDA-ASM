"""fu2/tests/test_eval.py — FU2 Phase E-T (docs/specs/D1PAIR_FU2_E.md section 3): checks T-1 .. T-8 of fu2/evaluate.py
(+ X-1 .. X-3) BEFORE any FU2 test score is computed -> results/fu2/E_tests.json; exit 3 if any check fails.

  cd analysis/d1_pair && sbatch -p cpu1 --export=ALL,PYENV=fu2 fu2/jobs/fu2.slurm fu2/tests/test_eval.py

Inputs: rev 9 results only (results/M_predictions.parquet, M_report.json, E_bends.csv, E_metrics.csv,
E_secondary_tests.csv, E_representative.json, S_folds.csv, F_pairs.csv), FU2 unit JSONs (s_y and split sizes; no
prediction is opened), the exported split files, the tokens (train.Data) and synthetic frames. The whole run is inside
evaluate.no_test_scores(strict=False): reading the FU2 M_predictions.parquet or any FU2 unit parquet would STOP.

  T-1  env_d1fu2 imports numpy, pandas, scipy, sklearn, pyarrow, matplotlib, torch, rdkit (report only, never install)
  T-1env  this run's interpreter is env_d1fu2 (sys.prefix; gated: the T-2 / T-3 / X-3 certificates are for that stack)
  T-2  evaluate.bends_tab(rev 9 arm B representative T1 rows, split="T1") = results/E_bends.csv T1 rows, all columns
       (bool / int / str exact; float |a - b| <= 1e-9 max(1, |b|), exact equality recorded; decision D3)
  T-3  evaluate.metrics_tab (d1_eval.family_eval path) on the same rows: r2_fam_median, rho_err_median (and n_fam) of
       E_metrics.csv T1 / B / rep / P_main / all within 1e-12 (n / MAE / RMSE of every set x position recorded)
  T-4  s_y path: for the 30 units, evaluate.sy_of(D, outer-training pairs) = unit json final[i].sy (all 5 seeds, 1e-12);
       the same function on the tuning-training pairs (first inner fold) is finite and > 0
  T-5  inner split sizes: json n_inner_* + dropped_no_atoms[inner_*] = S_splits.json n_inner_* and json n_inner_* =
       len(train.Data.split()[inner_*]) (decision D2: S_splits.json counts are before the atom filter)
  T-6  compare sign / p on synthetic pairs: x = b -> d = 0, p = 1, CI [0, 0]; b smaller by 0.1 everywhere -> d = 0.1,
       CI lower > 0, b lower in all 5 folds; swapped -> d = -0.1, CI upper < 0
  T-7  negative pair-list gate: one pair (all targets) or one row removed from one model, or a duplicated row ->
       pair_sets_gate exits 3; a pair missing from b -> compare exits 3; positive controls pass
  T-8  representative ties (ADD2 3): exactly equal means -> AW-A, AW-A+G, GNN-A, GNN-A+G, TF-A, TF-A+G order; input
       row order irrelevant; equal means from different fold values
  X-1  no_test_scores blocks FU2 M_predictions / FU2 unit parquets (and with strict the rev 9 / FU1 units) before
       reading, allows pairs.parquet, restores pandas.read_parquet
  X-2  evaluate unit names / config order = train.uname / unit_list / models.MODELS x VARIANTS; env prefix = train
  X-3  evaluate.compare reproduces rev 9 E_secondary_tests.csv "B vs C" (T1, P_main, 10 targets) within 1e-12
gate_pass (required by evaluate.require_review) = every check except T-1 passed (decision D1; T-1env is gated).
E_tests.json also records evaluate.code_sha() (sha256 of the reused code), compared by evaluate.require_review.
T-3 / X-3 differences: both NaN = 0, one NaN = inf (a NaN can never pass the 1e-12 check vacuously).
"""
from __future__ import annotations

import importlib
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                    # analysis/d1_pair/fu2/tests
sys.path.insert(0, str(HERE.parent))
import evaluate as EV  # noqa: E402

F, C, E9 = EV.F, EV.C, EV.E9
ENV_D1FU2 = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2")         # = train.ENV_PREFIX (X-2 checks it)
TOL12 = 1e-12
RESULTS = {}


def record(name, ok, **kw):
    RESULTS[name] = dict(ok=bool(ok), **kw)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {json.dumps(kw, default=str)[:2000]}", flush=True)


def run(name, fn, *a):
    try:
        return fn(*a)
    except SystemExit as e:
        record(name, False, error=f"SystemExit {e.code}")
    except Exception as e:                                # noqa: BLE001  (a crashing check is a failed check)
        record(name, False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    return None


def adiff(a, b):
    """|a - b| with both NaN = 0.0 and exactly one NaN = inf (NaN-safe max / <= checks)."""
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if np.isnan(a) or np.isnan(b):
        return float("inf")
    return abs(a - b)


def exit_code(fn):
    """SystemExit code raised by fn(), or None if it returned."""
    try:
        fn()
    except SystemExit as e:
        return e.code
    return None


# ---------------------------------------------------------------- T-1
def t1():
    got = {}
    for m in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "matplotlib", "torch", "rdkit"):
        try:
            got[m] = str(getattr(importlib.import_module(m), "__version__", "imported"))
        except Exception as e:                            # noqa: BLE001
            got[m] = f"FAIL {type(e).__name__}: {e}"
    failed = [m for m, v in got.items() if v.startswith("FAIL")]
    in_env = Path(sys.prefix).resolve() == ENV_D1FU2.resolve()
    record("T-1", not failed, versions=got, failed_imports=failed, prefix=sys.prefix, in_env_d1fu2=in_env,
           python=sys.version.split()[0], note="spec 3: missing packages are reported, not installed")
    record("T-1env", in_env, prefix=sys.prefix, want=str(ENV_D1FU2),
           note="gated (D1): E-T and the scoring commands must run in env_d1fu2 (PYENV=fu2)")


# ---------------------------------------------------------------- rev 9 frames (T-2, T-3, X-3)
def rev9_frames(S):
    mrep = EV.read_json(C.RES / "M_report.json")
    p = C.RES / "M_predictions.parquet"
    if F.sha256(p) != mrep["predictions"]["sha256"]:
        raise SystemExit("rev 9 M_predictions.parquet differs from rev 9 M_report.json")
    M = pd.read_parquet(p, columns=["split", "arm", "model", "target", "pair_id", "y", "yhat", "fold"])
    M = M[M.split == "T1"]
    rep9 = EV.read_json(C.RES / "E_representative.json")
    out = {}
    for arm in ("B", "C"):
        q = M[M.arm == arm]
        want = q.target.map(lambda t, a=arm: rep9[f"T1|{a}|{t}"]).to_numpy()
        out[arm] = EV.annotate(q[q.model.to_numpy() == want].reset_index(drop=True), S)   # rev 9 file order kept
    return out


def frames_equal(got, ref, keys):
    if list(got.columns) != list(ref.columns):
        return False, dict(columns_got=list(got.columns), columns_ref=list(ref.columns))
    if len(got) != len(ref):
        return False, dict(n_got=len(got), n_ref=len(ref))
    g = got.sort_values(keys).reset_index(drop=True)
    r = ref.sort_values(keys).reset_index(drop=True)
    bad, maxd, exact = {}, 0.0, True
    for c in ref.columns:
        a, b = g[c], r[c]
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            same = (av == bv) | (np.isnan(av) & np.isnan(bv))
            d = np.where(same, 0.0, np.abs(av - bv))
            nbad = int((~(d <= 1e-9 * np.maximum(1.0, np.abs(np.nan_to_num(bv, posinf=0, neginf=0))))).sum())
            if nbad:
                bad[c] = nbad
            if len(d):
                maxd = max(maxd, float(np.nanmax(d)))
            exact &= bool(same.all())
        else:
            n = int((a.astype(str).to_numpy() != b.astype(str).to_numpy()).sum())
            if n:
                bad[c] = n
    return not bad, dict(n_rows=len(ref), mismatched_columns=bad, max_float_abs_diff=maxd, exactly_equal=exact and not bad)


def t2(R9):
    got = EV.bends_tab(R9["B"], split="T1")
    ref = pd.read_csv(C.RES / "E_bends.csv")
    ref = ref[ref.split == "T1"].reset_index(drop=True)
    ok, det = frames_equal(got, ref, ["fam", "target"])
    record("T-2", ok, **det)


def t3(R9):
    Mt = EV.metrics_tab(R9["B"], by=("model",))
    ref = pd.read_csv(C.RES / "E_metrics.csv")
    ref = ref[(ref.split == "T1") & (ref.arm == "B") & ref.rep.astype(bool)]
    g = Mt[(Mt["set"] == "P_main") & (Mt.pos == "all")].set_index("target")
    r = ref[(ref["set"] == "P_main") & (ref.pos == "all")].set_index("target")
    diffs, nfam_bad, model_bad = {}, [], []
    for t in EV.TARGETS:
        for c in ("r2_fam_median", "rho_err_median"):
            diffs[f"{t}:{c}"] = adiff(g.at[t, c], r.at[t, c])
        if int(g.at[t, "n_fam"]) != int(r.at[t, "n_fam"]):
            nfam_bad.append(t)
        if g.at[t, "model"] != r.at[t, "model"]:
            model_bad.append(t)
    J = Mt.merge(ref, on=["target", "set", "pos"], suffixes=("", "_ref"))
    info = {c: float(np.nanmax(np.abs(J[c].to_numpy(float) - J[f"{c}_ref"].to_numpy(float)))) for c in ("n", "mae", "rmse")}
    ok = max(diffs.values()) <= TOL12 and not nfam_bad and not model_bad and len(J) == len(ref) == len(Mt)
    record("T-3", ok, max_abs_diff=max(diffs.values()), diffs=diffs, n_fam_mismatch=nfam_bad, model_mismatch=model_bad,
           all_sets_max_abs_diff=info, n_rows=len(J))


# ---------------------------------------------------------------- T-4 / T-5 (tokens, unit jsons; no predictions)
def t45():
    import train as T
    D = T.Data()
    srep = EV.read_json(F.FU2_RES / "S_splits.json")
    r4, r5 = [], []
    for m, v in EV.CONFIGS:
        for k in range(EV.N_FOLDS):
            r = EV.unit_json(m, v, k)
            sp = D.split("T1", k)
            sy = EV.sy_of(D, sp["train"])
            dd = np.array([np.abs(sy - np.asarray(f["sy"], float)) for f in r["final"]], dtype=float)
            d = float(dd.max()) if dd.size and np.isfinite(dd).all() else float("inf")       # NaN-safe
            sy_in = EV.sy_of(D, sp["inner_train"])
            r4.append(dict(unit=EV.unit_name(m, v, k), max_abs_diff=d, n_finals=len(r["final"]),
                           n_train_json=r["n_train"], n_train_now=len(sp["train"]),
                           sy_inner_finite_positive=bool(np.isfinite(sy_in).all() and (sy_in > 0).all())))
            u = srep["units"][f"T1__f{k}.json"]
            for key in ("inner_train", "inner_val"):
                j, drop, now, s = int(r[f"n_{key}"]), int(r["dropped_no_atoms"][key]), len(sp[key]), int(u[f"n_{key}"])
                r5.append(dict(unit=EV.unit_name(m, v, k), split=key, json=j, dropped=drop, recomputed=now, s_splits=s,
                               literal_equal=j == s,
                               ok=bool(j + drop == s and j == now and drop == int(sp["dropped_no_atoms"][key]))))
    ok4 = all(x["max_abs_diff"] <= TOL12 and x["n_finals"] == EV.N_SEEDS and x["n_train_json"] == x["n_train_now"]
              and x["sy_inner_finite_positive"] for x in r4) and len(r4) == 30
    record("T-4", ok4, max_abs_diff=max(x["max_abs_diff"] for x in r4), units=r4)
    ok5 = all(x["ok"] for x in r5) and len(r5) == 60
    record("T-5", ok5, n_literal_equal=sum(x["literal_equal"] for x in r5), n_checks=len(r5),
           failed=[x for x in r5 if not x["ok"]], rows=r5,
           note="decision D2: S_splits.json counts are before the atom filter (folds 1-4 inner_train lose 1 pair)")
    return T


# ---------------------------------------------------------------- T-6 synthetic compare
def t6():
    rng = np.random.default_rng(7)
    n = 400
    pid = [f"Q{i:05d}" for i in range(n)]
    pairs = pd.DataFrame(dict(pair_id=pid, core=[f"core{i % 23}" for i in range(n)]))
    y = rng.normal(size=n)
    r = rng.uniform(0.05, 1.0, n)
    sg = rng.choice([-1.0, 1.0], n)

    def fr(yhat):
        return pd.DataFrame(dict(pair_id=pid, target="d1", y=y, yhat=yhat, fold=np.arange(n) % 5))
    b = fr(y + sg * r)
    a = EV.compare(b.copy(), b, pairs).iloc[0]
    c1 = dict(d=float(a.d), p=float(a.p), ci_lo=float(a.ci_lo), ci_hi=float(a.ci_hi))
    ok1 = a.d == 0 and a.p == 1.0 and a.ci_lo == 0 and a.ci_hi == 0
    x = fr(y + sg * (r + 0.1))
    q = EV.compare(x, b, pairs).iloc[0]
    c2 = dict(d=float(q.d), ci_lo=float(q.ci_lo), ci_hi=float(q.ci_hi), p=float(q.p), folds_b_lower=int(q.folds_b_lower))
    ok2 = abs(q.d - 0.1) <= TOL12 and q.ci_lo > 0 and q.folds_b_lower == 5 and q.n_folds == 5 and q.n_cores == 23
    q2 = EV.compare(b, x, pairs).iloc[0]
    c3 = dict(d=float(q2.d), ci_hi=float(q2.ci_hi))
    ok3 = abs(q2.d + 0.1) <= TOL12 and q2.ci_hi < 0
    record("T-6", ok1 and ok2 and ok3, x_equals_b=c1, b_smaller_by_0_1=c2, swapped=c3)


# ---------------------------------------------------------------- T-7 negative pair-list gate
def t7(S):
    tests = EV.split_tests()
    ref, removed = EV.ref_sets(S, tests)
    rng = np.random.default_rng(11)
    order = sorted(ref)
    fold = [k for k, _ in order]
    pid = [p for _, p in order]

    def syn():
        return pd.concat([pd.DataFrame(dict(pair_id=pid, fold=fold, target=t, y=S.loc[pid, f"dd_{t}"].to_numpy(float),
                                             yhat=S.loc[pid, f"dd_{t}"].to_numpy(float) + rng.normal(0, 0.3, len(pid))))
                          for t in EV.TARGETS], ignore_index=True)
    fr = {"m1": syn(), "m2": syn(), "m3": syn()}
    Pm = EV.eval_pairs(S, ref)
    victim = Pm.index[0]
    pos = exit_code(lambda: EV.pair_sets_gate(fr, ref, S))
    f_pair = dict(fr, m2=fr["m2"][fr["m2"].pair_id != victim])
    f_row = dict(fr, m3=fr["m3"].drop(fr["m3"][(fr["m3"].pair_id == victim) & (fr["m3"].target == "oi")].index))
    f_dup = dict(fr, m1=pd.concat([fr["m1"], fr["m1"].iloc[[0]]], ignore_index=True))
    neg = dict(pair_removed=exit_code(lambda: EV.pair_sets_gate(f_pair, ref, S)),
               one_row_removed=exit_code(lambda: EV.pair_sets_gate(f_row, ref, S)),
               duplicated_row=exit_code(lambda: EV.pair_sets_gate(f_dup, ref, S)),
               compare_pair_missing_in_b=exit_code(lambda: EV.compare(fr["m1"], f_pair["m2"], Pm, targets=["d1"])),
               compare_pair_missing_in_x=exit_code(lambda: EV.compare(f_pair["m2"], fr["m1"], Pm, targets=["d1"])))
    pos_cmp = exit_code(lambda: EV.compare(fr["m1"], fr["m2"], Pm, targets=["d1"]))
    ok = pos is None and pos_cmp is None and all(v == 3 for v in neg.values())
    record("T-7", ok, positive_gate=pos, positive_compare=pos_cmp, negative_exit_codes=neg, victim=victim,
           n_ref_pairs=len(ref), n_removed_no_atoms=len(removed), n_P_main=len(Pm))


# ---------------------------------------------------------------- T-8 ties
def t8():
    names = EV.CONFIG_NAMES

    def V(per_fold):
        return pd.DataFrame([dict(config=c, fold=k, target="d1", val_kcal=float(per_fold[c][k]))
                             for c in names for k in range(EV.N_FOLDS)])
    flat = lambda d: {c: [d[c]] * EV.N_FOLDS for c in names}               # noqa: E731
    base2 = {c: 2.0 for c in names}
    cases = [("all equal", flat({c: 1.0 for c in names}), "AW-A"),
             ("AW-A higher", flat({**{c: 1.0 for c in names}, "AW-A": 2.0}), "AW-A+G"),
             ("GNN-A+G = TF-A lowest", flat({**base2, "GNN-A+G": 1.0, "TF-A": 1.0}), "GNN-A+G"),
             ("GNN-A = AW-A+G lowest", flat({**base2, "GNN-A": 1.0, "AW-A+G": 1.0}), "AW-A+G"),
             ("TF-A = TF-A+G lowest", flat({**base2, "TF-A": 1.0, "TF-A+G": 1.0}), "TF-A"),
             ("GNN-A = GNN-A+G = TF-A+G lowest", flat({**base2, "GNN-A": 0.5, "GNN-A+G": 0.5, "TF-A+G": 0.5}), "GNN-A"),
             ("strict minimum TF-A+G", flat({**base2, "TF-A+G": 1.5}), "TF-A+G"),
             ("equal means from different folds", {**flat(base2), "TF-A": [1.0, 2.0, 3.0, 4.0, 5.0],
                                                   "GNN-A+G": [3.0] * 5, **{c: [4.0] * 5 for c in ("AW-A", "AW-A+G", "GNN-A")},
                                                   "TF-A+G": [3.5] * 5}, "GNN-A+G")]
    rng = np.random.default_rng(5)
    rows = []
    for name, pf, want in cases:
        v = V(pf)
        g1 = EV.pick_representatives(v)["d1"]
        g2 = EV.pick_representatives(v.iloc[rng.permutation(len(v))])["d1"]
        rows.append(dict(case=name, want=want, got=g1, got_shuffled=g2, ok=g1 == g2 == want))
    record("T-8", all(r["ok"] for r in rows), cases=rows, order=names)


# ---------------------------------------------------------------- extra checks
def x1():
    def read(p, strict):
        with EV.no_test_scores(strict=strict):
            pd.read_parquet(p)
    before = pd.read_parquet
    codes = dict(fu2_predictions=exit_code(lambda: read(F.FU2_RES / "M_predictions.parquet", True)),
                 fu2_unit=exit_code(lambda: read(EV.UNITS / f"{EV.unit_name('AW', 'A', 0)}.parquet", True)),
                 rev9_unit_strict=exit_code(lambda: read(EV.M9_T1 / "T1__C__XGB__d1__f0.parquet", True)),
                 xgbw_unit_strict=exit_code(lambda: read(EV.XW_DIR / "C__XGBW__d1__f0.parquet", True)),
                 splits_pairs_allowed=exit_code(lambda: read(F.SPLITS / "pairs.parquet", True)))
    restored = pd.read_parquet is before
    ok = restored and codes.pop("splits_pairs_allowed") is None and all(v == 3 for v in codes.values())
    record("X-1", ok, exit_codes=codes, restored=restored)


def x2(T):
    import models as Mo
    names = sorted(EV.unit_name(m, v, k) for m, v in EV.CONFIGS for k in range(EV.N_FOLDS))
    tn = sorted(T.uname(u) for u in T.unit_list("T1"))
    order = [(m, v) for m in Mo.MODELS for v in Mo.VARIANTS]
    ok = names == tn and order == EV.CONFIGS and Path(T.ENV_PREFIX) == ENV_D1FU2 and EV.TARGETS == list(F.TARGETS)
    record("X-2", ok, n_units=len(names), config_order=EV.CONFIG_NAMES, models_order=order)


def x3(R9):
    ref = pd.read_csv(C.RES / "E_secondary_tests.csv")
    ref = ref[(ref.comparison == "B vs C") & (ref.split == "T1") & (ref["set"] == "P_main")].set_index("target")
    B = R9["B"]
    pm = B[EV.set_mask(B, "P_main") & (B.target == "barrier")]
    pairs = pm.drop_duplicates("pair_id").set_index("pair_id")[["core"]]
    R = EV.compare(R9["C"], B, pairs, targets=EV.TARGETS).set_index("target")
    diffs = {}
    for t in EV.TARGETS:
        for a, b in (("mae_x", "mae_x"), ("mae_b", "mae_B"), ("d", "d_x_minus_B"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi"),
                     ("p", "p")):
            diffs[f"{t}:{a}"] = adiff(R.at[t, a], ref.at[t, b])
    nbad = [t for t in EV.TARGETS if int(R.at[t, "n_pairs"]) != int(ref.at[t, "n_pairs"])]
    record("X-3", max(diffs.values()) <= TOL12 and not nbad and len(ref) == len(EV.TARGETS),
           max_abs_diff=max(diffs.values()), n_pairs_mismatch=nbad, n_pairs=int(R.n_pairs.iloc[0]))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    with EV.no_test_scores(strict=False):                 # no FU2 test score can be opened in E-T
        run("T-1", t1)
        S = EV.s_table()
        R9 = run("rev9", rev9_frames, S)
        if R9 is not None:
            run("T-2", t2, R9)
            run("T-3", t3, R9)
            run("X-3", x3, R9)
        else:
            for n in ("T-2", "T-3", "X-3"):
                record(n, False, error="rev 9 frames not loaded")
        T = run("T-4", t45)
        if T is None:
            RESULTS.setdefault("T-5", dict(ok=False, error="T-4 / T-5 block failed"))
        run("T-6", t6)
        run("T-7", t7, S)
        run("T-8", t8)
        run("X-1", x1)
        if T is not None:
            run("X-2", x2, T)
        else:
            record("X-2", False, error="train not importable")
    RESULTS.pop("rev9", None)
    want = ["T-1", "T-1env", "T-2", "T-3", "T-4", "T-5", "T-6", "T-7", "T-8", "X-1", "X-2", "X-3"]
    for n in want:
        RESULTS.setdefault(n, dict(ok=False, error="not run"))
    failed = [n for n in want if not RESULTS[n]["ok"]]
    out = dict(tests=RESULTS, all_pass=not failed, gate_pass=not [n for n in failed if n != "T-1"], failed=failed,
               evaluate_sha256=F.sha256(EV.THIS), test_sha256=F.sha256(Path(__file__).resolve()),
               code_sha256=EV.code_sha(),
               python=sys.version, prefix=sys.prefix, host=socket.gethostname(), date=time.strftime("%Y-%m-%d %H:%M:%S"),
               seconds=time.time() - t0, fu2_test_scores_read=False, x_atom_method=EV.X_ATOM_METHOD,
               decisions="fu2/evaluate.py docstring D1-D15",
               gate_note="gate_pass = all but T-1 (D1; T-1env gated); T-1 failures are reported for the user's decision")
    F.write_json(F.FU2_RES / "E_tests.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "tests"}, indent=1))
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
