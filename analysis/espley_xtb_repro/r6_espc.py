#!/usr/bin/env python3
"""r6_espc.py — rev 6 Espley-protocol arms (ESP-C, OUR-C) and P-2 (spec §3-1, §6). Espley-pinned env (env_espley).

espley_protocol() = their code, patched only as results_rev6/P4_their_code_diff.patch allows (rows = split ∩ C via
R6_KEEP / R6_KEEP_TEST, XGB added), run as their scripts are run:
  1. dataset  <wd>/r6_<arm>.pkl, ds3 row order (index 0..3509): reaction_number, the X columns, and their target columns
              with OUR values (P-4): distortion_energy_1_dft = d1 (dipole), distortion_energy_2_dft = d2
              (dipolarophile), interaction_energies_dft = eint_spe (our sign), e_barrier_dft = barrier,
              sum_distortion_energies_dft = d1 + d2 (tuned by hyp_tuning.py, dropped by ml_analysis.py as in theirs);
              q_barrier_dft (ΔG‡) is not in our labels and is left out. Rows outside the label set carry NaN targets and
              are removed by R6_KEEP before anything is fitted.
  2. tune     hyp_tuning.py <dataset> in <wd>/hyp -> hps.pkl (seed-23 train ∩ keep, their GridSearchCV(cv=5) and
              Hyperband with validation = seed-23 val ∩ keep)
  3. ml       ml_analysis.py <seed> <dataset> <hps> in <wd>/ml once per (NN key, seed), NN keys in hps order
  4. collect  ml_results.pkl -> one unit per (model, target, seed) in $R6_SCRATCH/units/<group>/ (r6_train format):
              test rows = the split's test part ∩ R6_KEEP_TEST in split order (asserted: their y_test_true == our values)
              validation MAE (representative rule §3-3) = hps.pkl train_mae = |GridSearchCV.best_score_| for the
              sklearn models / XGB; for the NNs 5-fold CV (KFold(5), no shuffle = their cv=5) of the tuned NN on the
              seed-23 train ∩ keep rows (r6_nn.py, fresh process, seed 23).

  python r6_espc.py p2_ml | p2_hyp | p2_nn | p2_compare     # P-2 on their data (their hps for ml)
  python r6_espc.py arm <ESP-C|OUR-C|ESP-C_all>             # steps 1-4 (idempotent)
Exit codes: 0 ok, 2 bad inputs, 3 STOP.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev6_common as R6  # noqa: E402

P4_DIR = R6.SCRATCH / "P4"
PATCH = R6.RES6 / "P4_their_code_diff.patch"
ESPC = R6.SCRATCH / "espc"
TOL_P2 = 1e-8
MODEL_OF = {"ridge": "Ridge", "krr": "KRR", "svr": "SVR", "xgb": "XGB", "rf": "RF", "2_st_nn": "NN2", "4_st_nn": "NN4"}
TCOL = {"d1": "distortion_energy_1_dft", "d2": "distortion_energy_2_dft", "Int": "interaction_energies_dft",
        "dEa": "e_barrier_dft"}
T_OF = {v: k for k, v in TCOL.items()}
ARM_CFG = {"ESP-C": dict(feat=R6.FEAT_ESP, keep="C", keep_test=("C", "D"), group="B"),
           "OUR-C": dict(feat=R6.FEAT_OUR, keep="C", keep_test=("C",), group="B"),
           "ESP-C_all": dict(feat=R6.FEAT_ESP, keep="C_all", keep_test=("C_all",), group="Call")}


def split_key(mt):
    pre = next(k for k in sorted(MODEL_OF, key=len, reverse=True) if mt.startswith(k + "_"))
    return pre, mt[len(pre) + 1:]


def patched_code():
    """P4/r6/{hyp_tuning,ml_analysis}.py after checking orig/ = their files and r6/ = the sha256 recorded with the
    committed patch (results_rev6/P4_patched_sha256.json; P4_their_code_diff.patch = diff -u orig r6)."""
    src = {"hyp_tuning.py": R6.HYP_SRC, "ml_analysis.py": R6.ESP_ML_SRC}
    rec = json.loads((R6.RES6 / "P4_patched_sha256.json").read_text())
    for name, orig in src.items():
        a, b = P4_DIR / "orig" / name, P4_DIR / "r6" / name
        if R6.sha256(a) != R6.sha256(orig) or R6.sha256(orig) != rec["orig"][name]:
            sys.exit(f"STOP: {a} differs from their {orig} / the recorded sha256")
        if R6.sha256(b) != rec["patched"][name]:
            sys.exit(f"STOP: {b} differs from the patched file recorded with {PATCH.name}")
    return P4_DIR / "r6" / "hyp_tuning.py", P4_DIR / "r6" / "ml_analysis.py"


def run(cmd, wd, env_extra=None):
    wd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **(env_extra or {}))
    for k in ("R6_KEEP", "R6_KEEP_TEST"):
        if k not in (env_extra or {}):
            env.pop(k, None)
    with open(wd / "runner.log", "a") as fo:
        fo.write(f"\n### {time.strftime('%F %T')} {' '.join(map(str, cmd))} {env_extra or ''}\n")
        fo.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=wd, stdout=fo, stderr=subprocess.STDOUT, env=env)
    return r.returncode


def load_hps(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def ml_done(wd, key, seed):
    f = wd / "ml_results.pkl"
    if not f.is_file():
        return False
    ml = pd.read_pickle(f)
    row = ml[ml["model_target"] == key]
    return len(row) > 0 and int(seed) in [int(s) for s in row["random_state"].iloc[0]]


def run_ml(ml_src, ds, hps_path, wd, env_extra=None):
    hps = load_hps(hps_path)
    keys = [k for k in hps if "nn" in k and "sum_distortion_energies" not in k]
    for key in keys:
        for s in R6.SEEDS:
            if ml_done(wd, key, s):
                continue
            rc = run([sys.executable, ml_src, s, ds, hps_path], wd, env_extra)
            if not ml_done(wd, key, s):
                sys.exit(f"STOP: ml_analysis did not record {key} seed {s} (rc {rc}; {wd}/runner.log)")
            print(f"  ml {key} seed {s} ok", flush=True)


# ---------------------------------------------------------------- P-2
def cmd_p2_ml():
    _, ml_src = patched_code()
    run_ml(ml_src, R6.ESP_DS, R6.ESP_HPS, R6.SCRATCH / "P2" / "ml")
    return 0


def cmd_p2_hyp():
    hyp_src, _ = patched_code()
    wd = R6.SCRATCH / "P2" / "hyp"
    if not (wd / "hps.pkl").is_file():
        run([sys.executable, hyp_src, R6.ESP_DS], wd)
    return 0 if (wd / "hps.pkl").is_file() else 3


def _vals(cell):
    a = np.asarray(cell, dtype=object)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.reshape(-1)[0], dtype=object)
    return np.asarray(a, dtype=float).ravel()


def cmd_p2_nn():
    """r6_nn.fit_predict (the NN of our-pipeline arms) on their data with their stored hps, one fresh process per
    (NN key, seed), vs their ml_analysis.py rerun of P-1 -> results_rev6/P2_nn_reimpl.csv."""
    out = R6.SCRATCH / "P2" / "nn_reimpl"
    out.mkdir(parents=True, exist_ok=True)
    hps = load_hps(R6.ESP_HPS)
    keys = [k for k in hps if "nn" in k and "sum_distortion_energies" not in k]
    for key in keys:
        for s in R6.SEEDS:
            f = out / f"{key}__{s}.json"
            if f.is_file():
                continue
            r = subprocess.run([sys.executable, str(HERE / "r6_espc.py"), "_nn_one", key, str(s), str(f)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-3000:], file=sys.stderr)
                return 3
    mine = pd.read_pickle(R6.SCRATCH / "P1" / "ml" / "ml_results.pkl")
    rows = []
    for key in keys:
        m = mine[mine.model_target == key].iloc[0]
        rs = [int(x) for x in m.random_state]
        for s in R6.SEEDS:
            got = json.loads((out / f"{key}__{s}.json").read_text())
            ref = _vals(m.y_test_pred_values[rs.index(s)])
            d = np.abs(np.asarray(got["test"]) - ref)
            rows.append(dict(key=key, seed=s, max_absdiff=float(d.max()), median_absdiff=float(np.median(d))))
    df = pd.DataFrame(rows)
    R6.write_atomic(R6.RES6 / "P2_nn_reimpl.csv", lambda t: df.to_csv(t, index=False))
    print(df.describe().to_string())
    return 0


def _nn_one(key, seed, out):
    """Internal: their data, their split for `seed`, their stored hps -> r6_nn.fit_predict test predictions."""
    import r6_nn as NN
    NN.set_seeds(int(seed))
    esp = pd.read_pickle(R6.ESP_DS)
    rm = ["contributions", "reactant", "structure", "path", "reaction_number", "sum_distortion_energies"]
    df = esp.drop(columns=[c for c in esp.columns if any(t in c for t in rm)])
    ycols = [c for c in df if "_dft" in c]
    X, Y = df.drop(columns=ycols), df[ycols]
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=int(seed))
    Xte, Xva, yte, yva = train_test_split(Xte, yte, test_size=0.5, random_state=int(seed))
    pre, target = split_key(key)
    hps = load_hps(R6.ESP_HPS)[key]
    pr, _ = NN.fit_predict(Xtr, ytr[target], Xva, yva[target], {"test": Xte}, int(pre[0]), hps, target)
    R6.write_json(out, dict(key=key, seed=int(seed), test=[float(v) for v in pr["test"]]))


def cmd_p2_compare():
    rows = []
    a = pd.read_pickle(R6.SCRATCH / "P1" / "ml" / "ml_results.pkl")
    b = pd.read_pickle(R6.SCRATCH / "P2" / "ml" / "ml_results.pkl")
    for _, r in a.iterrows():
        m = b[b.model_target == r.model_target]
        for i, s in enumerate(r.random_state):
            if not len(m):
                rows.append(dict(key=r.model_target, seed=int(s), max_absdiff=np.inf)); continue
            j = [int(x) for x in m.random_state.iloc[0]].index(int(s))
            d = np.abs(_vals(r.y_test_pred_values[i]) - _vals(m.y_test_pred_values.iloc[0][j]))
            rows.append(dict(key=r.model_target, seed=int(s), max_absdiff=float(d.max())))
    df = pd.DataFrame(rows)
    R6.write_atomic(R6.RES6 / "P2_protocol_vs_theirs.csv", lambda t: df.to_csv(t, index=False))
    h1, h2 = load_hps(R6.SCRATCH / "P1" / "hyp" / "hps.pkl"), load_hps(R6.SCRATCH / "P2" / "hyp" / "hps.pkl")
    hdiff = [k for k in h1 if json.dumps(h1[k], sort_keys=True, default=str) != json.dumps(h2.get(k), sort_keys=True,
                                                                                            default=str)]
    nn_re = pd.read_csv(R6.RES6 / "P2_nn_reimpl.csv") if (R6.RES6 / "P2_nn_reimpl.csv").is_file() else None
    rep = dict(ml_max_absdiff=float(df.max_absdiff.max()), ml_n=len(df), ml_ok=bool((df.max_absdiff <= TOL_P2).all()),
               ml_sklearn_max=float(df[~df.key.str.contains("nn")].max_absdiff.max()),
               ml_nn_max=float(df[df.key.str.contains("nn")].max_absdiff.max()),
               hps_common_keys_differ=hdiff, hps_extra_keys_in_patched=sorted(set(h2) - set(h1)),
               nn_reimpl_max_absdiff=None if nn_re is None else float(nn_re.max_absdiff.max()),
               nn_reimpl_ok=None if nn_re is None else bool((nn_re.max_absdiff <= TOL_P2).all()), tol=TOL_P2)
    rep["stop"] = not (rep["ml_ok"] and not hdiff and (rep["nn_reimpl_ok"] in (True,)))
    R6.write_json(R6.RES6 / "P2_protocol_check.json", rep)
    print(json.dumps(rep, indent=1))
    return 3 if rep["stop"] else 0


# ---------------------------------------------------------------- arms
def build_dataset(arm, wd):
    cfg = ARM_CFG[arm]
    S = pd.read_csv(R6.RES6 / "S_rows.csv")                         # ds3 order, pos 0..3509
    Sj = json.loads((R6.RES6 / "S.json").read_text())
    ids = S.rxn_id.to_numpy()
    if cfg["feat"] == R6.FEAT_ESP:
        X = pd.read_parquet(R6.S_X_ESP).set_index("rxn_id").loc[ids, Sj["esp_cols_47"]]
    else:
        X = pd.read_parquet(R6.X182_ALL).set_index("rxn_id").reindex(ids)[Sj["feature_cols_182"]]
    Y = pd.read_parquet(R6.S_TARGETS).set_index("rxn_id").reindex(ids)
    ds = pd.DataFrame({"reaction_number": ids})
    for c in X.columns:
        ds[c] = X[c].to_numpy()
    ds["e_barrier_dft"] = Y["dft_barrier_kcal"].to_numpy()
    ds["sum_distortion_energies_dft"] = (Y["dft_d1_kcal"] + Y["dft_d2_kcal"]).to_numpy()
    ds["interaction_energies_dft"] = Y["dft_eint_spe_kcal"].to_numpy()
    ds["distortion_energy_1_dft"] = Y["dft_d1_kcal"].to_numpy()
    ds["distortion_energy_2_dft"] = Y["dft_d2_kcal"].to_numpy()
    ds.index = pd.RangeIndex(len(ds))
    keep = S.index[S[cfg["keep"]]].tolist()
    keep_t = S.index[np.logical_or.reduce([S[c].to_numpy() for c in cfg["keep_test"]])].tolist()
    assert not ds.loc[keep, [c for c in ds if c.endswith("_dft")]].isna().any().any()
    assert not ds.loc[keep_t, list(X.columns)].isna().any().any()
    wd.mkdir(parents=True, exist_ok=True)
    p = wd / f"r6_{arm.replace('-', '').lower()}.pkl"
    if not p.is_file():
        R6.write_atomic(p, lambda t: ds.to_pickle(t))
    R6.write_atomic(wd / "keep.txt", "\n".join(map(str, keep)) + "\n")
    R6.write_atomic(wd / "keep_test.txt", "\n".join(map(str, keep_t)) + "\n")
    return p, ds, keep, keep_t


def cmd_arm(arm):
    R6.gate_prereg()
    cfg = ARM_CFG[arm]
    hyp_src, ml_src = patched_code()
    wd = ESPC / arm
    ds_path, ds, keep, keep_t = build_dataset(arm, wd)
    env = dict(R6_KEEP=str(wd / "keep.txt"), R6_KEEP_TEST=str(wd / "keep_test.txt"))
    if not (wd / "hyp" / "hps.pkl").is_file():
        rc = run([sys.executable, hyp_src, ds_path], wd / "hyp", env)
        if rc != 0 or not (wd / "hyp" / "hps.pkl").is_file():
            sys.exit(f"STOP: hyp_tuning failed for {arm} (rc {rc})")
    print(f"[{arm}] tuned: {sorted(load_hps(wd / 'hyp' / 'hps.pkl'))}", flush=True)
    run_ml(ml_src, ds_path, wd / "hyp" / "hps.pkl", wd / "ml", env)
    return collect(arm, cfg, wd, ds, keep, keep_t)


def nn_cv_tune_rows(arm, wd, ds, keep, key, hps):
    """Representative-rule validation MAE of an Espley-protocol NN: KFold(5) (no shuffle, their cv=5) on the
    seed-23 train ∩ keep rows in split order, tuned hps, fresh process seeded 23."""
    out = wd / "nn_cv" / f"{key}.json"
    if out.is_file():
        return json.loads(out.read_text())["val_mae"]
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(HERE / "r6_espc.py"), "_nn_cv", arm, key, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"STOP: NN CV failed for {arm} {key}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())["val_mae"]


def _nn_cv(arm, key, out):
    import r6_nn as NN
    from sklearn.model_selection import KFold
    NN.set_seeds(R6.TUNE_SEED)
    wd = ESPC / arm
    ds = pd.read_pickle(next(wd.glob("r6_*.pkl")))
    keep = set(int(l) for l in open(wd / "keep.txt") if l.strip())
    tr, _, _ = R6.espley_split3(len(ds), R6.TUNE_SEED)
    tr = np.array([p for p in tr if p in keep])
    pre, target = split_key(key)
    xcols = [c for c in ds.columns if c != "reaction_number" and "_dft" not in c
             and "sum_distortion_energies" not in c]
    hps = load_hps(wd / "hyp" / "hps.pkl")[key]
    errs = []
    for a, b in KFold(5).split(tr):
        pr, _ = NN.fit_predict(ds.loc[tr[a], xcols], ds.loc[tr[a], target], ds.loc[tr[b], xcols], ds.loc[tr[b], target],
                               {"held": ds.loc[tr[b], xcols]}, int(pre[0]), hps, target)
        errs.append(float(np.mean(np.abs(pr["held"] - ds.loc[tr[b], target].to_numpy(float)))))
    R6.write_json(out, dict(key=key, cv_fold_mae=errs, val_mae=float(np.mean(errs)), n=len(tr)))


def collect(arm, cfg, wd, ds, keep, keep_t):
    import r6_train as TR
    ml = pd.read_pickle(wd / "ml" / "ml_results.pkl")
    hps = load_hps(wd / "hyp" / "hps.pkl")
    S = pd.read_csv(R6.RES6 / "S_rows.csv")
    ids = S.rxn_id.to_numpy()
    inC = S[cfg["keep"]].to_numpy()
    keep_t = set(keep_t)
    n_units, bias = 0, []
    for _, r in ml.iterrows():
        pre, tc = split_key(r.model_target)
        if tc not in T_OF:
            continue
        model, t = MODEL_OF[pre], T_OF[tc]
        val_mae = (nn_cv_tune_rows(arm, wd, ds, keep, r.model_target, hps[r.model_target]) if "nn" in pre
                   else float(hps[r.model_target]["train_mae"]))
        for i, s in enumerate(r.random_state):
            s = int(s)
            tr, te, va = R6.espley_split3(len(ds), s)
            te_k = [p for p in te if p in keep_t]
            yp, yt = _vals(r.y_test_pred_values[i]), _vals(r.y_test_true[i])
            assert len(yp) == len(te_k), (arm, r.model_target, s, len(yp), len(te_k))
            ours = ds.loc[te_k, tc].to_numpy(float)
            assert np.allclose(yt, ours, atol=1e-9, rtol=0), f"{arm} {r.model_target} seed {s}: y_test_true != ours"
            df = pd.DataFrame(dict(rxn_id=ids[te_k], y_true=ours, y_pred=yp, pos=te_k))
            df["inC"] = inC[te_k]
            main = df[df.inC]
            if "D" in cfg["keep_test"]:
                bias.append(df.assign(arm=arm, model=model, target=t, seed=s))
            u = dict(group=cfg["group"], arm=arm.replace("_all", ""), target=t, model=model, seed=s,
                     rowset=cfg["keep"], feat=cfg["feat"])
            pred = pd.DataFrame(dict(group=u["group"], arm=u["arm"], model=model, target=t, seed=s, rowset=u["rowset"],
                                     rxn_id=main.rxn_id, y_true=main.y_true, y_pred=main.y_pred))
            n_tr = int(sum(1 for p in tr if p in set(keep)))
            hp = hps[r.model_target]
            rec = dict(u, fingerprint=hashlib.sha256((R6.sha256(wd / "ml" / "ml_results.pkl") + R6.sha256(
                wd / "hyp" / "hps.pkl")).encode()).hexdigest(), n_train=n_tr, n_test=len(main),
                n_val=int(sum(1 for p in va if p in set(keep))), mae=float(np.mean(np.abs(main.y_pred - main.y_true))),
                nmae=R6.nmae(main.y_true, main.y_pred), val_mae=val_mae,
                val_source="nn_5fold_cv_seed23_train" if "nn" in pre else "their_gridsearch_best_score",
                their_y_train_cv=None if "nn" in pre else float(np.asarray(r.y_train_CV[i], float)),
                best_params=hp.get("b_hps", hp) if isinstance(hp, dict) else hp, edge_hits={},
                protocol="espley", code_sha256={k: R6.sha256(v) for k, v in zip(("hyp", "ml"), patched_code())},
                versions=TR.versions())
            TR.write_unit(u, rec, pred)
            n_units += 1
    if bias:
        b = pd.concat(bias)
        R6.write_atomic(R6.SCRATCH / "units" / "bias" / f"{arm}.parquet", lambda t: b.to_parquet(t, index=False))
    print(f"[{arm}] collected {n_units} units", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["_nn_one"]:
        _nn_one(a[1], a[2], Path(a[3])); sys.exit(0)
    if a[:1] == ["_nn_cv"]:
        _nn_cv(a[1], a[2], Path(a[3])); sys.exit(0)
    cmds = {"p2_ml": cmd_p2_ml, "p2_hyp": cmd_p2_hyp, "p2_nn": cmd_p2_nn, "p2_compare": cmd_p2_compare}
    if a[:1] == ["arm"] and len(a) == 2 and a[1] in ARM_CFG:
        sys.exit(cmd_arm(a[1]))
    if a[:1] and a[0] in cmds:
        sys.exit(cmds[a[0]]())
    raise SystemExit(__doc__)
