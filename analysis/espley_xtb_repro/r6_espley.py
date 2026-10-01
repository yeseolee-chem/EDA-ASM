#!/usr/bin/env python3
"""r6_espley.py — rev 6 Phase P (docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md §6): Espley's code, run unchanged.

  P-1  (Espley-pinned env, env_espley)
    python r6_espley.py p1_ml        # their ml_analysis.py, unchanged, on their ds3 + their hps.pkl, seeds 22/23/14/1/2
    python r6_espley.py p1_hyp       # their hyp_tuning.py, unchanged, on their ds3
    python r6_espley.py p1_compare   # vs their stored ml_results.pkl / hps.pkl -> results_rev6/P1_*.{json,csv}
  P-3  (reactot)
    python r6_espley.py p3           # rev 4 swap vector -> role check vs our d1/d2 -> results_rev6/P3_*.{json,csv}

Their scripts write into the current directory (ml_results.pkl, hps.pkl, checkpoint.pkl, *.log, figures/); every run
gets its own work directory under $R6_SCRATCH/P1/. Their scripts are executed from their own files (sha256 recorded);
nothing is copied or edited. ml_analysis.py runs one NN per invocation (the first NN key of hps.pkl that has < 5 seeds)
plus every sklearn model not yet run with that seed, so p1_ml invokes it once per (NN key, seed) in hps.pkl order —
the sklearn models run in the first five invocations. Both runners are idempotent through their own checkpoints
(ml_results.pkl / checkpoint.pkl). Exit codes: 0 ok, 2 bad inputs, 3 STOP.
"""
from __future__ import annotations

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

P1 = R6.SCRATCH / "P1"
TOL_PRED = 1e-6                               # P-1 gate (kcal/mol)
SUM_TAG = "sum_distortion_energies"          # ml_analysis.py :336, :342-346 skip these targets


def espley_env_check():
    from importlib.metadata import version
    got = {k: version(k) for k in R6.ESP_PINS}
    bad = {k: (got[k], v) for k, v in R6.ESP_PINS.items() if got[k] != v}
    if bad:
        print(f"not the Espley-pinned env: {bad}", file=sys.stderr)
        sys.exit(2)
    return got


def load_hps(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def nn_keys(hps):
    """ml_analysis.py :322, :336 — NN keys in hps order, without the sum-of-distortions target."""
    return [k for k in hps if "nn" in k and SUM_TAG not in k]


def ml_done(wd, key, seed):
    f = wd / "ml_results.pkl"
    if not f.is_file():
        return False
    ml = pd.read_pickle(f)
    row = ml[ml["model_target"] == key]
    return len(row) > 0 and int(seed) in [int(s) for s in row["random_state"].iloc[0]]


def run_their(cmd, wd, log):
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(wd / log, "a") as fo:
        fo.write(f"\n### {time.strftime('%F %T')} {' '.join(map(str, cmd))}\n")
        fo.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=wd, stdout=fo, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def cmd_p1_ml():
    ver = espley_env_check()
    wd = P1 / "ml"
    hps = load_hps(R6.ESP_HPS)
    keys = nn_keys(hps)
    print(f"[p1_ml] {len(keys)} NN keys x {len(R6.SEEDS)} seeds; sklearn keys "
          f"{[k for k in hps if 'nn' not in k]}", flush=True)
    for key in keys:
        for s in R6.SEEDS:
            if ml_done(wd, key, s):
                continue
            rc, dt = run_their([sys.executable, R6.ESP_ML_SRC, s, R6.ESP_DS, R6.ESP_HPS], wd, "runner.log")
            ok = ml_done(wd, key, s)
            print(f"[p1_ml] {key} seed {s}: rc {rc}, {dt / 60:.1f} min, recorded {ok}", flush=True)
            if not ok:
                print(f"STOP: their ml_analysis.py did not record {key} seed {s} (see {wd}/runner.log)",
                      file=sys.stderr)
                return 3
    R6.write_json(wd / "p1_ml_done.json", dict(versions=ver, ml_src_sha256=R6.sha256(R6.ESP_ML_SRC),
                                               ds_sha256=R6.sha256(R6.ESP_DS), hps_sha256=R6.sha256(R6.ESP_HPS)))
    return 0


def cmd_p1_hyp():
    ver = espley_env_check()
    wd = P1 / "hyp"
    if (wd / "hps.pkl").is_file():
        print(f"{wd}/hps.pkl exists - skip"); return 0
    rc, dt = run_their([sys.executable, R6.HYP_SRC, R6.ESP_DS], wd, "runner.log")
    print(f"[p1_hyp] rc {rc}, {dt / 3600:.2f} h", flush=True)
    if rc != 0 or not (wd / "hps.pkl").is_file():
        print(f"STOP: their hyp_tuning.py did not write hps.pkl (see {wd}/runner.log)", file=sys.stderr)
        return 3
    R6.write_json(wd / "p1_hyp_done.json", dict(versions=ver, hyp_src_sha256=R6.sha256(R6.HYP_SRC),
                                                ds_sha256=R6.sha256(R6.ESP_DS), hours=dt / 3600))
    return 0


def _vals(cell):
    """One stored per-seed prediction list as a flat float array (NN cells are [[...]])."""
    a = np.asarray(cell, dtype=object)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.reshape(-1)[0], dtype=object)
    return np.asarray(a, dtype=float).ravel()


def _same_hp(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same_hp(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def cmd_p1_compare():
    espley_env_check()
    stored = pd.read_pickle(R6.ESP_RES)
    mine = pd.read_pickle(P1 / "ml" / "ml_results.pkl")
    rows = []
    for _, r in stored.iterrows():
        key = r["model_target"]
        m = mine[mine["model_target"] == key]
        for i, s in enumerate(r["random_state"]):
            rec = dict(model_target=key, seed=int(s), nn="nn" in key)
            if len(m) == 0 or int(s) not in [int(x) for x in m["random_state"].iloc[0]]:
                rec.update(status="missing"); rows.append(rec); continue
            j = [int(x) for x in m["random_state"].iloc[0]].index(int(s))
            for what in ("y_test_pred_values", "y_val_pred_values"):
                a, b = _vals(r[what][i]), _vals(m[what].iloc[0][j])
                if a.shape != b.shape:
                    rec[f"{what}_maxdiff"] = np.inf
                    continue
                d = np.abs(a - b)
                rec[f"{what}_maxdiff"] = float(d.max())
                if what == "y_test_pred_values":
                    rec.update(n=len(a), test_median_absdiff=float(np.median(d)), test_p90_absdiff=float(np.quantile(d, .9)),
                               test_mae_stored=float(np.mean(np.abs(a - _vals(r["y_test_true"][i])))),
                               test_mae_rerun=float(np.mean(np.abs(b - _vals(m["y_test_true"].iloc[0][j])))))
            rec["y_test_true_same"] = bool(np.array_equal(_vals(r["y_test_true"][i]), _vals(m["y_test_true"].iloc[0][j])))
            rec["status"] = "ok" if rec["y_test_pred_values_maxdiff"] <= TOL_PRED else "differs"
            rows.append(rec)
    df = pd.DataFrame(rows)
    R6.write_atomic(R6.RES6 / "P1_pred_diffs.csv", lambda t: df.to_csv(t, index=False))
    hp_rows, hyp_f = [], P1 / "hyp" / "hps.pkl"
    hs = load_hps(R6.ESP_HPS)
    # their script writes hps.pkl only at the very end; when the NN stage crashed, the sklearn results it already
    # checkpointed (checkpoint.pkl, same dict format) are compared
    ck = P1 / "hyp" / "checkpoint.pkl"
    hm = load_hps(hyp_f) if hyp_f.is_file() else (load_hps(ck) if ck.is_file() else {})
    hyp_source = "hps.pkl" if hyp_f.is_file() else ("checkpoint.pkl (NN stage did not finish)" if ck.is_file() else None)
    for k, v in hs.items():
        got = hm.get(k)
        sv = v.get("b_hps", v) if isinstance(v, dict) else v
        gv = (got.get("b_hps", got) if isinstance(got, dict) else got) if got is not None else None
        hp_rows.append(dict(key=k, nn="nn" in k, stored=json.dumps(sv, default=str), rerun=json.dumps(gv, default=str),
                            same=got is not None and _same_hp(sv, gv),
                            stored_train_mae=v.get("train_mae") if isinstance(v, dict) else None,
                            rerun_train_mae=got.get("train_mae") if isinstance(got, dict) else None))
    hp = pd.DataFrame(hp_rows)
    R6.write_atomic(R6.RES6 / "P1_hps_compare.csv", lambda t: hp.to_csv(t, index=False))
    sk, nn = df[~df.nn], df[df.nn]
    rep = dict(
        n_cells=len(df), missing=int((df.status == "missing").sum()),
        sklearn=dict(n=len(sk), n_ok=int((sk.status == "ok").sum()),
                     max_absdiff=float(sk.y_test_pred_values_maxdiff.max()) if len(sk) else None,
                     by_model={k: float(g.y_test_pred_values_maxdiff.max())
                               for k, g in sk.groupby(sk.model_target.str.split("_").str[0])}),
        nn=dict(n=len(nn), n_ok=int((nn.status == "ok").sum()),
                max_absdiff=float(nn.y_test_pred_values_maxdiff.max()) if len(nn) else None,
                median_of_median_absdiff=float(nn.test_median_absdiff.median()) if len(nn) else None,
                mae_stored_minus_rerun=dict(zip(nn.model_target + "@" + nn.seed.astype(str),
                                                (nn.test_mae_stored - nn.test_mae_rerun).round(4))) if len(nn) else {}),
        y_test_true_all_same=bool(df.y_test_true_same.fillna(False).all()),
        hps_source=hyp_source, hps_sklearn_same=bool(hp[~hp.nn].same.all()) if len(hm) else None,
        hps_nn_same=(bool(hp[hp.nn].same.all()) if any("nn" in k for k in hm) else None),
        hps_nn_tuning_ran=any("nn" in k for k in hm),
        hps_differ=hp.loc[~hp.same & (~hp.nn | any("nn" in k for k in hm)), "key"].tolist() if len(hm) else None,
        rerun_extra_keys=sorted(set(hm) - set(hs)))
    rep["stop_sklearn"] = bool(rep["sklearn"]["n_ok"] < rep["sklearn"]["n"] or rep["missing"] or
                               (len(hm) and not rep["hps_sklearn_same"]))
    rep["stop_nn_not_bitwise"] = bool(rep["nn"]["n_ok"] < rep["nn"]["n"] or rep["hps_nn_same"] is False)
    rep["stop_nn_tuning_cannot_run"] = bool(len(hm) and not rep["hps_nn_tuning_ran"])
    R6.write_json(R6.RES6 / "P1_reproduction.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop_sklearn"] or rep["stop_nn_not_bitwise"] or rep["stop_nn_tuning_cannot_run"] else 0


def cmd_p3():
    """P-3 (reactot): rev 4 swap vector -> their DFT distortions re-assigned to dipole / dipolarophile vs our d1/d2."""
    P = pickle.loads(R6.R4_PREP.read_bytes())
    esp = pd.read_pickle(R6.ESP_DS).reset_index(drop=True)
    ids = esp["reaction_number"].astype(int).to_numpy()
    assert np.array_equal(np.asarray(P["ids"]).astype(int), ids), "prep.pkl ids != ds3 order"
    lab = {int(d["rxn_id"]): d for d in json.loads(Path(R6.R5.LABELS).read_text())}
    sw = np.asarray(P["swap"], float)
    ok = ~np.isnan(sw)
    t1, t2 = esp["distortion_energy_1_dft"].to_numpy(float), esp["distortion_energy_2_dft"].to_numpy(float)
    dip, dpo = np.where(sw == 1, t2, t1), np.where(sw == 1, t1, t2)
    d1 = np.array([lab[i]["d1_kcal"] if ok[k] else np.nan for k, i in enumerate(ids)])
    d2 = np.array([lab[i]["d2_kcal"] if ok[k] else np.nan for k, i in enumerate(ids)])
    e1, e2 = np.abs(d1 - dip)[ok], np.abs(d2 - dpo)[ok]
    rep = dict(n=int(ok.sum()), n_swapped=int(np.nansum(sw)), swap_share=float(np.nansum(sw) / ok.sum()),
               d1_median_absdiff=float(np.median(e1)), d2_median_absdiff=float(np.median(e2)),
               d1_mae=float(e1.mean()), d2_mae=float(e2.mean()), d1_p90=float(np.quantile(e1, .9)),
               d2_p90=float(np.quantile(e2, .9)), gate_median_lt=0.5, prep_sha256=R6.sha256(R6.R4_PREP))
    rep["stop"] = not (rep["d1_median_absdiff"] < 0.5 and rep["d2_median_absdiff"] < 0.5)
    out = pd.DataFrame(dict(rxn_id=ids, swap=sw))
    R6.write_atomic(R6.RES6 / "P3_swap.csv", lambda t: out.to_csv(t, index=False))
    rep["swap_csv_sha256"] = R6.sha256(R6.RES6 / "P3_swap.csv")
    R6.write_json(R6.RES6 / "P3_role_check.json", rep)
    print(json.dumps(rep, indent=1))
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"p1_ml": cmd_p1_ml, "p1_hyp": cmd_p1_hyp, "p1_compare": cmd_p1_compare, "p3": cmd_p3}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
