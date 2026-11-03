#!/usr/bin/env python3
r"""i2_checks.py — DEPLOY-MODEL-W Phase I2 checks (docs/specs/DEPLOY_MODEL_W.md 13-2) with the extended predict_tf.py
(`python tests_w.py i2`; env_d1fu2 on the K / W GPU type, analysis/deploy/jobs/gpu_fu2.slurm). Exit 0 pass, 2 not
ready, 3 fail. Results -> results/w/I2_selftest.json, prediction files -> $DEPW/I2/.

  I2-0  the DEPLOY_MODEL I-1 / I-2 computations (predict_tf.cmd_check_i1 / cmd_check_i2 logic, written to $DEPW/I2
        instead of results/I_selftest.json) with the extended predict_tf.py and default options: same criteria
        (<= 1e-4 / <= 1e-6, nothing missing / excluded); the values are compared with results/I_selftest.json.
  I2-1  --model cv --model-dir $DEPW/units --rung lfW --readout direct on the lfW test pairs vs
        results/w/K_predictions_w.parquet (rung lfW, per fold, seed mean): <= 1e-4 kcal/mol
  I2-2  --readout graph (group_id = Phase B gid) on the test groups vs the Phase E graph values
        ($DEPW/eval/E_graph_pred.parquet): lfW and lf, <= 1e-4
  I2-3  (W2 done) --model final --model-dir $DEPW/model on W_check_pairs.csv vs W_check_pred.parquet: <= 1e-6 (same GPU)
  I2-4  DEPW_I3_DIR (the DEPLOY_MODEL I-3 chain outputs: tokens/, x182.parquet, pairs.csv with group_id) predicted with
        the adopted model and readout (results/w/D_decision.json): runs, format (columns, rows, finite, readout); pending
        (not failed) while DEPW_I3_DIR is unset.
DEPW_PREDICT_TF=<path> loads predict_tf from another file (developer dry runs before the patch is applied).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w_common as W  # noqa: E402

C = W.C
TOL1, TOL3 = 1e-4, 1e-6
FOLDS = [int(x) for x in os.environ.get("DEPW_I2_FOLDS", "0,1,2,3,4").split(",")]   # developer runs: fewer folds


def predict_module():
    p = os.environ.get("DEPW_PREDICT_TF")
    if p:
        spec = importlib.util.spec_from_file_location("predict_tf_ext", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import predict_tf as mod
    if not hasattr(mod, "graph_readout") or not hasattr(mod, "card_path"):
        W.not_ready("predict_tf.py is not the extended version (apply analysis/deploy/w/predict_tf_ext.patch)")
    return mod


def _cmp(ref, out, keys, tol, a="yhat", b="yhat"):
    import numpy as np
    m = ref[keys + [a]].merge(out[keys + [b] + (["readout"] if "readout" in out.columns else [])], on=keys, how="left",
                              suffixes=("_ref", "_p"))
    ca, cb = (f"{a}_ref", f"{b}_p") if a == b else (a, b)
    miss = int(m[cb].isna().sum())
    d = np.abs(m[ca] - m[cb]).to_numpy(float)
    mx = float(np.nanmax(d)) if len(d) else float("nan")
    return dict(ok=bool(miss == 0 and mx <= tol), tol=tol, max_abs_diff=mx, n_rows=int(len(m)), n_missing=miss,
                readout=sorted(set(m.readout.dropna())) if "readout" in m.columns else None)


def i2_0(PT, source="lf"):
    import pandas as pd
    import data_lf as DL
    cfg = DL.SOURCES[source]
    tr = Path(cfg["res"]) / "T_report.json"
    out = {}
    kp = C.RES / "K_predictions.parquet"
    prev = W.read_json(C.RES / "I_selftest.json") if (C.RES / "I_selftest.json").is_file() else {}
    if kp.is_file():
        K = pd.read_parquet(kp)
        K = K[K.fold.isin(FOLDS)]
        D = DL.get_data(source)
        pc = W.I2_DIR / "i1_pairs.csv"
        W.write_csv(pc, D.P.loc[sorted(K.pair_id.unique()), ["a_rxn", "b_rxn"]].reset_index())
        o, e, rep = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "cv", C.RUNG, folds=FOLDS, t_report=tr)
        PT.write_outputs(W.assert_write_ok(W.I2_DIR / "i1_pred.parquet"), o, e, rep)
        r = _cmp(K, o, ["pair_id", "fold", "target"], PT.I1_TOL)
        r.update(ok=bool(r["ok"] and len(e) == 0), n_excluded=int(len(e)), readout=sorted(set(o.readout)),
                 previous=(prev.get("I-1") or {}).get("max_abs_diff"))
        out["I-1"] = r
    else:
        out["I-1"] = dict(ok=False, error=f"{kp} missing")
    pc, pp = C.MODEL_DIR / "W_check_pairs.csv", C.MODEL_DIR / "W_check_pred.parquet"
    if pc.is_file() and pp.is_file():
        Wc = pd.read_parquet(pp)
        o, e, rep = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "final", C.RUNG, t_report=tr)
        PT.write_outputs(W.assert_write_ok(W.I2_DIR / "i2_pred.parquet"), o, e, rep)
        r = _cmp(Wc, o, ["pair_id", "target"], PT.I2_TOL)
        r.update(ok=bool(r["ok"] and len(e) == 0), n_excluded=int(len(e)), readout=sorted(set(o.readout)),
                 previous=(prev.get("I-2") or {}).get("max_abs_diff"), gpu=PT.FT.versions()["gpu"])
        out["I-2"] = r
    else:
        out["I-2"] = dict(ok=False, error=f"{pc} missing (DEPLOY_MODEL W)")
    return dict(ok=bool(out["I-1"]["ok"] and out["I-2"]["ok"]), **out)


def _group_pairs(source="lf"):
    import pandas as pd
    import within as WI
    T = pd.concat([WI.load_test_groups(k, source=source).assign(fold=k) for k in range(W.NF)], ignore_index=True)
    return T


def i2_1(PT, source="lf"):
    import pandas as pd
    import data_lf as DL
    cfg = DL.SOURCES[source]
    KW = pd.read_parquet(W.RESW / "K_predictions_w.parquet")
    K = KW[(KW.rung == "lfW") & KW.fold.isin(FOLDS)]
    D = DL.get_data(source)
    pc = W.I2_DIR / "i2_1_pairs.csv"
    W.write_csv(pc, D.P.loc[sorted(K.pair_id.unique()), ["a_rxn", "b_rxn"]].reset_index())
    o, e, rep = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "cv", "lfW", folds=FOLDS, t_report=Path(cfg["res"]) / "T_report.json",
                               model_dir=W.UNITS_W, readout="direct")
    PT.write_outputs(W.assert_write_ok(W.I2_DIR / "i2_1_pred.parquet"), o, e, rep)
    r = _cmp(K, o, ["pair_id", "fold", "target"], TOL1)
    return dict(r, ok=bool(r["ok"] and len(e) == 0), n_excluded=int(len(e)))


def i2_2(PT, source="lf"):
    import pandas as pd
    import data_lf as DL
    cfg = DL.SOURCES[source]
    G = pd.read_parquet(W.EVAL_W / "E_graph_pred.parquet")
    T = _group_pairs(source)
    T = T[T.fold.isin(FOLDS)]
    pc = W.I2_DIR / "i2_2_pairs.csv"
    W.write_csv(pc, T.assign(group_id=T.gid)[["pair_id", "a_rxn", "b_rxn", "group_id"]])
    out = {}
    for m, mdir in (("lfW", W.UNITS_W), ("lf", C.UNITS)):
        o, e, rep = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "cv", m, folds=FOLDS, t_report=Path(cfg["res"]) / "T_report.json",
                                   model_dir=mdir, readout="graph")
        PT.write_outputs(W.assert_write_ok(W.I2_DIR / f"i2_2_{m}_pred.parquet"), o, e, rep)
        ref = G[(G.model == m) & G.fold.isin(FOLDS)][["pair_id", "fold", "target", "yhat"]]
        r = _cmp(ref, o, ["pair_id", "fold", "target"], TOL1)
        out[m] = dict(r, ok=bool(r["ok"] and len(e) == 0 and r["readout"] == ["graph"]), n_excluded=int(len(e)),
                      fallback=(rep.get("graph") or {}).get("fallback"))
    return dict(ok=bool(all(v["ok"] for v in out.values())), **out)


def i2_3(PT, source="lf"):
    import pandas as pd
    import data_lf as DL
    pc, pp = W.MODEL_W / "W_check_pairs.csv", W.MODEL_W / "W_check_pred.parquet"
    if not (pc.is_file() and pp.is_file()):
        return dict(ok=None, status="not run (no W2)")
    cfg = DL.SOURCES[source]
    o, e, rep = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "final", t_report=Path(cfg["res"]) / "T_report.json",
                               model_dir=W.MODEL_W, readout="direct")
    PT.write_outputs(W.assert_write_ok(W.I2_DIR / "i2_3_pred.parquet"), o, e, rep)
    r = _cmp(pd.read_parquet(pp), o, ["pair_id", "target"], TOL3)
    w2 = W.read_json(W.RESW / "W2_report.json") if (W.RESW / "W2_report.json").is_file() else {}
    gpu_now, gpu_w = PT.FT.versions()["gpu"], (w2.get("versions") or {}).get("gpu")
    return dict(r, ok=bool(r["ok"] and len(e) == 0), rung=rep["rung"], gpu=gpu_now, gpu_W2=gpu_w, same_gpu=gpu_now == gpu_w)


def i2_4(PT):
    import numpy as np
    import pandas as pd
    d = os.environ.get("DEPW_I3_DIR")
    if not d:
        return dict(ok=None, status="pending: DEPW_I3_DIR (DEPLOY_MODEL I-3 chain outputs) not given")
    d = Path(d)
    need = [d / "tokens", d / "x182.parquet", d / "pairs.csv"]
    if not all(p.exists() for p in need):
        return dict(ok=None, status=f"pending: {[str(p) for p in need if not p.exists()]} missing")
    dec = W.read_json(W.RESW / "D_decision.json")
    mdir = W.MODEL_W if dec["adopted"] == "lfW" else C.MODEL_DIR
    o, e, rep = PT.run_predict(d / "tokens", d / "x182.parquet", d / "pairs.csv", "final", model_dir=mdir,
                               readout=dec["readout"], t_report=(d / "T_report.json") if (d / "T_report.json").is_file()
                               else None)
    PT.write_outputs(W.assert_write_ok(W.I2_DIR / "i2_4_pred.parquet"), o, e, rep)
    P = pd.read_csv(d / "pairs.csv", dtype={"pair_id": str})
    cols_ok = list(o.columns) == PT.OUT_COLS + ["readout"]
    n_ok = len(o) == (len(P) - len(e)) * len(PT.TARGETS)
    fin = bool(np.isfinite(o.yhat.to_numpy(float)).all())
    return dict(ok=bool(cols_ok and n_ok and fin), columns_ok=cols_ok, rows_ok=n_ok, finite=fin, n_pairs=int(len(P)),
                n_excluded=int(len(e)), readout=sorted(set(o.readout)), adopted=dec["adopted"], model_dir=str(mdir))


def main():
    import data_lf as DL
    t0 = time.time()
    PT = predict_module()
    W.assert_write_ok(W.I2_DIR / "probe")
    W.I2_DIR.mkdir(parents=True, exist_ok=True)
    src = os.environ.get("DEPW_TEST_SOURCE", "lf")
    rec = {}
    for k, fn in (("I2-0", lambda: i2_0(PT, src)), ("I2-1", lambda: i2_1(PT, src)), ("I2-2", lambda: i2_2(PT, src)),
                  ("I2-3", lambda: i2_3(PT, src)), ("I2-4", lambda: i2_4(PT))):
        try:
            rec[k] = fn()
        except SystemExit as e:
            rec[k] = dict(ok=False, error=f"SystemExit {e.code}")
    need = ["I2-0", "I2-1", "I2-2"] + (["I2-3"] if rec["I2-3"].get("ok") is not None else [])
    out = dict(all_pass=bool(all(rec[k].get("ok") for k in need)), required=need, checks=rec,
               predict_tf_sha256=W.sha256(getattr(PT, "__file__", "")), versions=DL.FT.versions(),
               seconds=time.time() - t0, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    W.write_json(W.RESW / "I2_selftest.json", out)
    print(json.dumps(W._jsonable(dict(all_pass=out["all_pass"], checks={k: v.get("ok") for k, v in rec.items()})), indent=1))
    return 0 if out["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
