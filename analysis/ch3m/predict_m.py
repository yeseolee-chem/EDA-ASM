#!/usr/bin/env python3
r"""predict_m.py — CH3M 9: the predict_tf / predict_ch3 interface for the CH3M final model (TF or GNN; 10 targets or
strain / inter / solv / barrier). env_d1fu2 (GPU for I-1 / I-2); jobs/gpu_m.slurm. Exit 0 done, 2 not ready, 3 STOP.

  python predict_m.py check-i1 <candidate>   I-1: --model cv (the 25 fold .pt of the candidate in $CH3M/units) on every
                                             KM test pair vs results/KM_predictions.parquet rows of that rung / model
                                             (per fold, mean of 5 seeds) <= 1e-4 -> results/IM_selftest.json 'I-1'
  python predict_m.py check-i2 <candidate>   I-2: --model final ($CH3M/model) on the 1,000 W check pairs vs the
                                             predictions recorded right after W (same GPU) <= 1e-6 -> 'I-2'
  candidate: TF10 | GNN10 | GNN3

The deploy predict_tf module is used unchanged; this process binds it to the candidate (predict_tf.MODEL = model type,
predict_tf.TARGETS = the target list, fu2 target constants through data_ch3.use_targets, thresholds from
results/EM_ad.json / the model card). The .pt 'model' field must equal the candidate's model type.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402
import data_ch3 as DC3  # noqa: E402

DL, FT = DC3.DL, DC3.FT
import predict_tf as PT  # noqa: E402

I1_TOL, I2_TOL = PT.I1_TOL, PT.I2_TOL
I_DIR = C.CH3M / "I"


def parse(cand):
    if cand not in ("TF10", "GNN10", "GNN3"):
        C.stop(f"candidate {cand}?")
    mode = 10 if cand.endswith("10") else 3
    return cand[:-2] if mode == 10 else cand[:-1], mode


def bind(model, mode):
    tg = C.TARGETS10 if mode == 10 else C.TARGETS3
    DC3.use_targets(tg)
    PT.TARGETS = list(tg)
    PT.MODEL = model

    def _thresholds(m, model_dir=None):
        card = PT.card_path(model_dir or C.MODEL_DIR)
        if m == "final" and card is not None:
            th = (C.read_json(card).get("applicability") or {}).get("thresholds_yhat_sd")
            if th:
                return th, str(card)
        adp = C.RES / "EM_ad.json"
        if adp.is_file():
            th = (C.read_json(adp).get("thresholds") or {}).get(f"{model}{'10' if mode == 10 else '3'}")
            if th:
                return th, str(adp)
        return None, None
    PT.thresholds = _thresholds
    return tg


def _src():
    cfg = DL.SOURCES["lf"]
    return cfg["tokens"], cfg["x182"], Path(cfg["res"]) / "T_report.json"


def _merge(key, rec):
    p = C.RES / "IM_selftest.json"
    cur = C.read_json(p) if p.is_file() else {}
    cur[key] = rec
    cur["all_pass_I1_I2"] = bool(all((cur.get(k) or {}).get("ok") for k in ("I-1", "I-2")))
    C.write_json(p, cur)


def _write(out_path, out, excl, rep):
    C.write_parquet(out_path, out)
    s = out_path.with_name(out_path.name[:-len(".parquet")])
    C.write_csv(s.with_name(s.name + ".excluded.csv"), excl)
    C.write_json(s.with_name(s.name + ".report.json"), dict(rep, output=str(out_path), output_sha256=C.sha256(out_path)))


def cmd_check_i1(cand):
    model, mode = parse(cand)
    tg = bind(model, mode)
    rung = C.RUNGS[mode]
    kp = C.RES / "KM_predictions.parquet"
    if not kp.is_file():
        C.not_ready(f"{kp} missing")
    K = pd.read_parquet(kp)
    K = K[(K.rung == rung) & (K.model == model)]
    if set(K.target) != set(tg):
        C.stop(f"KM rows of {cand}: targets {sorted(set(K.target))}")
    D = DC3.get_data("lf" if mode == 10 else "ch3")
    DC3.use_targets(tg)
    tok, x182, trep = _src()
    ids = sorted(K.pair_id.unique())
    pc = I_DIR / f"i1_pairs_{cand}.csv"
    C.write_csv(pc, D.P.loc[ids, ["a_rxn", "b_rxn"]].reset_index())
    out, excl, rep = PT.run_predict(tok, x182, pc, "cv", rung, None, None, trep, C.UNITS, "direct")
    _write(I_DIR / f"i1_pred_{cand}.parquet", out, excl, rep)
    m = K[["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].merge(
        out[["pair_id", "fold", "target", "yhat", "yhat_sd"]], on=["pair_id", "fold", "target"], how="left",
        suffixes=("_k", "_p"))
    miss = int(m.yhat_p.isna().sum())
    d = np.abs(m.yhat_k - m.yhat_p).to_numpy(float)
    mx = float(np.nanmax(d)) if len(d) else float("nan")
    rec = dict(check="I-1", candidate=cand, ok=bool(miss == 0 and len(excl) == 0 and mx <= I1_TOL), tol=I1_TOL,
               max_abs_diff=mx, n_rows=int(len(m)), n_missing=miss, n_excluded_pairs=int(len(excl)),
               n_models=len(rep["models"]), gpu=FT.versions()["gpu"], km_predictions_sha256=C.sha256(kp), written=C.now())
    _merge("I-1", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 3


def cmd_check_i2(cand):
    model, mode = parse(cand)
    bind(model, mode)
    rung = C.RUNGS[mode]
    pc = C.MODEL_DIR / f"W_check_pairs_{rung}_{model}.csv"
    pp = C.MODEL_DIR / f"W_check_pred_{rung}_{model}.parquet"
    if not (pc.is_file() and pp.is_file()):
        C.not_ready(f"{pc} / {pp} missing (train_m.py final)")
    W = pd.read_parquet(pp)
    tok, x182, trep = _src()
    out, excl, rep = PT.run_predict(tok, x182, pc, "final", rung, None, None, trep, C.MODEL_DIR, "direct")
    _write(I_DIR / f"i2_pred_{cand}.parquet", out, excl, rep)
    m = W[["pair_id", "target", "yhat", "yhat_sd"]].merge(out[["pair_id", "target", "yhat", "yhat_sd"]],
                                                          on=["pair_id", "target"], how="left", suffixes=("_w", "_p"))
    miss = int(m.yhat_p.isna().sum())
    d = np.abs(m.yhat_w - m.yhat_p).to_numpy(float)
    mx = float(np.nanmax(d)) if len(d) else float("nan")
    wr = C.read_json(C.RES / "WM_report.json") if (C.RES / "WM_report.json").is_file() else {}
    gpu_now, gpu_w = FT.versions()["gpu"], (wr.get("versions") or {}).get("gpu")
    rec = dict(check="I-2", candidate=cand, ok=bool(miss == 0 and len(excl) == 0 and mx <= I2_TOL), tol=I2_TOL,
               max_abs_diff=mx, n_rows=int(len(m)), n_missing=miss, n_excluded_pairs=int(len(excl)), gpu=gpu_now,
               gpu_W=gpu_w, same_gpu=gpu_now == gpu_w, written=C.now())
    _merge("I-2", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check-i1"] and len(a) == 2:
        sys.exit(cmd_check_i1(a[1]))
    if a[:1] == ["check-i2"] and len(a) == 2:
        sys.exit(cmd_check_i2(a[1]))
    print(__doc__)
    sys.exit(2)
