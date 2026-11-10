#!/usr/bin/env python3
r"""predict_ch3.py — CH3-MODEL 11: the deploy predict_tf.py interface for the 3-channel model (4 outputs:
strain, inter, solv, barrier). env_d1fu2 (GPU for I-1 / I-2; CPU works too); jobs/gpu_ch3.slurm.
Exit 0 done, 2 not ready / usage, 3 STOP.

  python predict_ch3.py run <tokens_root> <x182.parquet> <pairs.csv> <out.parquet> [--model final|cv] [--folds ..]
                            [--seeds ..] [--t-report T_report.json] [--model-dir <dir>]
        = predict_tf.py run with: model files D__lf3__S__TF__{all__s<j>|f<k>__s<j>}.pt in $CH3/model (final) or
        $CH3/units (cv), TARGETS = (strain, inter, solv, barrier), applicability thresholds = MODEL_CARD.json (final)
        or results/E3_ad.json; direct readout only. Outputs as predict_tf (out.parquet + .excluded.csv + .report.json).
  python predict_ch3.py check-i1   I-1: --model cv on every K3 test pair vs results/K3_predictions.parquet (per fold,
                                   mean of 5 seeds), <= 1e-4 kcal/mol -> results/I3_selftest.json 'I-1'
  python predict_ch3.py check-i2   I-2: --model final on $CH3/model/W_check_pairs.csv (1,000 training pairs) vs
                                   W_check_pred.parquet (recorded right after W, same GPU), <= 1e-6 -> 'I-2'

Implementation: the deploy predict_tf module is imported and used unchanged (model loading / input checks / Std of each
.pt / fu2 predict x s_y); only its module settings are bound to ch3 in this process (spec 5: deploy files not edited):
predict_tf.TARGETS = TARGETS3 (load_models compares the .pt TARGETS and builds 4-output models), the fu2 target
constants via data_ch3.use_targets, the default model folders and thresholds (predict_tf.thresholds -> ch3 version).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402
import data_ch3 as DC3  # noqa: E402

DL, FT = DC3.DL, DC3.FT
import predict_tf as PT  # noqa: E402

TARGETS3 = list(C3.TARGETS3)
I1_TOL, I2_TOL = PT.I1_TOL, PT.I2_TOL
I_DIR = C3.CH3 / "I"


def _thresholds(mode, model_dir=None):
    """ch3 applicability thresholds: final -> MODEL_CARD applicability (fallback results/E3_ad.json); cv -> E3_ad."""
    card = PT.card_path(model_dir or C3.MODEL_DIR)
    if mode == "final" and card is not None:
        th = (C3.read_json(card).get("applicability") or {}).get("thresholds_yhat_sd")
        if th:
            return th, str(card)
    adp = C3.RES / "E3_ad.json"
    if adp.is_file():
        return C3.read_json(adp).get("thresholds"), str(adp)
    return None, None


def bind():
    """bind the deploy predictor to ch3 (this process only)."""
    DC3.use_targets(TARGETS3)
    PT.TARGETS = list(TARGETS3)
    PT.thresholds = _thresholds
    return DC3.current_mode()


def run_predict(tokens_root, x182, pairs_csv, mode="final", folds=None, seeds=None, t_report=None, model_dir=None):
    bind()
    if model_dir is None:
        model_dir = C3.MODEL_DIR if mode == "final" else C3.UNITS
    out, excl, rep = PT.run_predict(tokens_root, x182, pairs_csv, mode, C3.RUNG3, folds, seeds, t_report,
                                    model_dir, "direct")
    if set(out.target) - set(TARGETS3):
        C3.stop(f"predicted targets {sorted(set(out.target))} != {TARGETS3}")
    rep["targets"] = TARGETS3
    return out, excl, rep


def _stem(out):
    out = Path(out)
    return out.with_name(out.name[:-len(".parquet")] if out.name.endswith(".parquet") else out.stem)


def write_outputs(out_path, out, excl, rep):
    C3.write_parquet(out_path, out)
    s = _stem(out_path)
    C3.write_csv(s.with_name(s.name + ".excluded.csv"), excl)
    C3.write_json(s.with_name(s.name + ".report.json"), dict(rep, output=str(out_path),
                                                             output_sha256=C3.sha256(out_path)))


def cmd_run(a):
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
    out, excl, rep = run_predict(a.tokens_root, a.x182, a.pairs, a.model, folds, seeds, a.t_report, a.model_dir)
    write_outputs(a.out, out, excl, rep)
    print(f"[predict] {rep['pairs']['n_predicted']} pairs predicted, {rep['pairs']['n_excluded']} excluded, "
          f"{len(rep['models'])} models -> {a.out}", flush=True)
    return 0


def _merge_selftest(key, rec):
    p = C3.RES / "I3_selftest.json"
    cur = C3.read_json(p) if p.is_file() else {}
    cur[key] = rec
    cur["all_pass_I1_I2"] = bool(all((cur.get(k) or {}).get("ok") for k in ("I-1", "I-2")))
    C3.write_json(p, cur)


def _src():
    cfg = DL.SOURCES["lf"]
    return cfg["tokens"], cfg["x182"], Path(cfg["res"]) / "T_report.json"


def cmd_check_i1():
    kp = C3.RES / "K3_predictions.parquet"
    if not kp.is_file():
        C3.not_ready(f"{kp} missing (train_ch3.py collect)")
    K = pd.read_parquet(kp)
    D = DC3.get_data("ch3")
    tok, x182, trep = _src()
    ids = sorted(K.pair_id.unique())
    P = D.P.loc[ids, ["a_rxn", "b_rxn"]].reset_index()
    pc = I_DIR / "i1_pairs.csv"
    C3.write_csv(pc, P)
    out, excl, rep = run_predict(tok, x182, pc, "cv", t_report=trep)
    write_outputs(I_DIR / "i1_pred.parquet", out, excl, rep)
    m = K[["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].merge(
        out[["pair_id", "fold", "target", "yhat", "yhat_sd"]], on=["pair_id", "fold", "target"], how="left",
        suffixes=("_k", "_p"))
    miss = int(m.yhat_p.isna().sum())
    d = np.abs(m.yhat_k - m.yhat_p).to_numpy(float)
    dsd = np.abs(m.yhat_seed_sd - m.yhat_sd).to_numpy(float)
    mx = float(np.nanmax(d)) if len(d) else float("nan")
    rec = dict(check="I-1", ok=bool(miss == 0 and len(excl) == 0 and mx <= I1_TOL), tol=I1_TOL, max_abs_diff=mx,
               max_abs_diff_sd=float(np.nanmax(dsd)) if len(dsd) else None, n_rows=int(len(m)), n_missing=miss,
               n_excluded_pairs=int(len(excl)),
               per_fold_max={int(k): float(np.nanmax(g)) for k, g in pd.Series(d).groupby(m.fold.to_numpy())},
               per_target_max={t: float(np.nanmax(g)) for t, g in pd.Series(d).groupby(m.target.to_numpy())},
               k3_predictions_sha256=C3.sha256(kp), n_models=len(rep["models"]), gpu=FT.versions()["gpu"],
               written=C3.now())
    _merge_selftest("I-1", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 3


def cmd_check_i2():
    pc, pp = C3.MODEL_DIR / "W_check_pairs.csv", C3.MODEL_DIR / "W_check_pred.parquet"
    if not (pc.is_file() and pp.is_file()):
        C3.not_ready(f"{pc} / {pp} missing (train_ch3.py final)")
    W = pd.read_parquet(pp)
    tok, x182, trep = _src()
    out, excl, rep = run_predict(tok, x182, pc, "final", t_report=trep)
    write_outputs(I_DIR / "i2_pred.parquet", out, excl, rep)
    m = W[["pair_id", "target", "yhat", "yhat_sd"]].merge(out[["pair_id", "target", "yhat", "yhat_sd"]],
                                                          on=["pair_id", "target"], how="left", suffixes=("_w", "_p"))
    miss = int(m.yhat_p.isna().sum())
    d = np.abs(m.yhat_w - m.yhat_p).to_numpy(float)
    mx = float(np.nanmax(d)) if len(d) else float("nan")
    wr = C3.read_json(C3.RES / "W3_report.json") if (C3.RES / "W3_report.json").is_file() else {}
    gpu_now, gpu_w = FT.versions()["gpu"], (wr.get("versions") or {}).get("gpu")
    rec = dict(check="I-2", ok=bool(miss == 0 and len(excl) == 0 and mx <= I2_TOL), tol=I2_TOL, max_abs_diff=mx,
               max_abs_diff_sd=float(np.nanmax(np.abs(m.yhat_sd_w - m.yhat_sd_p))), n_rows=int(len(m)),
               n_missing=miss, n_excluded_pairs=int(len(excl)), gpu=gpu_now, gpu_W=gpu_w, same_gpu=gpu_now == gpu_w,
               w_check_pred_sha256=C3.sha256(pp), written=C3.now())
    _merge_selftest("I-2", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 3


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if a == ["check-i1"]:
        return cmd_check_i1()
    if a == ["check-i2"]:
        return cmd_check_i2()
    if a[:1] == ["run"]:
        ap = argparse.ArgumentParser(prog="predict_ch3.py run")
        ap.add_argument("tokens_root")
        ap.add_argument("x182")
        ap.add_argument("pairs")
        ap.add_argument("out")
        ap.add_argument("--model", choices=("final", "cv"), default="final")
        ap.add_argument("--folds", default=None)
        ap.add_argument("--seeds", default=None)
        ap.add_argument("--t-report", default=None)
        ap.add_argument("--model-dir", default=None)
        return cmd_run(ap.parse_args(a[1:]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
