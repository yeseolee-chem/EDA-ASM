#!/usr/bin/env python3
r"""predict_tf.py — DEPLOY-MODEL 14-1: apply the lf . S . TF models to token / 182-column inputs (any reactions).
env_d1fu2 (GPU for I-1 / I-2; CPU works too); jobs/gpu_fu2.slurm. Exit 0 done, 2 not ready / usage, 3 STOP.

  python predict_tf.py run <tokens_root> <x182.parquet> <pairs.csv> <out.parquet> [--model final|cv] [--rung lf]
                           [--folds 0,1,...] [--seeds 0,1,...] [--t-report T_report.json]
        pairs.csv: pair_id, a_rxn, b_rxn (other columns ignored). tokens_root/<rxn_id>.npz (fu2/tokens.py format),
        x182.parquet: rxn_id + the 182 columns.
        --model final (default): $DEP/model/D__<rung>__S__TF__all__s{0..4}.pt, yhat = mean of the 5, yhat_sd = their SD
        --model cv: the fold units $DEP/units/D__<rung>__S__TF__f<k>__s<j>.pt (25; validation): one row per pair x
                    target x fold, yhat / yhat_sd = mean / SD over that fold's seeds (= K_predictions yhat /
                    yhat_seed_sd). --folds / --seeds restrict the files (tests).
        out.parquet columns: pair_id, a_rxn, b_rxn, target, yhat, yhat_sd, out_of_range [, fold (cv)]; out_of_range =
        yhat_sd > the target's applicability threshold (results/E_ad.json; final mode: MODEL_CARD applicability), None
        while no threshold exists. Next to it: <out stem>.excluded.csv (pair_id, a_rxn, b_rxn, reason) and
        <out stem>.report.json (models, sha256, input checks, counts).
  python predict_tf.py check-i1          I-1: --model cv on every K test pair vs results/K_predictions.parquet (per
                                         fold, mean of 5 seeds), <= 1e-4 kcal/mol -> results/I_selftest.json 'I-1'
  python predict_tf.py check-i2          I-2: --model final on $DEP/model/W_check_pairs.csv (1,000 training pairs) vs
                                         W_check_pred.parquet (recorded right after W), <= 1e-6 -> 'I-2'

Input checks per reaction (spec 14-1; a failing reaction removes its pairs, reason recorded): token file present;
token_version = the model layout (MODEL_CARD / .pt layout); elements subset of the 9 model elements; bin / num widths =
the layout (names too with --t-report); 182 names and order of x182.parquet = the layout glob_cols; token num / pos /
glob and the x182 row finite; token glob = x182 row (1e-12 relative); 2 formed edges (T-a).
Standardisation: each .pt's own statistics (mu, sd, gmu, gsd, sy) through fu2 train.Std.tensors (fu2/train.py 155-160)
on a Std object carrying them (never recomputed on new inputs); output = fu2 train.predict (186-193, bs 256) x s_y.
Model = fu2 models.build('TF', cell, d_in, 10, d_glob, 'A+G') (models.py 283) + load_state_dict.
Extension point for DEPLOY_MODEL_W 13-1 (--model-dir, --readout graph): model_files() / load_models() / predict_all().
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402
import data_lf as DL  # noqa: E402

FT, Mo, F = DL.FT, DL.Mo, DL.F
import torch  # noqa: E402

TARGETS = list(F.TARGETS)
MODEL, VARIANT, ARM = "TF", "A+G", "S"
STD_KEYS = ("mu", "sd", "gmu", "gsd", "sy")
I1_TOL, I2_TOL = 1e-4, 1e-6
OUT_COLS = ["pair_id", "a_rxn", "b_rxn", "target", "yhat", "yhat_sd", "out_of_range"]


# ================================================================ models
def model_files(mode, rung=C.RUNG, folds=None, seeds=None, model_dir=None):
    """[(tag, fold, seed index, path)] of the models of a mode (final: $DEP/model, cv: $DEP/units)."""
    seeds = list(range(len(F.FINAL_SEEDS))) if seeds is None else [int(s) for s in seeds]
    if mode == "final":
        d = Path(model_dir) if model_dir else C.MODEL_DIR
        return [(f"all_s{j}", None, j, d / f"D__{rung}__{ARM}__{MODEL}__all__s{j}.pt") for j in seeds]
    if mode == "cv":
        d = Path(model_dir) if model_dir else C.UNITS
        folds = list(range(DL.NF)) if folds is None else [int(k) for k in folds]
        return [(f"f{k}_s{j}", k, j, d / f"D__{rung}__{ARM}__{MODEL}__f{k}__s{j}.pt") for k in folds for j in seeds]
    C.stop(f"unknown --model {mode}")


def std_object(std):
    """a fu2 train.Std carrying saved statistics (its tensors() method standardises with them, unchanged code)."""
    s = FT.Std.__new__(FT.Std)
    for k in STD_KEYS:
        setattr(s, k, np.asarray(std[k], dtype=float))
    return s


def load_models(files):
    """load every .pt: model (eval), Std object, layout; all models must share one layout (else STOP)."""
    out, layout = [], None
    for tag, fold, j, p in files:
        if not p.is_file():
            C.not_ready(f"{p} missing")
        obj = torch.load(p, map_location=FT.DEV)
        need = ("state", "cell", "std", "d_in", "d_glob", "TARGETS", "layout")
        miss = [k for k in need if k not in obj]
        if miss or obj.get("model", MODEL) != MODEL or obj.get("variant", VARIANT) != VARIANT:
            C.stop(f"{p}: not a train_tf model file (missing {miss})")
        if list(obj["TARGETS"]) != TARGETS:
            C.stop(f"{p}: TARGETS {obj['TARGETS']} != {TARGETS}")
        lay = obj["layout"]
        if layout is None:
            layout = lay
        elif json.dumps(lay, sort_keys=True) != json.dumps(layout, sort_keys=True):
            C.stop(f"{p}: input layout differs from the other models")
        model = Mo.build(MODEL, obj["cell"], int(obj["d_in"]), len(TARGETS), int(obj["d_glob"]), VARIANT,
                         F.DROPOUT).to(FT.DEV)
        model.load_state_dict(obj["state"])
        model.eval()
        out.append(dict(tag=tag, fold=fold, seed=j, path=str(p), sha256=C.sha256(p), model=model,
                        std=std_object(obj["std"]), sy=np.asarray(obj["std"]["sy"], dtype=float), cell=obj["cell"]))
    return out, layout


# ================================================================ inputs
class InputSet(FT.Data):
    """token reactions for prediction as fu2 train.Data arrays (data_lf.fill_token_arrays = the Data.__init__ loop),
    rows in ascending rxn_id order; static() inherited."""

    def __init__(self, tokens_root, rids, layout):
        self.rids = np.array(sorted(int(r) for r in rids), dtype=int)
        self.row = {int(r): k for k, r in enumerate(self.rids)}
        self.elements = list(layout["elements"])
        self.bin_cols, self.num_cols = list(layout["bin_cols"]), list(layout["num_cols"])
        paths = [Path(tokens_root) / f"{r}.npz" for r in self.rids]
        arr, self.file_sha = DL.fill_token_arrays(paths, self.elements, len(self.bin_cols), len(self.num_cols))
        for k in ("onehot", "bin", "num", "pos", "mask", "etype", "frag", "glob", "nmax"):
            setattr(self, k, arr[k])
        self.n = self.mask.sum(1).astype(int)
        self.d_in = len(self.elements) + len(self.bin_cols) + len(self.num_cols)
        self.d_glob = self.glob.shape[1]


def check_reactions(tokens_root, x182_path, rids, layout, t_report=None):
    """spec 14-1 input checks -> (ok rxn ids, {rxn_id: reason}, info)."""
    bad, info = {}, {}
    if t_report is not None:
        tr = C.read_json(t_report)
        diff = [k for k in ("bin_cols", "num_cols", "glob_cols") if list(tr.get(k) or []) != list(layout[k])]
        if not set(int(z) for z in tr.get("elements") or []) <= set(int(z) for z in layout["elements"]):
            diff.append("elements")
        if tr.get("token_version") != layout["token_version"]:
            diff.append("token_version")
        info["t_report"] = dict(path=str(t_report), sha256=C.sha256(t_report), diff=diff)
        if diff:
            C.stop(f"{t_report}: token layout differs from the model layout: {diff}")
    X = pd.read_parquet(x182_path)
    cols = [c for c in X.columns if c != "rxn_id"]
    info["x182"] = dict(path=str(x182_path), sha256=C.sha256(x182_path), n_rows=int(len(X)))
    if "rxn_id" not in X.columns or cols != list(layout["glob_cols"]):
        C.stop(f"{x182_path}: rxn_id + 182 column names / order != the model layout glob_cols")
    X = X.assign(rxn_id=X.rxn_id.astype(int))
    if X.rxn_id.duplicated().any():
        C.stop(f"{x182_path}: duplicated rxn_id")
    X = X.set_index("rxn_id")
    els = set(int(z) for z in layout["elements"])
    nb, nm = len(layout["bin_cols"]), len(layout["num_cols"])
    for r in sorted(set(int(x) for x in rids)):
        p = Path(tokens_root) / f"{r}.npz"
        if not p.is_file():
            bad[r] = "token file missing"
            continue
        try:
            t = np.load(p)
            tv = str(t["token_version"][0]) if "token_version" in t.files else None
            Z = t["Z"]
            n = len(Z)
            why = None
            if tv != layout["token_version"]:
                why = f"token_version {tv} != {layout['token_version']}"
            elif not set(int(z) for z in Z) <= els:
                why = f"elements {sorted(set(int(z) for z in Z) - els)} outside the model elements"
            elif t["bin"].shape != (n, nb) or t["num"].shape != (n, nm) or t["pos"].shape != (n, 3):
                why = f"token widths bin {t['bin'].shape} num {t['num'].shape} pos {t['pos'].shape} != ({n}, {nb} / {nm} / 3)"
            elif len(t["glob"]) != len(layout["glob_cols"]):
                why = f"token glob width {len(t['glob'])} != {len(layout['glob_cols'])}"
            elif not (np.isfinite(t["num"]).all() and np.isfinite(t["pos"]).all() and np.isfinite(t["glob"]).all()):
                why = "non-finite token values"
            elif int((t["etype"] == 2).sum()) != 2:
                why = f"{int((t['etype'] == 2).sum())} formed edges (T-a needs 2)"
            elif r not in X.index:
                why = "no x182 row"
            else:
                g = X.loc[r, cols].to_numpy(dtype=float)
                if not np.isfinite(g).all():
                    why = "non-finite x182 values"
                elif (np.abs(g - t["glob"]) > DL.GLOB_REL_TOL * np.maximum(1.0, np.abs(t["glob"]))).any():
                    why = "token glob differs from the x182 row"
            if why:
                bad[r] = why
        except Exception as e:                                     # noqa: BLE001  (an unreadable file = excluded)
            bad[r] = f"unreadable token file: {type(e).__name__}: {e}"
    ok = sorted(set(int(x) for x in rids) - set(bad))
    return ok, bad, info


# ================================================================ prediction
def predict_all(models, X, ra, rb):
    """[n_models, n_pairs, 10] kcal/mol: per model its own standardisation (Std.tensors) + fu2 predict x s_y."""
    out = []
    for m in models:
        T = m["std"].tensors(X)
        out.append(FT.predict(m["model"], T, ra, rb) * m["sy"])
        del T
    return np.stack(out) if out else np.zeros((0, len(ra), len(TARGETS)))


def thresholds(mode, model_dir=None):
    """applicability thresholds per target: final -> MODEL_CARD applicability (fallback results/E_ad.json); cv ->
    results/E_ad.json; None when absent."""
    card = (Path(model_dir) if model_dir else C.MODEL_DIR) / "MODEL_CARD.json"
    if mode == "final" and card.is_file():
        th = ((C.read_json(card).get("applicability") or {}).get("thresholds_yhat_sd"))
        if th:
            return th, str(card)
    adp = C.RES / "E_ad.json"
    if adp.is_file():
        return C.read_json(adp).get("thresholds"), str(adp)
    return None, None


def run_predict(tokens_root, x182, pairs_csv, mode="final", rung=C.RUNG, folds=None, seeds=None, t_report=None,
                model_dir=None):
    """the run command without writing: (out frame, excluded frame, report dict)."""
    t0 = time.time()
    files = model_files(mode, rung, folds, seeds, model_dir)
    models, layout = load_models(files)
    if mode == "final":
        card = (Path(model_dir) if model_dir else C.MODEL_DIR) / "MODEL_CARD.json"
        if card.is_file():
            cd = C.read_json(card)
            want = dict(elements=cd["token"]["elements"], bin_cols=cd["token"]["bin_cols"],
                        num_cols=cd["token"]["num_cols"], glob_cols=cd["glob_cols"], token_version=cd["token"]["version"])
            diff = [k for k, v in want.items() if json.dumps(v) != json.dumps(layout[k])]
            if diff:
                C.stop(f"MODEL_CARD.json layout differs from the model files: {diff}")
        else:
            C.not_ready(f"{card} missing (train_tf.py final)")
    P = pd.read_csv(pairs_csv, dtype={"pair_id": str})
    for c in ("pair_id", "a_rxn", "b_rxn"):
        if c not in P.columns:
            C.stop(f"{pairs_csv}: column {c} missing")
    if P.pair_id.duplicated().any():
        C.stop(f"{pairs_csv}: duplicated pair_id")
    P = P.assign(a_rxn=P.a_rxn.astype(int), b_rxn=P.b_rxn.astype(int))
    rids = set(P.a_rxn) | set(P.b_rxn)
    ok, bad, info = check_reactions(tokens_root, x182, rids, layout, t_report)
    oks = set(ok)
    keep = P.a_rxn.isin(oks) & P.b_rxn.isin(oks)
    reason = [f"a {a}: {bad[a]}" if a in bad else (f"b {b}: {bad[b]}" if b in bad else "")
              for a, b in zip(P.a_rxn.tolist(), P.b_rxn.tolist())]
    excl = P.loc[~keep, ["pair_id", "a_rxn", "b_rxn"]].assign(reason=np.array(reason, dtype=object)[~keep.to_numpy()])
    Pk = P[keep].reset_index(drop=True)
    th, th_src = thresholds(mode, model_dir)
    parts = []
    if len(Pk):
        X = InputSet(tokens_root, sorted(set(Pk.a_rxn) | set(Pk.b_rxn)), layout)
        ra = np.array([X.row[int(r)] for r in Pk.a_rxn])
        rb = np.array([X.row[int(r)] for r in Pk.b_rxn])
        Y = predict_all(models, X, ra, rb)                                 # [models, n, 10]
        groups = [(None, list(range(len(models))))] if mode == "final" else \
            [(k, [i for i, m in enumerate(models) if m["fold"] == k]) for k in sorted({m["fold"] for m in models})]
        for k, idx in groups:
            Yk = Y[idx]
            for j, t in enumerate(TARGETS):
                sd = Yk[:, :, j].std(0)
                oor = [None] * len(sd) if not th or th.get(t) is None else (sd > float(th[t])).tolist()
                d = dict(pair_id=Pk.pair_id.to_numpy(), a_rxn=Pk.a_rxn.to_numpy(), b_rxn=Pk.b_rxn.to_numpy(), target=t,
                         yhat=Yk[:, :, j].mean(0), yhat_sd=sd, out_of_range=pd.array(oor, dtype="boolean"))
                if k is not None:
                    d["fold"] = int(k)
                parts.append(pd.DataFrame(d))
    cols = OUT_COLS + (["fold"] if mode == "cv" else [])
    out = pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
    rep = dict(mode=mode, rung=rung, models=[dict(tag=m["tag"], fold=m["fold"], seed=m["seed"], path=m["path"],
                                                   sha256=m["sha256"], cell=m["cell"]) for m in models],
               layout=dict(layout, glob_cols_n=len(layout["glob_cols"])), tokens_root=str(tokens_root),
               pairs=dict(path=str(pairs_csv), sha256=C.sha256(pairs_csv), n=int(len(P)), n_predicted=int(len(Pk)),
                          n_excluded=int(len(excl))),
               n_reactions=dict(requested=len(rids), ok=len(ok), excluded=len(bad)),
               excluded_reactions={str(k): v for k, v in sorted(bad.items())}, inputs=info,
               thresholds=th, thresholds_source=th_src, versions=FT.versions(), seconds=time.time() - t0)
    rep["layout"].pop("glob_cols", None)
    return out, excl, rep


def _stem(out):
    out = Path(out)
    return out.with_name(out.name[:-len(".parquet")] if out.name.endswith(".parquet") else out.stem)


def write_outputs(out_path, out, excl, rep):
    C.write_parquet(out_path, out)
    s = _stem(out_path)
    C.write_csv(s.with_name(s.name + ".excluded.csv"), excl)
    C.write_json(s.with_name(s.name + ".report.json"), dict(rep, output=str(out_path), output_sha256=C.sha256(out_path)))


def cmd_run(a):
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
    out, excl, rep = run_predict(a.tokens_root, a.x182, a.pairs, a.model, a.rung, folds, seeds, a.t_report)
    write_outputs(a.out, out, excl, rep)
    print(f"[predict] {rep['pairs']['n_predicted']} pairs predicted, {rep['pairs']['n_excluded']} excluded, "
          f"{len(rep['models'])} models -> {a.out}", flush=True)
    return 0


# ================================================================ I-1 / I-2 (spec 14-3)
I_DIR = Path(os.environ.get("DEP_I_DIR", C.DEP / "I"))           # I-1 / I-2 scratch (override: tests only)


def _merge_selftest(key, rec):
    p = C.RES / "I_selftest.json"
    cur = C.read_json(p) if p.is_file() else {}
    cur[key] = rec
    cur["all_pass_I1_I2"] = bool(all((cur.get(k) or {}).get("ok") for k in ("I-1", "I-2")))
    C.write_json(p, cur)


def cmd_check_i1(source="lf"):
    kp = C.RES / "K_predictions.parquet"
    if not kp.is_file():
        C.not_ready(f"{kp} missing (train_tf.py collect)")
    K = pd.read_parquet(kp)
    D = DL.get_data(source)
    cfg = DL.SOURCES[source]
    ids = sorted(K.pair_id.unique())
    P = D.P.loc[ids, ["a_rxn", "b_rxn"]].reset_index()
    pc = I_DIR / "i1_pairs.csv"
    C.write_csv(pc, P)
    out, excl, rep = run_predict(cfg["tokens"], cfg["x182"], pc, "cv", D.rung, t_report=Path(cfg["res"]) / "T_report.json")
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
               n_excluded_pairs=int(len(excl)), per_fold_max={int(k): float(np.nanmax(g)) for k, g in
                                                             pd.Series(d).groupby(m.fold.to_numpy())},
               k_predictions_sha256=C.sha256(kp), n_models=len(rep["models"]),
               gpu=FT.versions()["gpu"])
    _merge_selftest("I-1", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 3


def cmd_check_i2(source="lf"):
    cfg = DL.SOURCES[source]
    rung = DL.get_data(source).rung
    pc, pp = C.MODEL_DIR / "W_check_pairs.csv", C.MODEL_DIR / "W_check_pred.parquet"
    if not (pc.is_file() and pp.is_file()):
        C.not_ready(f"{pc} / {pp} missing (train_tf.py final)")
    W = pd.read_parquet(pp)
    out, excl, rep = run_predict(cfg["tokens"], cfg["x182"], pc, "final", rung, t_report=Path(cfg["res"]) / "T_report.json")
    write_outputs(I_DIR / "i2_pred.parquet", out, excl, rep)
    m = W[["pair_id", "target", "yhat", "yhat_sd"]].merge(out[["pair_id", "target", "yhat", "yhat_sd"]],
                                                          on=["pair_id", "target"], how="left", suffixes=("_w", "_p"))
    miss = int(m.yhat_p.isna().sum())
    d = np.abs(m.yhat_w - m.yhat_p).to_numpy(float)
    mx = float(np.nanmax(d)) if len(d) else float("nan")
    wr = C.read_json(C.RES / "W_report.json") if (C.RES / "W_report.json").is_file() else {}
    gpu_now, gpu_w = FT.versions()["gpu"], (wr.get("versions") or {}).get("gpu")
    rec = dict(check="I-2", ok=bool(miss == 0 and len(excl) == 0 and mx <= I2_TOL), tol=I2_TOL, max_abs_diff=mx,
               max_abs_diff_sd=float(np.nanmax(np.abs(m.yhat_sd_w - m.yhat_sd_p))), n_rows=int(len(m)),
               n_missing=miss, n_excluded_pairs=int(len(excl)), gpu=gpu_now, gpu_W=gpu_w, same_gpu=gpu_now == gpu_w,
               w_check_pred_sha256=C.sha256(pp))
    _merge_selftest("I-2", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 3


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if a[:1] == ["check-i1"] and len(a) == 1:
        return cmd_check_i1()
    if a[:1] == ["check-i2"] and len(a) == 1:
        return cmd_check_i2()
    if a[:1] == ["run"]:
        ap = argparse.ArgumentParser(prog="predict_tf.py run")
        ap.add_argument("tokens_root")
        ap.add_argument("x182")
        ap.add_argument("pairs")
        ap.add_argument("out")
        ap.add_argument("--model", choices=("final", "cv"), default="final")
        ap.add_argument("--rung", default=C.RUNG)
        ap.add_argument("--folds", default=None)
        ap.add_argument("--seeds", default=None)
        ap.add_argument("--t-report", default=None)
        return cmd_run(ap.parse_args(a[1:]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
