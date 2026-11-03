#!/usr/bin/env python3
r"""patch_test.py — test of analysis/deploy/w/predict_tf_ext.patch (DEPLOY_MODEL_W 13-1) before it is applied
(`python tests_w.py patch`; CPU, env_d1fu2, OMP_NUM_THREADS=1 inside). Exit 0 pass, 3 fail.

  1. `git apply --check` of the patch on the current analysis/deploy/predict_tf.py; the patch applied to a copy
     ($DEPW/dev/patch/t/analysis/deploy/predict_tf.py) is the tested file.
  2. default options = the original file, run in separate processes on the DEPLOY developer units (g2 data,
     $DEP/dev_tf/{kunits, kmodel, kres}; read only): I-1 (cmd_check_i1) and I-2 (cmd_check_i2) records and their
     prediction files are equal (every original column bitwise; the new column readout = 'direct'); run --model cv and
     --model final with default options on 300 pairs: equal outputs.
  3. new options: --model-dir <units> = the DEP_UNITS default (bitwise); --readout graph on the fold-0 test groups =
     w_common.graph_values on the direct predictions of every group pair (<= 1e-6); a group of 82 reactions is read
     directly with its reason, one of 80 by graph; two b_rxn in one group and a missing group_id STOP; a final folder
     with MODEL_CARD_W.json (rung lfW, readout graph) sets the defaults (= the lf files with --readout graph,
     bitwise); both cards in one folder STOP.
Results -> results/w/patch_test.json (+ $DEPW/dev/patch/*).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w_common as W  # noqa: E402

C = W.C
PATCH = HERE / "predict_tf_ext.patch"
ORIG = W.DEPLOY_DIR / "predict_tf.py"
ROOT = W.DEV / "patch"
DEVTF = C.DEP / "dev_tf"
GRAPH_TOL = 1e-6


def patched_copy():
    t = ROOT / "t"
    W.assert_write_ok(t / "probe")
    if t.exists():
        shutil.rmtree(t)
    (t / "analysis" / "deploy").mkdir(parents=True)
    shutil.copyfile(ORIG, t / "analysis" / "deploy" / "predict_tf.py")
    chk = subprocess.run(["git", "-C", str(C.REPO), "apply", "--check", str(PATCH)], capture_output=True, text=True)
    app = subprocess.run(["git", "apply", str(PATCH)], cwd=t, capture_output=True, text=True)
    return t / "analysis" / "deploy" / "predict_tf.py", dict(check_rc=chk.returncode, check_err=chk.stderr[-500:],
                                                            apply_rc=app.returncode, apply_err=app.stderr[-500:],
                                                            orig_sha256=W.sha256(ORIG), patch_sha256=W.sha256(PATCH))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- child process (one variant)
def child(variant, path, out):
    """runs inside a fresh process whose DEP_* variables point to the developer folders."""
    import numpy as np
    import pandas as pd
    import data_lf as DL
    PT = load_module(path, f"predict_tf_{variant}")
    out = Path(out)
    res = {}
    res["i1_rc"] = PT.cmd_check_i1("g2")
    res["i2_rc"] = PT.cmd_check_i2("g2")
    cfg = DL.SOURCES["g2"]
    K = pd.read_parquet(C.RES / "K_predictions.parquet")
    D = DL.get_data("g2")
    ids = sorted(K.pair_id.unique())[:300]
    pc = out / "pairs300.csv"
    W.write_csv(pc, D.P.loc[ids, ["a_rxn", "b_rxn"]].reset_index())
    tr = Path(cfg["res"]) / "T_report.json"
    for mode in ("cv", "final"):            # rung given (the original default rung is 'lf'; the developer units are g2)
        o, e, r = PT.run_predict(cfg["tokens"], cfg["x182"], pc, mode, "g2", t_report=tr)
        W.write_parquet(out / f"run_{mode}.parquet", o)
        res[f"run_{mode}_n"] = int(len(o))
    W.write_json(out / "child.json", res)
    return 0


def start_child(variant, path):
    """one child process (variants orig / patched / new run in parallel, one thread each)."""
    out = ROOT / f"out_{variant}"
    res = ROOT / f"res_{variant}"
    for d in (out, res):
        W.assert_write_ok(d / "probe")
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for f in ("K_predictions.parquet", "W_report.json"):
        shutil.copyfile(DEVTF / "kres" / f, res / f)
    env = dict(os.environ, DEP_RESULTS=str(res), DEP_UNITS=str(DEVTF / "kunits"), DEP_MODEL_DIR=str(DEVTF / "kmodel"),
               DEP_I_DIR=str(out / "I"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=str(W.DEPLOY_DIR))
    pr = subprocess.Popen([sys.executable, str(Path(__file__)), "_child", variant, str(path), str(out)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out, res, pr


def finish_child(pr):
    so, se = pr.communicate()
    return dict(rc=pr.returncode, stdout=so[-1500:], stderr=se[-3000:])


def frames_equal(a, b, cols):
    import numpy as np
    if len(a) != len(b):
        return False
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            if not np.array_equal(x.astype(float), y.astype(float), equal_nan=True):
                return False
        elif not a[c].reset_index(drop=True).astype(object).equals(b[c].reset_index(drop=True).astype(object)):
            return False
    return True


def new_options(path):
    """section 3 (child process 'new': DEP_UNITS / DEP_MODEL_DIR = the developer units, OMP_NUM_THREADS=1)."""
    import numpy as np
    import pandas as pd
    import data_lf as DL
    import within as WI
    PT = load_module(path, "predict_tf_new")
    cfg = DL.SOURCES["g2"]
    tr = Path(cfg["res"]) / "T_report.json"
    rec = {}
    T = WI.load_test_groups(0, source="g2")
    T = T[T.gid.isin(sorted(T.gid.unique())[:12])]
    pc = ROOT / "graph_pairs.csv"
    W.write_csv(pc, T.assign(group_id=T.gid)[["pair_id", "a_rxn", "b_rxn", "group_id"]])
    # --model-dir = the env default (cv)
    o1, _, _ = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "cv", "g2", folds=[0], t_report=tr)
    o2, _, _ = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "cv", "g2", folds=[0], t_report=tr,
                              model_dir=DEVTF / "kunits")
    rec["model_dir_equals_default"] = frames_equal(o1, o2, list(o1.columns))
    # --readout graph vs w_common.graph_values on the direct pair predictions
    og, eg, rg = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "cv", "g2", folds=[0], t_report=tr, readout="graph")
    import within_predict as WP
    plan = WP.pair_plan(T)
    pp = ROOT / "graph_allpairs.csv"
    W.write_csv(pp, plan.assign(pair_id=[f"Q{i:06d}" for i in range(len(plan))])[["pair_id", "a_rxn", "b_rxn"]])
    files = PT.model_files("cv", "g2", [0], None, None)
    models, layout = PT.load_models(files)
    X = PT.InputSet(cfg["tokens"], sorted(set(plan.a_rxn) | set(plan.b_rxn)), layout)
    Y = PT.predict_all(models, X, np.array([X.row[int(r)] for r in plan.a_rxn]), np.array([X.row[int(r)] for r in plan.b_rxn]))
    import eval_w as EW
    rows = []
    for j in range(len(Y)):
        for ti, t in enumerate(W.TARGETS):
            rows.append(pd.DataFrame(dict(gid=plan.gid, a_rxn=plan.a_rxn, b_rxn=plan.b_rxn, seed=j, target=t, g=Y[j, :, ti])))
    G = EW.graph_predictions(pd.concat(rows, ignore_index=True), T, 0)
    m = og.merge(G[["pair_id", "target", "yhat", "yhat_seed_sd"]], on=["pair_id", "target"], suffixes=("", "_w"))
    rec["graph_vs_w_common"] = dict(n=int(len(m)), max_abs=float(np.abs(m.yhat - m.yhat_w).max()),
                                    max_abs_sd=float(np.abs(m.yhat_sd - m.yhat_seed_sd).max()),
                                    readout=sorted(set(og.readout)), ok=bool(len(m) == len(og) and
                                                                             np.abs(m.yhat - m.yhat_w).max() <= GRAPH_TOL))
    # size fallback: 81 + 1 (direct) and 79 + 1 (graph) reactions; synthetic pairs over g2 reactions
    D = DL.get_data("g2")
    rx = [int(r) for r in D.rids[:200]]
    ref = rx[0]
    S = pd.DataFrame(dict(pair_id=[f"Z{i:04d}" for i in range(81 + 79)], a_rxn=rx[1:82] + rx[82:161],
                          b_rxn=[ref] * 160, group_id=["big"] * 81 + ["edge"] * 79))
    ps = ROOT / "size_pairs.csv"
    W.write_csv(ps, S)
    os_, _, rs = PT.run_predict(cfg["tokens"], cfg["x182"], ps, "cv", "g2", folds=[0], seeds=[0], t_report=tr,
                                readout="graph")
    rd = os_.drop_duplicates("pair_id").set_index("pair_id").readout
    rec["size_rule"] = dict(fallback=rs["graph"]["fallback"], big=sorted(set(rd.loc[S.pair_id[:81]])),
                            edge=sorted(set(rd.loc[S.pair_id[81:]])), max_m=rs["graph"].get("max_m"),
                            ok=bool(sorted(set(rd.loc[S.pair_id[:81]])) == ["direct"] and
                                    sorted(set(rd.loc[S.pair_id[81:]])) == ["graph"] and "big" in rs["graph"]["fallback"]
                                    and rs["graph"].get("max_m") == 80))
    # STOP cases
    def stops(fn):
        try:
            fn()
            return False
        except SystemExit as e:
            return e.code == 3
    S2 = S.iloc[:5].assign(b_rxn=[ref, ref, rx[199], ref, ref], group_id="x")
    p2 = ROOT / "two_refs.csv"
    W.write_csv(p2, S2)
    rec["two_refs_stop"] = stops(lambda: PT.run_predict(cfg["tokens"], cfg["x182"], p2, "cv", "g2", folds=[0], seeds=[0],
                                                        t_report=tr, readout="graph"))
    p3 = ROOT / "no_group.csv"
    W.write_csv(p3, S.iloc[:5][["pair_id", "a_rxn", "b_rxn"]])
    rec["no_group_id_stop"] = stops(lambda: PT.run_predict(cfg["tokens"], cfg["x182"], p3, "cv", "g2", folds=[0],
                                                           seeds=[0], t_report=tr, readout="graph"))
    # MODEL_CARD_W defaults (rung lfW, readout graph) = the same weights with --readout graph
    fd = ROOT / "model_w"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir(parents=True)
    card = W.read_json(DEVTF / "kmodel" / "MODEL_CARD.json")
    for j in range(5):
        shutil.copyfile(DEVTF / "kmodel" / f"D__g2__S__TF__all__s{j}.pt", fd / f"D__lfW__S__TF__all__s{j}.pt")
    W.write_json(fd / "MODEL_CARD_W.json", dict(card, rung="lfW", r=4, readout="graph"))
    ow, _, rw = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "final", t_report=tr, model_dir=fd)
    ol, _, _ = PT.run_predict(cfg["tokens"], cfg["x182"], pc, "final", t_report=tr, readout="graph")
    rec["card_w_defaults"] = dict(rung=rw["rung"], readout=rw["readout"], equal_to_lf_graph=frames_equal(
        ow, ol, ["pair_id", "target", "yhat", "yhat_sd", "readout"]),
        ok=bool(rw["rung"] == "lfW" and rw["readout"] == "graph" and frames_equal(ow, ol, ["pair_id", "target", "yhat",
                                                                                         "yhat_sd", "readout"])))
    shutil.copyfile(DEVTF / "kmodel" / "MODEL_CARD.json", fd / "MODEL_CARD.json")
    rec["two_cards_stop"] = stops(lambda: PT.run_predict(cfg["tokens"], cfg["x182"], pc, "final", t_report=tr,
                                                         model_dir=fd))
    ok = (rec["model_dir_equals_default"] and rec["graph_vs_w_common"]["ok"] and rec["size_rule"]["ok"]
          and rec["two_refs_stop"] and rec["no_group_id_stop"] and rec["card_w_defaults"]["ok"] and rec["two_cards_stop"])
    return dict(rec, ok=bool(ok))


def main():
    import numpy as np
    import pandas as pd
    t0 = time.time()
    W.assert_write_ok(ROOT / "probe")
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    torch.set_num_threads(1)
    path, ap = patched_copy()
    rec = dict(apply=ap, apply_ok=ap["check_rc"] == 0 and ap["apply_rc"] == 0)
    if not rec["apply_ok"]:
        rec["ok"] = False
        W.write_json(W.RESW / "patch_test.json", rec)
        print(json.dumps(rec, indent=1))
        return 3
    o_out, o_res, o_pr = start_child("orig", ORIG)
    p_out, p_res, p_pr = start_child("patched", path)
    n_out, _, n_pr = start_child("new", path)
    o_run, p_run, n_run = finish_child(o_pr), finish_child(p_pr), finish_child(n_pr)
    rec["children"] = dict(orig=o_run, patched=p_run, new=n_run)
    dflt = {}
    if o_run["rc"] == 0 and p_run["rc"] == 0:
        so, sp = W.read_json(o_res / "I_selftest.json"), W.read_json(p_res / "I_selftest.json")
        for k in ("I-1", "I-2"):
            a, b = dict(so[k]), dict(sp[k])
            dflt[f"{k}_record_equal"] = {x: a[x] == b[x] for x in a if x not in ("gpu",)}
            dflt[f"{k}_ok"] = [a["ok"], b["ok"]]
            dflt[f"{k}_max_abs_diff"] = [a["max_abs_diff"], b["max_abs_diff"]]
        for f in ("I/i1_pred.parquet", "I/i2_pred.parquet", "run_cv.parquet", "run_final.parquet"):
            A, B = pd.read_parquet(o_out / f), pd.read_parquet(p_out / f)
            cols = list(A.columns)
            dflt[f] = dict(orig_columns=cols, patched_columns=list(B.columns), equal=frames_equal(A, B, cols),
                           extra=sorted(set(B.columns) - set(A.columns)),
                           readout_values=sorted(set(B.readout)) if "readout" in B.columns else None)
        dflt["ok"] = bool(all(all(v.values()) for k, v in dflt.items() if k.endswith("_record_equal"))
                          and all(v["equal"] and v["extra"] == ["readout"] and v["readout_values"] == ["direct"]
                                  for k, v in dflt.items() if k.endswith(".parquet"))
                          and dflt["I-1_ok"] == [True, True] and dflt["I-2_ok"] == [True, True])
    else:
        dflt["ok"] = False
    rec["default_options"] = dflt
    rec["new_options"] = W.read_json(n_out / "new.json") if (n_out / "new.json").is_file() else \
        dict(ok=False, error=f"child rc {n_run['rc']}")
    rec["ok"] =bool(rec["apply_ok"] and dflt.get("ok") and rec["new_options"].get("ok"))
    rec.update(patched_sha256=W.sha256(path), seconds=time.time() - t0, threads=1,
               units=dict(cv=str(DEVTF / "kunits"), final=str(DEVTF / "kmodel"), K=str(DEVTF / "kres")))
    W.write_json(W.RESW / "patch_test.json", rec)
    print(json.dumps(W._jsonable(dict(ok=rec["ok"], apply_ok=rec["apply_ok"], default_ok=dflt.get("ok"),
                                      new=({k: (v.get("ok") if isinstance(v, dict) else v)
                                            for k, v in rec["new_options"].items()}))), indent=1))
    return 0 if rec["ok"] else 3


def child_new(path, out):
    import torch
    torch.set_num_threads(1)
    try:
        rec = new_options(path)
    except SystemExit as e:
        import traceback
        rec = dict(ok=False, error=f"SystemExit {e.code}", traceback=traceback.format_exc()[-2000:])
    W.write_json(Path(out) / "new.json", rec)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["_child"]:
        if sys.argv[2] == "new":
            sys.exit(child_new(sys.argv[3], sys.argv[4]))
        sys.exit(child(sys.argv[2], sys.argv[3], sys.argv[4]))
    sys.exit(main())
