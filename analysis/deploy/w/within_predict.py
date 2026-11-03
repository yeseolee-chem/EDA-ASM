#!/usr/bin/env python3
r"""within_predict.py — DEPLOY-MODEL-W Phase P: every unordered pair inside the test groups, predicted by the fold models
(docs/specs/DEPLOY_MODEL_W.md 9). env_d1fu2 on GPU (analysis/deploy/jobs/gpu_fu2.slurm, submitted from analysis/deploy/w).
Exit 0 done, 2 not ready / usage, 3 STOP.

  python within_predict.py run <i|@I> <n> [--source lf|g2]
        tasks (model x fold), model in (lf, lfV, lfW), fold 0..4 -> 15 tasks, this call runs [i::n]
        -> $DEPW/within_pred/D__<model>__S__TF__f<k>.parquet (gid, a_rxn, b_rxn, seed, target, g, fold, model) + .json
        (models + sha256, input checks, counts, fingerprint). Idempotent: an output whose json fingerprint equals the
        current one is skipped; a different one = STOP.
  python within_predict.py list

Models: lf = $DEP/units/D__lf__S__TF__f<k>__s0..4.pt (DEPLOY_MODEL K); lfV / lfW = the .pt files named in
$DEPW/units/D__<model>__S__TF__f<k>.json (pt_files). Groups = $DEPW/within/test_groups__f<k>.parquet (Phase B).
Pairs of a group: nodes = the group's a_rxn (ascending) then its reference b_rxn; every node pair (u, v) with u before v
(the reference is always second, so (a, ref) rows are g(a, ref) = the direct prediction). The graph readout is computed
in Phase E (eval_w.py).
Reused unchanged (import): predict_tf.load_models (input layout, Std with each .pt's statistics), check_reactions
(spec 14-1 input checks), InputSet (fu2 Data token arrays), predict_all (fu2 predict x s_y per model).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w_common as W  # noqa: E402

C = W.C
CODE_VERSION = "deploy-w-wpred-1"
MODELS = ("lf", "lfV", "lfW")
OUT_COLS = ["gid", "a_rxn", "b_rxn", "seed", "target", "g", "fold", "model"]


def tasks():
    return [(m, k) for m in MODELS for k in range(W.NF)]


def uname(m, k):
    return f"D__{m}__S__TF__f{k}"


def model_files(m, k):
    """[(tag, fold, seed, path)] of the 5 seed models of (model, fold) (predict_tf.model_files format)."""
    if m == "lf":
        return [(f"f{k}_s{j}", k, j, C.UNITS / f"D__lf__S__TF__f{k}__s{j}.pt") for j in range(5)]
    js = W.UNITS_W / f"{uname(m, k)}.json"
    if not js.is_file():
        W.not_ready(f"{js} missing (train_w.py run)")
    rec = W.read_json(js)
    if rec.get("smoke") or not rec.get("pt_files"):
        W.stop(f"{js}: smoke unit or no pt_files")
    return [(f"f{k}_s{j}", k, j, W.UNITS_W / p) for j, p in enumerate(rec["pt_files"])]


def pair_plan(T):
    """unordered node pairs of every group of a test-group table (gid, a_rxn, b_rxn): nodes = sorted a_rxn + the
    reference (last) -> DataFrame gid, a_rxn, b_rxn (first / second reaction of the pair), m."""
    out = []
    for g, q in T.groupby("gid", sort=True):
        refs = q.b_rxn.unique()
        if len(refs) != 1:
            W.stop(f"group {g}: {len(refs)} reference reactions")
        if q.a_rxn.duplicated().any():
            W.stop(f"group {g}: a reaction twice")
        ref = int(refs[0])
        nodes = sorted(int(x) for x in q.a_rxn) + [ref]
        if ref in nodes[:-1]:
            W.stop(f"group {g}: the reference is also a substituted reaction")
        m = len(nodes)
        iu, iv = np.triu_indices(m, 1)
        out.append(pd.DataFrame(dict(gid=int(g), a_rxn=np.asarray(nodes)[iu], b_rxn=np.asarray(nodes)[iv], m=m)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["gid", "a_rxn", "b_rxn", "m"])


def predict_task(m, k, source="lf", out_dir=None):
    import within as WI
    import data_lf as DL
    import predict_tf as PT
    out_dir = Path(out_dir) if out_dir else W.PRED_W
    cfg = DL.SOURCES[source]
    t_report = Path(cfg["res"]) / "T_report.json"
    T = WI.load_test_groups(k, source=source)
    files = model_files(m, k)
    miss = [str(p) for *_, p in files if not p.is_file()]
    if miss:
        W.not_ready(f"{uname(m, k)}: model files missing {miss[:2]}")
    tg = WI.default_root(source) / f"test_groups__f{k}.parquet"
    fp = dict(code=CODE_VERSION, code_sha256={f: W.sha256(HERE / f) for f in ("within_predict.py", "w_common.py")},
              predict_tf_sha256=W.sha256(W.DEPLOY_DIR / "predict_tf.py"),
              models=[W.sha256(p) for *_, p in files], test_groups_sha256=W.sha256(tg),
              tokens=dict(t_report=W.sha256(t_report), t_manifest=W.sha256(Path(cfg["res"]) / "T_manifest.csv"),
                          x182=W.sha256(cfg["x182"])), source=source)
    pq, js = out_dir / f"{uname(m, k)}.parquet", out_dir / f"{uname(m, k)}.json"
    if js.is_file() and pq.is_file():
        old = W.read_json(js)
        if old.get("fingerprint") != fp:
            W.stop(f"{js}: computed with other models / inputs / code (keys "
                   f"{sorted(x for x in fp if (old.get('fingerprint') or {}).get(x) != fp[x])})")
        if old.get("parquet_sha256") == W.sha256(pq):
            print(f"[P] {uname(m, k)}: cached", flush=True)
            return 0
    t0 = time.time()
    models, layout = PT.load_models(files)
    plan = pair_plan(T)
    rids = sorted(set(plan.a_rxn) | set(plan.b_rxn))
    ok, bad, info = PT.check_reactions(cfg["tokens"], cfg["x182"], rids, layout, t_report)
    oks = set(ok)
    keep = plan.a_rxn.isin(oks) & plan.b_rxn.isin(oks)
    P = plan[keep].reset_index(drop=True)
    X = PT.InputSet(cfg["tokens"], sorted(set(P.a_rxn) | set(P.b_rxn)), layout)
    ra = np.array([X.row[int(r)] for r in P.a_rxn])
    rb = np.array([X.row[int(r)] for r in P.b_rxn])
    Y = PT.predict_all(models, X, ra, rb)                                  # [5, n_pairs, 10] kcal/mol
    ns, npairs, nt = Y.shape
    seeds = np.array([mm["seed"] for mm in models])
    df = pd.DataFrame(dict(gid=np.tile(np.repeat(P.gid.to_numpy(), nt), ns), a_rxn=np.tile(np.repeat(P.a_rxn.to_numpy(), nt), ns),
                           b_rxn=np.tile(np.repeat(P.b_rxn.to_numpy(), nt), ns), seed=np.repeat(seeds, npairs * nt),
                           target=np.tile(np.array(W.TARGETS * 1), ns * npairs), g=Y.reshape(-1), fold=int(k), model=m))
    W.write_parquet(pq, df[OUT_COLS])
    sizes = plan.groupby("gid").m.first()
    rec = dict(unit=uname(m, k), model=m, fold=k, fingerprint=fp, parquet_sha256=W.sha256(pq),
               models=[dict(tag=mm["tag"], seed=mm["seed"], path=mm["path"], sha256=mm["sha256"], cell=mm["cell"])
                       for mm in models],
               n_groups=int(T.gid.nunique()), n_test_pairs=int(len(T)), n_pairs_planned=int(len(plan)),
               n_pairs_predicted=int(len(P)), n_excluded_reactions=len(bad),
               excluded_reactions={str(x): v for x, v in sorted(bad.items())}, group_size=dict(
                   min=int(sizes.min()), median=float(sizes.median()), max=int(sizes.max())) if len(sizes) else {},
               inputs=info, versions=__import__("data_lf").FT.versions(), seconds=time.time() - t0)
    W.write_json(js, rec)
    print(f"[P] {uname(m, k)}: {len(P)} pairs in {rec['n_groups']} groups, {len(bad)} excluded reactions "
          f"({rec['seconds']:.0f} s)", flush=True)
    return 0


def cmd_run(i, n, source="lf"):
    import data_lf as DL
    if source == "lf":
        DL.FT.require_gpu()
    i = C.array_task(i)
    n = int(n)
    if not (0 <= i < n):
        print(f"usage: run <i|@I> <n> with 0 <= i < n (got {i}, {n})", file=sys.stderr)
        return 2
    rc = 0
    for m, k in tasks()[i::n]:
        rc = max(rc, predict_task(m, k, source))
    return rc


def main(argv):
    a = argv[1:]
    src = "lf"
    if "--source" in a:
        j = a.index("--source")
        src = a[j + 1]
        a = a[:j] + a[j + 2:]
    if a == ["list"]:
        print("\n".join(f"{q}: {uname(m, k)}" for q, (m, k) in enumerate(tasks())))
        return 0
    if a[:1] == ["run"] and len(a) == 3:
        return cmd_run(a[1], a[2], src)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
