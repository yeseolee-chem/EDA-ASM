#!/usr/bin/env python3
r"""within.py — DEPLOY-MODEL-W Phase B: within-group pairs (docs/specs/DEPLOY_MODEL_W.md 3). CPU, env_d1fu2
(jobs/cpu_py.slurm DEPENV=d1fu2) or espley. Exit 0 done, 2 not ready / usage, 3 STOP.

  python within.py build [--source lf|g2] [--root DIR]
        lf (production): DeployData('lf') -> $DEPW/within/T1__f<k>__{train,inner_train}.npz (k = 0..4), all__train.npz,
        test_groups__f<k>.parquet, meta.json; results/w/B_report.json. Idempotent: an existing file whose meta record
        matches the current inputs is kept; a stale one = STOP.
        g2 (developer dry run; LADDER g2 inputs + LADDER usable rule, as tests.py C-2): default root $DEPW/dev/within_g2,
        report in the root (never the production folder).
  python within.py summary [--source lf|g2]    the group statistics of S_folds.csv and of the source (no writes)

Definition (spec 3-1): for the usable training pairs `tr` of one fit (DeployData split rows), every pair i < j with the same
group key (b_rxn, fam, pos) of D1PAIR S_folds.csv gives one within-group pair ra = D.ra[i], rb = D.ra[j],
y = D.y[i] - D.y[j] (10 targets, kcal/mol; = y_a1 - y_a2 because dd = y_a - y_b, d1_pairs.py 9), sampling weight
w = 2 / (n_g - 1) (n_g = training pairs of the group). Groups: global ids gid = rank of (b_rxn, fam, pos) over every
S_folds pair (w_common.group_table); ref_kind reference and anchor both allowed. Pair order: gid, then (i, j) in pair-row
order (itertools.combinations). npz arrays: ra, rb, y, w, gid, pi, pj (pair-table rows), pid_i, pid_j (pair ids),
a_rxn_i, a_rxn_j (reaction ids), n_g (per within pair).
test_groups__f<k>.parquet: the fold's usable test pairs with pair_id, gid, a_rxn, b_rxn, fam, pos, Xb, sigma, ref_kind,
core, in_P_main (evaluation and graph readout).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import itertools  # noqa: E402
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
CODE_VERSION = "deploy-w-within-1"
FIT_KEYS = ("train", "inner_train")
SPEC_EXPECT = dict(groups=353, median_size=19, pairs_before_split=79600)


def fname(k, key):
    return f"all__{key}.npz" if str(k) == "all" else f"{W.SPLIT}__f{k}__{key}.npz"


def default_root(source):
    """lf: $DEPW/within; other sources (developer dry runs): $DEPW_WITHIN_<SOURCE> or $DEPW/dev/within_<source>."""
    if source == "lf":
        return W.WITHIN
    return Path(os.environ.get(f"DEPW_WITHIN_{source.upper()}", W.DEV / f"within_{source}"))


# ================================================================ construction
def pool(D, rows, gid_of_pair):
    """within-group pairs of the pair-table rows `rows` (spec 3-1) -> dict of arrays (see module doc)."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    pid = D.P.index.to_numpy()[rows].astype(str)
    gid = gid_of_pair.reindex(pid).to_numpy()
    if pd.isna(gid).any():
        W.stop(f"{int(pd.isna(gid).sum())} pairs without a group (S_folds)")
    gid = gid.astype(np.int64)
    by = {}
    for r, g in zip(rows.tolist(), gid.tolist()):
        by.setdefault(g, []).append(r)
    pi, pj, gg, ng = [], [], [], []
    for g in sorted(by):
        rr = by[g]
        for i, j in itertools.combinations(rr, 2):
            pi.append(i)
            pj.append(j)
            gg.append(g)
            ng.append(len(rr))
    pi = np.asarray(pi, dtype=np.int64)
    pj = np.asarray(pj, dtype=np.int64)
    ng = np.asarray(ng, dtype=np.int64)
    P = D.P
    out = dict(ra=D.ra[pi].astype(np.int64), rb=D.ra[pj].astype(np.int64), y=(D.y[pi] - D.y[pj]).astype(np.float64),
               w=(2.0 / (ng - 1)).astype(np.float64) if len(ng) else np.zeros(0), gid=np.asarray(gg, dtype=np.int64),
               pi=pi, pj=pj, pid_i=P.index.to_numpy()[pi].astype(str), pid_j=P.index.to_numpy()[pj].astype(str),
               a_rxn_i=P.a_rxn.to_numpy()[pi].astype(np.int64), a_rxn_j=P.a_rxn.to_numpy()[pj].astype(np.int64),
               n_g=ng)
    sizes = np.array([len(v) for v in by.values()], dtype=np.int64)
    stats = dict(n_base=int(len(rows)), n_groups=int(len(sizes)), n_groups_ge2=int((sizes >= 2).sum()),
                 n_base_in_groups_ge2=int(sizes[sizes >= 2].sum()), n_within=int(len(pi)),
                 group_size=_dist(sizes), group_size_ge2=_dist(sizes[sizes >= 2]),
                 within_over_base=float(len(pi) / len(rows)) if len(rows) else None)
    return out, stats


def _dist(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {}
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return dict(n=int(len(x)), min=float(q[0]), q25=float(q[1]), median=float(q[2]), q75=float(q[3]), max=float(q[4]),
                mean=float(x.mean()))


def test_groups(D, rows, S, gid_of_pair):
    pid = D.P.index.to_numpy()[np.sort(np.asarray(rows))].astype(str)
    s = S.reindex(pid)
    return pd.DataFrame(dict(pair_id=pid, gid=gid_of_pair.reindex(pid).to_numpy().astype(np.int64),
                             a_rxn=s.a_rxn.to_numpy().astype(np.int64), b_rxn=s.b_rxn.to_numpy().astype(np.int64),
                             fam=s.fam.to_numpy(), pos=s.pos.to_numpy(), Xb=s.Xb.astype(str).to_numpy(),
                             sigma=s.sigma.to_numpy(float), ref_kind=s.ref_kind.to_numpy(), core=s.core.to_numpy(),
                             in_P_main=s.in_P_main.astype(bool).to_numpy()))


def val_groups(D, rows, S=None, min_n=W.MIN_GROUP_VAL, positions=W.VNL_POS):
    """V_nl groups of a validation list (spec 5): positions in `rows` grouped by (b_rxn, fam, pos), pos in {p, m}, pairs
    with a finite sigma only, >= min_n such pairs -> list of dict(gid, key, idx, sigma)."""
    S = W.load_s_folds() if S is None else S
    G, gid_of_pair = W.group_table(S)
    rows = np.asarray(rows)
    pid = D.P.index.to_numpy()[rows].astype(str)
    s = S.reindex(pid)
    sig = s.sigma.to_numpy(float)
    ok = s.pos.isin(positions).to_numpy() & np.isfinite(sig)
    gid = gid_of_pair.reindex(pid).to_numpy()
    out = []
    for g in sorted(set(gid[ok].tolist())):
        idx = np.where(ok & (gid == g))[0]
        if len(idx) >= min_n:
            row = G.loc[G.gid == g].iloc[0]
            out.append(dict(gid=int(g), key=(int(row.b_rxn), str(row.fam), str(row.pos)), idx=idx, sigma=sig[idx]))
    return out


def data_fingerprint(D):
    return dict(source=D.source, usable_rule=D.usable_rule, inputs_sha256=D.sha, pair_list=D.pair_list_sha,
                n_reactions=int(len(D.rids)))


# ================================================================ build
def build(source="lf", root=None):
    import data_lf as DL
    t0 = time.time()
    root = Path(root) if root else default_root(source)
    if source != "lf" and os.path.abspath(root) == os.path.abspath(W.WITHIN):
        W.stop(f"source {source}: never written to the production folder {W.WITHIN}")
    W.assert_write_ok(root / "probe")
    root.mkdir(parents=True, exist_ok=True)
    D = DL.get_data(source)
    S = W.load_s_folds()
    agree = W.check_pairs_vs_sfolds(S, D.P)
    G, gid_of_pair = W.group_table(S)
    fp = dict(code=CODE_VERSION, within_sha256=W.sha256(Path(__file__)), w_common_sha256=W.sha256(HERE / "w_common.py"),
              s_folds_sha256=W.sha256(W.S_FOLDS), data=data_fingerprint(D), min_group_val=W.MIN_GROUP_VAL)
    fp = json.loads(json.dumps(W._jsonable(fp)))
    mpath = root / "meta.json"
    old = W.read_json(mpath) if mpath.is_file() else None
    if old is not None and old.get("fingerprint") != fp:
        diff = sorted(k for k in set(fp) | set(old.get("fingerprint") or {}) if (old.get("fingerprint") or {}).get(k) != fp.get(k))
        W.stop(f"{mpath}: built from other inputs / code (keys {diff}); move the folder away to rebuild")
    files, folds = {}, {}
    plan = [(k, key) for k in range(W.NF) for key in FIT_KEYS] + [("all", "train")]
    for k, key in plan:
        sp = D.split(k)
        rows = sp[key]
        arr, st = pool(D, rows, gid_of_pair)
        p = root / fname(k, key)
        if old is not None and p.is_file() and (old.get("files") or {}).get(p.name, {}).get("sha256") == W.sha256(p):
            rec = old["files"][p.name]
        else:
            W.write_npz(p, **arr)
            rec = dict(st, sha256=W.sha256(p), pairs_sha=sp.get("pairs_sha"),
                       base_sha=DL.list_sha(D._ids(np.sort(rows))))
        files[p.name] = rec
        folds.setdefault(str(k), {})[key] = {x: rec[x] for x in ("n_base", "n_groups", "n_groups_ge2", "n_within",
                                                                  "group_size_ge2", "within_over_base")}
    for k in range(W.NF):
        sp = D.split(k)
        T = test_groups(D, sp["test"], S, gid_of_pair)
        p = root / f"test_groups__f{k}.parquet"
        if not (old is not None and p.is_file() and (old.get("files") or {}).get(p.name, {}).get("sha256") == W.sha256(p)):
            W.write_parquet(p, T)
        gs = T.groupby("gid").size()
        files[p.name] = dict(sha256=W.sha256(p), n_pairs=int(len(T)), n_groups=int(len(gs)),
                             group_size=_dist(gs.to_numpy()), test_sha=sp.get("test_sha"),
                             n_graph_pairs=int(((gs + 1) * gs // 2).sum()))
        folds[str(k)]["test"] = {x: files[p.name][x] for x in ("n_pairs", "n_groups", "group_size", "n_graph_pairs")}
    meta = dict(fingerprint=fp, files=files, n_groups_total=int(len(G)), built=time.strftime("%Y-%m-%dT%H:%M:%S"),
                seconds=time.time() - t0, pair_table_vs_s_folds=agree)
    W.write_json(mpath, meta)
    rep = dict(phase="B", source=source, root=str(root), meta=str(mpath), meta_sha256=W.sha256(mpath), folds=folds,
               s_folds=s_folds_stats(S), source_stats=source_stats(D, S), spec_expectation=SPEC_EXPECT,
               rule="group key (b_rxn, fam, pos) of S_folds.csv; within pair for every i < j of a fit's usable training "
                    "pairs in one group; w = 2 / (n_g - 1)", pair_table_vs_s_folds=agree, seconds=time.time() - t0)
    out = (W.RESW / "B_report.json") if source == "lf" and os.path.abspath(root) == os.path.abspath(W.WITHIN) \
        else root / "B_report.json"
    W.write_json(out, rep)
    print(json.dumps(dict(report=str(out), folds={k: {kk: (vv.get("n_base"), vv.get("n_within"), vv.get("n_groups_ge2"))
                                                       for kk, vv in v.items() if kk in FIT_KEYS} for k, v in folds.items()},
                          s_folds=rep["s_folds"]["all"], seconds=round(rep["seconds"], 1)), indent=1, default=str))
    return 0


def _gstats(q):
    g = q.groupby(list(W.GROUP_KEY)).size().to_numpy()
    return dict(n_pairs=int(len(q)), n_groups=int(len(g)), n_groups_ge2=int((g >= 2).sum()),
                median_size=float(np.median(g)) if len(g) else None,
                median_size_ge2=float(np.median(g[g >= 2])) if (g >= 2).any() else None,
                within_pairs=int((g * (g - 1) // 2).sum()))


def s_folds_stats(S):
    """group statistics of S_folds.csv itself (spec 0-2 'scale (D1 S_folds, before splitting)')."""
    out = dict(all=_gstats(S), T_main=_gstats(S[S.in_T_main.astype(bool)]),
               reference=_gstats(S[S.ref_kind == "reference"]), anchor=_gstats(S[S.ref_kind == "anchor"]))
    out["by_pos_all"] = {p: _gstats(S[S.pos == p]) for p in sorted(S.pos.unique())}
    return out


def source_stats(D, S):
    pid = np.array(D.pair_list, dtype=str)
    return dict(P_source=_gstats(S.reindex(pid)),
                usable_T_main=_gstats(S.reindex(D.P.index.to_numpy()[D.split("all")["train"]].astype(str))))


# ================================================================ loading (train_w, tests, within_predict)
def load_pool(D, k, key, root=None, check=True):
    """npz of (fold k | 'all', key) after checks against meta.json and the data object D: file sha256 = meta record,
    data fingerprint = D, base rows = D.split(k)[key], arrays consistent with D (pid, ra, rb, y bitwise, w)."""
    import data_lf as DL
    root = Path(root) if root else default_root(D.source)
    mpath = root / "meta.json"
    p = root / fname(k, key)
    if not (mpath.is_file() and p.is_file()):
        W.not_ready(f"{p} / {mpath} missing (within.py build)")
    meta = W.read_json(mpath)
    rec = (meta.get("files") or {}).get(p.name) or {}
    sha = W.sha256(p)
    if rec.get("sha256") != sha:
        W.stop(f"{p} sha256 differs from {mpath}")
    if check:
        fp = json.loads(json.dumps(W._jsonable(data_fingerprint(D))))
        if (meta.get("fingerprint") or {}).get("data") != fp:
            W.stop(f"{mpath}: built on other data than the current {D.source} DeployData")
    with np.load(p) as z:
        arr = {x: z[x] for x in z.files}
    if check:
        rows = np.sort(D.split(k)[key])
        if rec.get("base_sha") != DL.list_sha(D._ids(rows)):
            W.stop(f"{p}: base pairs differ from DeployData split({k})['{key}']")
        ids = D.P.index.to_numpy()
        ok = (np.array_equal(ids[arr["pi"]].astype(str), arr["pid_i"]) and np.array_equal(ids[arr["pj"]].astype(str), arr["pid_j"])
              and np.array_equal(D.ra[arr["pi"]], arr["ra"]) and np.array_equal(D.ra[arr["pj"]], arr["rb"])
              and np.array_equal(D.y[arr["pi"]] - D.y[arr["pj"]], arr["y"])
              and np.array_equal(2.0 / (arr["n_g"] - 1), arr["w"]))
        if not ok:
            W.stop(f"{p}: arrays inconsistent with the data object")
    arr["sha256"] = sha
    arr["file"] = str(p)
    return arr


def load_test_groups(k, root=None, source="lf"):
    root = Path(root) if root else default_root(source)
    p = root / f"test_groups__f{k}.parquet"
    mpath = root / "meta.json"
    if not (p.is_file() and mpath.is_file()):
        W.not_ready(f"{p} missing (within.py build)")
    if ((W.read_json(mpath).get("files") or {}).get(p.name) or {}).get("sha256") != W.sha256(p):
        W.stop(f"{p} sha256 differs from {mpath}")
    T = pd.read_parquet(p)
    T["pair_id"] = T.pair_id.astype(str)
    return T


def main(argv):
    a = argv[1:]
    src, root = "lf", None
    if "--source" in a:
        i = a.index("--source")
        src = a[i + 1]
        a = a[:i] + a[i + 2:]
    if "--root" in a:
        i = a.index("--root")
        root = a[i + 1]
        a = a[:i] + a[i + 2:]
    if a == ["build"]:
        return build(src, root)
    if a == ["summary"]:
        import data_lf as DL
        S = W.load_s_folds()
        D = DL.get_data(src)
        print(json.dumps(dict(s_folds=s_folds_stats(S), source=source_stats(D, S)), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
