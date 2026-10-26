#!/usr/bin/env python3
"""Phase E: GEO-1 evaluation on the D1 test folds (G2 input only). python evaluate.py"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import evaluate_preds  # noqa: E402

B_BOOT, SEED, THR = 2000, 20261026, 0.005


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def preds_for(name, folds):
    """uid -> xyz for the given test folds; None if a fold file is missing."""
    P = {}
    for k in folds:
        if name == "model":
            S = [GC.GEO / "preds" / f"final_f{k}_s{s}.pkl" for s in (0, 1, 2)]
            if not all(p.is_file() for p in S):
                return None
            parts = [load_pkl(p) for p in S]
            assert all(set(q) == set(parts[0]) for q in parts), f"seed prediction sets differ in fold {k}"
            P.update({u: np.mean([q[u] for q in parts], axis=0) for u in parts[0]})   # same input frame -> averageable
        elif name.startswith("seed"):
            p = GC.GEO / "preds" / f"final_f{k}_s{name[4:]}.pkl"
            if not p.is_file():
                return None
            P.update(load_pkl(p))
        else:
            p = GC.GEO / "preds" / f"abl_{name}_f{k}.pkl"
            if not p.is_file():
                return None
            P.update(load_pkl(p))
    return P


def core_counts(F):
    """per core: (#primary pairs with pe <= THR, #primary pairs)."""
    prim = F[(F.pos == "p") & F.same]
    g = prim.groupby("core").pe
    return pd.DataFrame(dict(ok=g.apply(lambda s: int((s <= THR).sum())), n=g.size()))


def boot_diff(Fa, Fb, rng):
    """core bootstrap of share(a), share(b), share(a) - share(b) on the same primary pairs."""
    A, B = core_counts(Fa), core_counts(Fb)
    assert A.index.equals(B.index) and (A.n == B.n).all(), "primary pair sets differ between the compared targets"
    cores = A.index.to_numpy()
    out = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(cores), len(cores), replace=True)
        na = A.n.to_numpy()[pick].sum()
        sa, sb = A.ok.to_numpy()[pick].sum() / na, B.ok.to_numpy()[pick].sum() / na
        out.append((sa, sb, sa - sb))
    q = np.percentile(np.array(out), [2.5, 97.5], axis=0)
    return dict(ci_a=list(q[:, 0]), ci_b=list(q[:, 1]), ci_diff=list(q[:, 2]))


def main():
    with open(GC.GEO / "data" / "records.pkl", "rb") as f:
        recs = pickle.load(f)
    folds = [0, 1, 2, 3, 4]
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in folds and r["xyz_g2"] is not None]
    by_fold = {k: [r for r in test if r["fold"] == k] for k in folds}
    targets = {"g2_raw": {r["uid"]: r["xyz_g2"] for r in test}}
    for name in ("model", "seed0", "seed1", "seed2", "no_d0", "q1_only", "r0"):
        avail = [k for k in folds if preds_for(name, [k]) is not None]
        if avail:
            targets[name] = preds_for(name, avail)
    if "model" not in targets or len(targets["model"]) != len(test):
        print("STOP: model predictions missing for some test reactions", file=sys.stderr)
        return 3
    res, tabs = {}, {}
    for name, P in targets.items():
        uids = set(P)
        m = evaluate_preds(P, [r for r in test if r["uid"] in uids])
        tabs[name] = m
        res[name] = {k: v for k, v in m.items() if not k.startswith("table_")}
        res[name]["folds"] = sorted({r["fold"] for r in test if r["uid"] in uids})
    rng = np.random.default_rng(SEED)
    ci = {"model_vs_g2_raw": boot_diff(tabs["model"]["table_pairs"], tabs["g2_raw"]["table_pairs"], rng)}
    for name in ("no_d0", "q1_only", "r0"):
        if name in tabs:
            fl = res[name]["folds"]
            Pm = {u: x for u, x in targets["model"].items() if int(u.split("_")[1]) in
                  {r["rxn_id"] for k in fl for r in by_fold[k]}}
            mm = evaluate_preds(Pm, [r for k in fl for r in by_fold[k]], with_rmsd=False)
            ci[f"model_vs_{name}"] = dict(folds=fl, model_share=mm["primary_share"],
                                          other_share=res[name]["primary_share"],
                                          **boot_diff(mm["table_pairs"], tabs[name]["table_pairs"], rng))
    s_m, d_lo = res["model"]["primary_share"], ci["model_vs_g2_raw"]["ci_diff"][0]
    verdict = "pass" if (s_m >= 0.5 and d_lo > 0) else ("partial" if d_lo > 0 else "fail")
    GC.RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).T.to_csv(GC.RES / "E_main.csv")
    tabs["model"]["table_pairs"].merge(tabs["g2_raw"]["table_pairs"][["pair_id", "pe"]], on="pair_id",
                                       suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_pairs.csv", index=False)
    tabs["model"]["table_rxn"].merge(tabs["g2_raw"]["table_rxn"][["rxn_id", "e_short", "e_long", "rmsd_heavy"]],
                                     on="rxn_id", suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_rxn.csv", index=False)
    rows_c, rows_d = [], []
    for name in ("model", "g2_raw"):
        F = tabs[name]["table_pairs"]
        for (pos, cls), g in F.groupby(["pos", "cls"]):
            rows_c.append(dict(target=name, pos=pos, cls=cls, n=len(g), share=float((g.pe <= THR).mean()),
                               median=float(g.pe.median())))
        for dip, g in F[(F.pos == "p") & F.same].groupby("dipole"):
            rows_d.append(dict(target=name, dipole=dip, n=len(g), share=float((g.pe <= THR).mean()),
                               median=float(g.pe.median())))
    pd.DataFrame(rows_c).to_csv(GC.RES / "E_by_class.csv", index=False)
    pd.DataFrame(rows_d).to_csv(GC.RES / "E_by_dipole.csv", index=False)
    GC.write_json(GC.RES / "E_report.json", dict(verdict=verdict, primary=res, ci=ci, n_test=len(test),
                                                 g2_raw_expected=GC.G2RAW_PRIMARY, thr_A=THR, n_boot=B_BOOT))
    print(f"verdict {verdict}: model {s_m:.3f} vs g2_raw {res['g2_raw']['primary_share']:.3f}, "
          f"diff CI {ci['model_vs_g2_raw']['ci_diff']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
