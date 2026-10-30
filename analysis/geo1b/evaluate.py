#!/usr/bin/env python3
"""GEO1b Phase E1: structure evaluation on the D1 test folds (G2 input only), descriptive (docs/specs/GEO1B.md 4-1).
The decision is made on energy in GEO2b; here only numbers and CIs. GPU env on a CPU partition.  python evaluate.py
Writes results/E_main.csv, E_pairs.csv, E_rxn.csv, E_by_bin.csv, E_by_dipole.csv, E_report.json."""
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
SEEDS = (0, 1, 2)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def preds_for(name, folds):
    """uid -> xyz for the given test folds; None if a fold file is missing. 'model' = 3-seed coordinate mean (GEO1)."""
    P = {}
    for k in folds:
        if name == "model":
            S = [GC.GEO / "preds" / f"final_f{k}_s{s}.pkl" for s in SEEDS]
            if not all(p.is_file() for p in S):
                return None
            parts = [load_pkl(p) for p in S]
            assert all(set(q) == set(parts[0]) for q in parts), f"seed prediction sets differ in fold {k}"
            P.update({u: np.mean([q[u] for q in parts], axis=0) for u in parts[0]})
        else:                                             # seed<s>
            p = GC.GEO / "preds" / f"final_f{k}_s{name[4:]}.pkl"
            if not p.is_file():
                return None
            P.update(load_pkl(p))
    return P


def core_counts(F):
    """per core over ALL para pairs: (#pairs with pe <= THR, #pairs)."""
    prim = F[F.pos == "p"]
    g = prim.groupby("core").pe
    return pd.DataFrame(dict(ok=g.apply(lambda s: int((s <= THR).sum())), n=g.size()))


def boot_diff(Fa, Fb, rng):
    """core bootstrap of share(a), share(b), share(a) - share(b) on the same para pairs."""
    A, B = core_counts(Fa), core_counts(Fb)
    assert A.index.equals(B.index) and (A.n == B.n).all(), "para pair sets differ between the compared targets"
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
    recs = GC.load_records()
    if any(r.get("xyz_g1") is not None or r.get("same") is not None for r in recs):
        print("STOP: a record carries G1 data (xyz_g1 / same); GEO1b records must be G1-free", file=sys.stderr)
        return 3
    folds = [0, 1, 2, 3, 4]
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in folds and r["xyz_g2"] is not None]
    targets = {"g2_raw": {r["uid"]: r["xyz_g2"] for r in test}}
    for name in ("model",) + tuple(f"seed{s}" for s in SEEDS):
        P = preds_for(name, folds)
        if P is not None:
            targets[name] = P
    if "model" not in targets or len(targets["model"]) != len(test):
        print("STOP: model predictions missing for some test reactions", file=sys.stderr)
        return 3
    res, tabs = {}, {}
    for name, P in targets.items():
        m = evaluate_preds(P, test)
        tabs[name] = m
        res[name] = {k: v for k, v in m.items() if not k.startswith("table_")}
    rng = np.random.default_rng(SEED)
    ci = {"model_vs_g2_raw": boot_diff(tabs["model"]["table_pairs"], tabs["g2_raw"]["table_pairs"], rng)}
    for s in SEEDS:
        if f"seed{s}" in tabs:
            ci[f"seed{s}_vs_g2_raw"] = boot_diff(tabs[f"seed{s}"]["table_pairs"], tabs["g2_raw"]["table_pairs"], rng)
    GC.RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).T.to_csv(GC.RES / "E_main.csv")
    tabs["model"]["table_pairs"].merge(tabs["g2_raw"]["table_pairs"][["pair_id", "pe"]], on="pair_id",
                                       suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_pairs.csv", index=False)
    tabs["model"]["table_rxn"].merge(tabs["g2_raw"]["table_rxn"][["rxn_id", "e_short", "e_long", "rmsd_heavy"]],
                                     on="rxn_id", suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_rxn.csv", index=False)
    rows_b, rows_d = [], []
    for name in ("model", "g2_raw"):
        F, D = tabs[name]["table_pairs"], tabs[name]["table_rxn"]
        for (pos, b), g in F.groupby(["pos", "bin"]):
            rows_b.append(dict(target=name, level="pair", pos=pos, bin=b, n=len(g),
                               share=float((g.pe <= THR).mean()), median=float(g.pe.median())))
        for b, g in D.groupby("bin"):
            rows_b.append(dict(target=name, level="rxn", pos="", bin=b, n=len(g), share=float("nan"),
                               median=float(np.median(np.abs(np.concatenate([g.e_short, g.e_long]))))))
        for dip, g in F[F.pos == "p"].groupby("dipole"):
            rows_d.append(dict(target=name, dipole=dip, n=len(g), share=float((g.pe <= THR).mean()),
                               median=float(g.pe.median())))
    pd.DataFrame(rows_b).to_csv(GC.RES / "E_by_bin.csv", index=False)
    pd.DataFrame(rows_d).to_csv(GC.RES / "E_by_dipole.csv", index=False)
    ex = GC.G2RAW_PRIMARY
    g2ok = (res["g2_raw"]["n_primary"] == ex["n"] and abs(res["g2_raw"]["primary_share"] - ex["share"]) <= 0.005
            and abs(res["g2_raw"]["pair_pe_median_para"] - ex["median"]) <= 0.001)
    GC.write_json(GC.RES / "E_report.json", dict(
        decision="none here: structure is descriptive; GEO2b decides on energy", primary=res, ci=ci, n_test=len(test),
        g2_raw_expected=ex, g2_raw_matches_expected=bool(g2ok), thr_A=THR, n_boot=B_BOOT, g1_used=False))
    d = ci["model_vs_g2_raw"]["ci_diff"]
    print(f"model {res['model']['primary_share']:.3f} vs g2_raw {res['g2_raw']['primary_share']:.3f} "
          f"(all para pairs, n={res['model']['n_primary']}), diff CI [{d[0]:+.3f}, {d[1]:+.3f}]", flush=True)
    if not g2ok:
        print("STOP: the G2-raw baseline does not reproduce GC.G2RAW_PRIMARY (data or metric changed)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
