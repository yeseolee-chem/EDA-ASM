#!/usr/bin/env python3
"""GEO1b Phase E1: structure evaluation on the D1 test folds (G2 input only), descriptive (docs/specs/GEO1B.md 4-1).
The decision is made on energy in GEO2b; here only numbers and CIs. GPU env on a CPU partition.  python evaluate.py
Writes results/E_main.csv, E_pairs.csv, E_rxn.csv, E_by_bin.csv, E_by_dipole.csv, E_report.json (D1, unchanged) and
results/E_rxn_d0.csv (decision D0D1: forming-bond errors of the D0 fold-k reactions predicted by final_f{k}, same
columns as E_rxn.csv + fold, g2c, twin_of, twin_kind, twin_m1; descriptive, for the GEO2b build_geo cross-check).
Every csv is written atomically (GC.write_csv); E_report.json is written last.
The G2-raw expectation is results/G2RAW_baseline.json (data.py baseline, decision A-G2c), not a hard-coded number."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import evaluate_preds, rmsd_bin, rxn_errors  # noqa: E402

B_BOOT, SEED, THR = 2000, 20261026, 0.005
SEEDS = (0, 1, 2)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def preds_for(name, folds, fold_of):
    """uid -> xyz for the given test folds; None if a fold file is missing. 'model' = 3-seed coordinate mean (GEO1).
    Every uid must belong to the fold of the file it is in (fold_of: uid -> record fold) and appear in one fold only
    (D0D1: final_f{k} also predicts D0 fold k)."""
    P = {}
    for k in folds:
        if name == "model":
            S = [GC.GEO / "preds" / f"final_f{k}_s{s}.pkl" for s in SEEDS]
            if not all(p.is_file() for p in S):
                return None
            parts = [load_pkl(p) for p in S]
            assert all(set(q) == set(parts[0]) for q in parts), f"seed prediction sets differ in fold {k}"
            new = {u: np.mean([q[u] for q in parts], axis=0) for u in parts[0]}
        else:                                             # seed<s>
            p = GC.GEO / "preds" / f"final_f{k}_s{name[4:]}.pkl"
            if not p.is_file():
                return None
            new = load_pkl(p)
        wrong = [u for u in new if fold_of.get(u) != k or u in P]
        assert not wrong, f"{name} fold {k}: {len(wrong)} predictions of another fold or a second time, e.g. {wrong[:5]}"
        P.update(new)
    return P


def d0_rxn_table(P, recs):
    """E_rxn-style rows (metrics.evaluate_preds table_rxn columns) for D0 records; evaluate_preds itself is D1 only."""
    rows = []
    for r in recs:
        x = np.asarray(P[r["uid"]], float)
        e = rxn_errors(x, r["lab"], r["pairs"])
        heavy = r["types"] != 0
        rows.append(dict(rxn_id=r["rxn_id"], core=r["core"], e_short=float(e[0]), e_long=float(e[1]),
                         abs_max=float(np.abs(e).max()), map_rmsd_g2=float(r.get("map_rmsd_g2", np.nan)),
                         bin=rmsd_bin(r.get("map_rmsd_g2")), fold=int(r["fold"]), g2c=r.get("g2c"),
                         twin_of="" if r.get("twin_of") is None else int(r["twin_of"]),
                         twin_kind=r.get("twin_kind") or "", twin_m1=bool(r.get("twin_m1")),     # GB-twin-m1
                         rmsd_heavy=GC.kabsch_align(x[heavy], r["lab"][heavy], reflect=True)[1]))
    return pd.DataFrame(rows, columns=["rxn_id", "core", "e_short", "e_long", "abs_max", "map_rmsd_g2", "bin", "fold",
                                       "g2c", "twin_of", "twin_kind", "twin_m1", "rmsd_heavy"])


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
    ex = GC.load_g2raw_baseline()                         # decision A-G2c: frozen before training (data.py baseline)
    if ex is None or ex.get("records_sha256") != GC.sha256(GC.GEO / "data" / "records.pkl"):
        print(f"STOP: {GC.G2RAW_BASELINE} missing or computed from other records (run data.py baseline before "
              "training)", file=sys.stderr)
        return 3
    folds = [0, 1, 2, 3, 4]
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in folds and r["xyz_g2"] is not None]
    test_d0 = [r for r in recs if r["src"] == "D0" and not r["overlap"] and r["fold"] in folds
               and r["xyz_g2"] is not None]                # D0D1: every one is predicted by final_f{fold}
    fold_of = {r["uid"]: r["fold"] for r in test + test_d0}
    targets = {"g2_raw": {r["uid"]: r["xyz_g2"] for r in test}}
    for name in ("model",) + tuple(f"seed{s}" for s in SEEDS):
        P = preds_for(name, folds, fold_of)
        if P is not None:
            targets[name] = P
    want = {r["uid"] for r in test + test_d0}
    if "model" not in targets or set(targets["model"]) != want:
        got = set(targets.get("model", {}))
        print(f"STOP: model predictions missing for some test reactions (D1 {len({r['uid'] for r in test} - got)}, "
              f"D0 {len({r['uid'] for r in test_d0} - got)} missing; {len(got - want)} unexpected)", file=sys.stderr)
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
    GC.RES.mkdir(parents=True, exist_ok=True)              # every csv is written atomically (GC.write_csv, GB-atomic)
    GC.write_csv(pd.DataFrame(res).T, GC.RES / "E_main.csv")
    GC.write_csv(tabs["model"]["table_pairs"].merge(tabs["g2_raw"]["table_pairs"][["pair_id", "pe"]], on="pair_id",
                                                    suffixes=("_model", "_g2raw")), GC.RES / "E_pairs.csv", index=False)
    GC.write_csv(tabs["model"]["table_rxn"].merge(tabs["g2_raw"]["table_rxn"][["rxn_id", "e_short", "e_long",
                                                                               "rmsd_heavy"]],
                                                  on="rxn_id", suffixes=("_model", "_g2raw")),
                 GC.RES / "E_rxn.csv", index=False)
    # D0D1: D0 fold-k reactions predicted by final_f{k} (3-seed mean), descriptive; E_rxn.csv layout + fold, g2c, twin_of,
    # twin_kind, twin_m1 (GB-twin-m1: twin_m1 = a T1 fold -1 D1 twin, which trained in every fold, P-twin-m1)
    D0m = d0_rxn_table(targets["model"], test_d0)
    D0g = d0_rxn_table({r["uid"]: r["xyz_g2"] for r in test_d0}, test_d0)
    GC.write_csv(D0m.merge(D0g[["rxn_id", "e_short", "e_long", "rmsd_heavy"]], on="rxn_id",
                           suffixes=("_model", "_g2raw")), GC.RES / "E_rxn_d0.csv", index=False)

    def d0_summ(D):
        return dict(n=int(len(D)), rxn_form_abs_median=float(np.median(np.abs(np.r_[D.e_short, D.e_long])))
                    if len(D) else float("nan"), rmsd_heavy_median=float(D.rmsd_heavy.median()) if len(D) else float("nan"))
    d0_rep = dict(note="descriptive only (D0 has no reference pairs); E_rxn_d0.csv", n_test=len(test_d0),
                  model=d0_summ(D0m), g2_raw=d0_summ(D0g),
                  by_fold={str(k): dict(model=d0_summ(D0m[D0m.fold == k]), g2_raw=d0_summ(D0g[D0g.fold == k]))
                           for k in folds},
                  n_twin_m1=int(D0m.twin_m1.sum()), twin_kind_counts={str(k): int(v) for k, v in
                                                                      D0m.twin_kind.value_counts().items()},
                  without_twin_m1=dict(model=d0_summ(D0m[~D0m.twin_m1]), g2_raw=d0_summ(D0g[~D0g.twin_m1])),
                  twin_m1_note="twin_m1 = a stereo-free D1 twin of T1 fold -1, in every final_f{k} training set while "
                               "this D0 reaction is tested (DECISIONS_PAIR P-twin-m1, accepted limitation)")
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
    GC.write_csv(pd.DataFrame(rows_b), GC.RES / "E_by_bin.csv", index=False)
    GC.write_csv(pd.DataFrame(rows_d), GC.RES / "E_by_dipole.csv", index=False)
    tol = GC.G2RAW_TOL
    g2ok = (res["g2_raw"]["n_primary"] == ex["n"] and abs(res["g2_raw"]["primary_share"] - ex["share"]) <= tol["share"]
            and abs(res["g2_raw"]["pair_pe_median_para"] - ex["median"]) <= tol["median"] and len(test) == ex["n_test"])
    GC.write_json(GC.RES / "E_report.json", dict(
        decision="none here: structure is descriptive; GEO2b decides on energy", primary=res, ci=ci, n_test=len(test),
        g2_raw_expected={k: ex.get(k) for k in ("n", "share", "median", "n_test", "records_sha256")},
        g2_raw_baseline=str(GC.G2RAW_BASELINE), g2_raw_matches_expected=bool(g2ok), thr_A=THR, n_boot=B_BOOT,
        g1_used=False, g2_input="G2c (decision A-G2c)", d0=d0_rep))
    d = ci["model_vs_g2_raw"]["ci_diff"]
    print(f"model {res['model']['primary_share']:.3f} vs g2_raw {res['g2_raw']['primary_share']:.3f} "
          f"(all para pairs, n={res['model']['n_primary']}), diff CI [{d[0]:+.3f}, {d[1]:+.3f}]; "
          f"D0 {len(test_d0)} reactions -> E_rxn_d0.csv", flush=True)
    if not g2ok:
        print(f"STOP: the G2-raw baseline does not reproduce {GC.G2RAW_BASELINE.name} (data or metric changed)",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
