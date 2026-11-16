#!/usr/bin/env python3
"""ESPB Phase C (docs/specs/ESPB.md 7): pair predictions of the 10 candidates and the representative "ESPB" per target.

    python collect.py           PYENV=espley (jobs/py.slurm, submitted from analysis/espb), CPU

Order (C-4, enforced in code): 1. only the unit json files and the OOF rows of the E-A unit parquets (pyarrow row filter
role == "oof"; no test row is loaded) -> inner ΔΔ score of every candidate -> results/C_selection.csv, its sha256 kept;
2. only then the test rows -> results/C_predictions.parquet (pair_id, ref_kind, pos, fold, arm, model, target, y, yhat,
yhat_seed_sd), results/C_reactions.parquet (E-A per-reaction test predictions), results/C_espb.parquet (the chosen
candidate per target, model "ESPB").
Inner score (7-2): per outer fold k and inner fold j, the MAE over the inner val pairs of j (training pairs of fold k
whose core is in inner fold j, results/P_inner_folds.csv); mean over j, then over k. E-A: the OOF reaction predictions
give the pair ΔΔ (strain = d1 + d2 of the same model); E-B: the best cell's inner fold MAEs in the unit json (NN: the cv
stage). Choice = lowest score; ties: EA before EB, then Ridge, KRR, SVR, NN2, NN4.
Gates C-1 .. C-4 -> results/C_report.json. Exit 0 pass, 2 = units missing (resubmit training), 3 = STOP.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eb_common as C  # noqa: E402

C.path_setup()
from gnn3 import metrics as M  # noqa: E402

TOL = 1e-9
EA_PARTS = {"strain": ("d1", "d2"), "inter": ("inter",), "solv": ("solv",), "barrier": ("barrier",)}
_STATE = {"test_rows_read": False}


def unit_paths(arm, model, target, k):
    base = (C.UNITS_EA if arm == "EA" else C.UNITS_EB) / f"{model}__{target}__f{k}"
    return base.with_suffix(".json"), base.with_suffix(".parquet")


def expected_units():
    out = [("EA", m, t, k) for m in C.MODELS for t in C.TARGETS_EA for k in range(C.N_FOLDS)]
    out += [("EB", m, t, k) for m in C.MODELS for t in C.TARGETS_EB for k in range(C.N_FOLDS)]
    return out


def read_rows(path, role):
    """rows of one E-A unit parquet with the given role only (pyarrow filter: the other role is never loaded)"""
    if role == "test":
        _STATE["test_rows_read"] = True
    return pd.read_parquet(path, filters=[("role", "==", role)])


def pairs_table():
    F = pd.read_csv(C.PAIRS, usecols=["pair_id", "ref_kind", "pos", "a_rxn", "b_rxn", "core"])
    A = pd.read_csv(C.S_FOLDS, usecols=["pair_id", "fold_T1"])
    return F.merge(A, on="pair_id", how="left", validate="1:1").set_index("pair_id")


def selection(P, truth):
    """inner ΔΔ score of the 10 candidates per target, from json files and OOF rows only"""
    inner = pd.read_csv(C.P_INNER)
    rows = []
    for t in C.CHANNELS:
        for arm in C.ARMS:
            for m in C.MODELS:
                per_fold = []
                for k in range(C.N_FOLDS):
                    unit = C.split_unit(k)
                    tr = P.loc[unit["train"]]
                    jmap = inner[inner.outer_fold == k].set_index("core").inner_fold
                    jj = tr.core.map(jmap).to_numpy(int)
                    if arm == "EB":
                        rec = C.read_json(unit_paths("EB", m, t, k)[0])
                        per_fold.append([float(v) for v in rec["inner_fold_mae"]])
                        continue
                    f = None
                    for part in EA_PARTS[t]:
                        o = read_rows(unit_paths("EA", m, part, k)[1], "oof").set_index("rxn_id")
                        f = o[["yhat", "inner_fold"]].copy() if f is None else f.assign(yhat=f.yhat + o.yhat.loc[f.index])
                        if f is not None and part != EA_PARTS[t][0] and not (o.inner_fold.loc[f.index] == f.inner_fold).all():
                            raise SystemExit(f"STOP: {m} fold {k}: d1 / d2 OOF inner folds differ")
                    fold_mae = []
                    for j in range(C.N_INNER):
                        v = tr[jj == j]
                        ja, jb = f.inner_fold.loc[v.a_rxn].to_numpy(int), f.inner_fold.loc[v.b_rxn].to_numpy(int)
                        if not ((ja == j).all() and (jb == j).all()):
                            raise SystemExit(f"STOP: {m} {t} fold {k}: inner val pair ends outside inner fold {j}")
                        dd = f.yhat.loc[v.a_rxn].to_numpy(float) - f.yhat.loc[v.b_rxn].to_numpy(float)
                        fold_mae.append(float(np.mean(np.abs(dd - truth.loc[v.index, t].to_numpy(float)))))
                    per_fold.append(fold_mae)
                score = float(np.mean([np.mean(x) for x in per_fold]))
                rows.append(dict(target=t, arm=arm, model=m, candidate=C.model_name(arm, m), score=score,
                                 per_fold=json.dumps([float(np.mean(x)) for x in per_fold])))
    S = pd.DataFrame(rows)
    S["tie"] = [C.ARMS.index(a) * 10 + C.MODELS.index(m) for a, m in zip(S.arm, S.model)]
    S = S.sort_values(["target", "score", "tie"], kind="mergesort")
    S["rank"] = S.groupby("target").cumcount() + 1
    S["chosen"] = S["rank"] == 1
    order = {t: i for i, t in enumerate(C.CHANNELS)}
    S = S.sort_values(["target", "rank"], key=lambda s: s.map(order) if s.name == "target" else s, kind="mergesort")
    return S.drop(columns="tie").reset_index(drop=True)


def ea_predictions(P, truth):
    """E-A test pair ΔΔ per model / target / fold; NN: per-seed ΔΔ, mean = yhat, std = yhat_seed_sd"""
    R = pd.read_csv(C.P_REACTIONS).set_index("rxn_id")
    out, rx = [], []
    for m in C.MODELS:
        for k in range(C.N_FOLDS):
            unit = C.split_unit(k)
            te = P.loc[unit["test"]]
            parts = {}
            for t in C.TARGETS_EA:
                d = read_rows(unit_paths("EA", m, t, k)[1], "test").set_index("rxn_id")
                parts[t] = d
                rx.append(pd.DataFrame(dict(rxn_id=d.index.to_numpy(), fold=k, model=m, target=t,
                                            y=R.loc[d.index, {"d1": "d1_kcal", "d2": "d2_kcal"}.get(t, t)].to_numpy(float),
                                            yhat=d.yhat.to_numpy(float))))
            seed_cols = [f"yhat_s{s}" for s in C.NN_FINAL_SEEDS] if m.startswith("NN") else ["yhat"]
            for c in C.CHANNELS:
                f = sum(parts[p][seed_cols] for p in EA_PARTS[c])          # per-seed reaction predictions
                dd = f.loc[te.a_rxn].to_numpy(float) - f.loc[te.b_rxn].to_numpy(float)   # [pairs, seeds]
                out.append(pd.DataFrame(dict(pair_id=te.index.to_numpy(), ref_kind=te.ref_kind.to_numpy(),
                                             pos=te.pos.to_numpy(), fold=k, arm="EA", model=m, target=c,
                                             y=truth.loc[te.index, c].to_numpy(float), yhat=dd.mean(axis=1),
                                             yhat_seed_sd=dd.std(axis=1) if dd.shape[1] > 1 else 0.0)))
    return pd.concat(out, ignore_index=True), pd.concat(rx, ignore_index=True)


def eb_predictions(P):
    out = []
    for m in C.MODELS:
        for t in C.TARGETS_EB:
            for k in range(C.N_FOLDS):
                _STATE["test_rows_read"] = True
                d = pd.read_parquet(unit_paths("EB", m, t, k)[1])
                seed_cols = [f"yhat_s{s}" for s in C.NN_FINAL_SEEDS]
                sd = d[seed_cols].to_numpy(float).std(axis=1) if m.startswith("NN") else np.zeros(len(d))
                out.append(pd.DataFrame(dict(pair_id=d.pair_id.to_numpy(), ref_kind=d.ref_kind.to_numpy(),
                                             pos=P.loc[d.pair_id, "pos"].to_numpy(), fold=k, arm="EB", model=m, target=t,
                                             y=d.y.to_numpy(float), yhat=d.yhat.to_numpy(float), yhat_seed_sd=sd)))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    t0 = time.time()
    missing = [f"{a}/{m}__{t}__f{k}" for a, m, t, k in expected_units()
               if not all(p.is_file() for p in unit_paths(a, m, t, k))]
    if missing:
        C.write_json(C.RES / "C_report.json", dict(all_pass=False, failed=["C-3"], n_missing=len(missing),
                                                   missing=missing[:50], note="units missing: resubmit training"))
        print(f"{len(missing)} units missing (exit 2): {missing[:10]}")
        return 2
    bad_units = []                                                  # C-3: the recorded unit gates
    for a, m, t, k in expected_units():
        r = C.read_json(unit_paths(a, m, t, k)[0])
        keys = ("gate_A2_finite", "gate_A4_seeds") if a == "EA" else ("gate_B2_finite", "gate_B4_seeds", "gate_M1")
        if not all(r.get(x) is True for x in keys):
            bad_units.append(f"{a}/{m}__{t}__f{k}: " + ", ".join(f"{x}={r.get(x)}" for x in keys))
    if bad_units:
        C.write_json(C.RES / "C_report.json", dict(all_pass=False, failed=["C-3"], bad_units=bad_units[:50],
                                                   note="a unit records a failed training gate (A-2/A-4/B-2/B-4/M-1)"))
        print(f"STOP (C-3): {len(bad_units)} units record a failed gate: {bad_units[:5]}")
        return 3
    P = pairs_table()
    truth, _ = M.channel_truth(C.PAIRS)
    # 1. selection from json + OOF only
    S = selection(P, truth)
    score_ok = bool(np.isfinite(S.score.to_numpy(float)).all()
                    and all(np.isfinite(json.loads(x)).all() for x in S.per_fold))
    c4_before = not _STATE["test_rows_read"]
    C.write_atomic(C.RES / "C_selection.csv", S.to_csv(index=False))
    sel_sha = C.sha256(C.RES / "C_selection.csv")
    sel_time = time.strftime("%Y-%m-%d %H:%M:%S")
    # 2. test predictions
    ea, rx = ea_predictions(P, truth)
    eb = eb_predictions(P)
    Cp = pd.concat([ea, eb], ignore_index=True)
    chosen = S[S.chosen].set_index("target")
    espb = pd.concat([Cp[(Cp.target == t) & (Cp.arm == chosen.loc[t, "arm"]) & (Cp.model == chosen.loc[t, "model"])]
                      .assign(candidate=chosen.loc[t, "candidate"]) for t in C.CHANNELS], ignore_index=True)
    espb = espb.assign(arm="ESPB", model="ESPB")
    # gates
    G = pd.concat([pd.read_parquet(C.GNN3_CV / f"gnn3_fold{k}.parquet") for k in range(C.N_FOLDS)], ignore_index=True)
    G["pair_id"] = G.pair_id.astype(str)
    c1, c2 = {}, {}
    for t in C.CHANNELS:
        g = G[G.target == t]
        gset = set(zip(g.pair_id, g.fold.astype(int)))
        gy = g.set_index("pair_id").y
        for arm in C.ARMS:
            for m in C.MODELS:
                d = Cp[(Cp.target == t) & (Cp.arm == arm) & (Cp.model == m)]
                c1[f"{arm}-{m}/{t}"] = dict(n=int(len(d)), equal=set(zip(d.pair_id, d.fold.astype(int))) == gset
                                            and len(d) == len(gset) == C.N_TEST)
                c2[f"{arm}-{m}/{t}"] = dict(
                    truth_max_abs=float(np.abs(d.y.to_numpy(float) - truth.loc[d.pair_id, t].to_numpy(float)).max()),
                    gnn3_y_max_abs=float(np.abs(d.y.to_numpy(float) - gy.loc[d.pair_id].to_numpy(float)).max()))
    gates = {"C-1": dict(per_candidate=c1, ok=all(v["equal"] for v in c1.values())),
             "C-2": dict(per_candidate=c2, tol=TOL, ok=all(max(v.values()) <= TOL for v in c2.values())),
             "C-3": dict(n_units=len(expected_units()), missing=0, unit_gates_ok=True, inner_scores_finite=score_ok,
                         nan=bool(Cp[["y", "yhat", "yhat_seed_sd"]].isna().any().any() or rx.isna().any().any()),
                         ok=score_ok and bool(np.isfinite(Cp[["y", "yhat", "yhat_seed_sd"]].to_numpy(float)).all()
                                              and np.isfinite(rx[["y", "yhat"]].to_numpy(float)).all())),
             "C-4": dict(selection_written_before_test_rows=c4_before, selection_sha256=sel_sha, selection_written=sel_time,
                         ok=c4_before)}
    C.write_atomic(C.RES / "C_predictions.parquet", lambda p: Cp.to_parquet(p, index=False))
    C.write_atomic(C.RES / "C_reactions.parquet", lambda p: rx.to_parquet(p, index=False))
    C.write_atomic(C.RES / "C_espb.parquet", lambda p: espb.to_parquet(p, index=False))
    bad = [g for g, v in gates.items() if not v["ok"]]
    outs = ("C_selection.csv", "C_predictions.parquet", "C_reactions.parquet", "C_espb.parquet")
    rep = dict(spec="docs/specs/ESPB.md 7", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1),
               chosen={t: chosen.loc[t, "candidate"] for t in C.CHANNELS},
               selection=S.to_dict("records"), gates=gates,
               outputs={str(C.RES / n): C.sha256(C.RES / n) for n in outs}, all_pass=not bad, failed=bad)
    C.write_json(C.RES / "C_report.json", rep)
    print(json.dumps(dict(all_pass=not bad, failed=bad, chosen=rep["chosen"], **{g: v["ok"] for g, v in gates.items()}),
                     indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
