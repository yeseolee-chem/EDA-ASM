#!/usr/bin/env python3
"""ESPB Phase E (docs/specs/ESPB.md 8): evaluation and comparison with the same scoring code as GNN3R2 (gnn3.metrics).

    python analysis/espb/compare.py         env_d1fu2 (gnn3/jobs/run.slurm, submitted from the repository root), CPU

E-0: the GNN3R2 CV predictions read with gnn3/scripts/evaluate.py read_predictions and scored with M.channel_truth /
M.pair_sets / M.assemble / M.error_table must reproduce analysis/gnn3r2/results/eval/metrics.csv (numeric columns
<= 1e-12), else STOP (exit 3). Then one M.assemble over {GNN3, ESPB, EA-Ridge .. EB-NN4} gives
  E_primary.csv    P_main, all positions, 4 targets: MAE(GNN3), MAE(ESPB), d = MAE(ESPB) - MAE(GNN3) with the core
                   bootstrap 95 % interval and two-sided p (M.bootstrap_mae_difference, 2,000 resamples, seed 20261019),
                   Holm over the 4 targets
  E_secondary.csv  ESPB vs GNN3 on P_main para, test_all (all, para), P_main positions and series; each candidate vs GNN3
                   (P_main all, para); the zero predictor (MAE(0), improvement MAE(0) - MAE); the shrinkage slope (OLS
                   slope of yhat on y, P_main barrier, per series). Holm within each group of 4 targets (holm_scope).
  E_metrics.csv, E_rank.csv, E_cause.csv, E_breakdown.csv (GNN3, ESPB) = M.error_table / ranking_table /
                   cause_table(cause_agreement) / breakdown_table
  E_native.csv     E-A per-reaction absolute test MAE (Espley's metric): 5 models x d1, d2, inter, solv, Int (= inter +
                   solv predictions), barrier; mean over the 5 folds of the fold MAE and of the fold MAE / (max - min of
                   the test y) in %
  E_preml.csv      xTB before ML against DFT: per reaction (8,594) and per P_main test pair
Writes results/E_report.json. Exit 0 pass, 3 = STOP (E-0).
"""
from __future__ import annotations

import importlib.util
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

E0_TOL = 1e-12
KEYS = ["model", "eval_set", "target", "pos"]
POSITIONS = ["p", "m", "o", "direct"]
SERIES = ["R1", "R2"]


def gnn3_evaluate():
    spec = importlib.util.spec_from_file_location("gnn3_evaluate", C.GNN3_PACKAGE / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def holm(p):
    """Holm-adjusted p values (step-down, monotone, capped at 1)"""
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def check_e0(E, truth, attrs, sets):
    pred = E.read_predictions(C.GNN3_CV, False)
    long = M.assemble({"GNN3": pred}, truth, attrs, sets)
    ids = sorted(long.pair_id.unique())
    mine = M.error_table(long, ids, ["GNN3"])
    ref = pd.read_csv(C.GNN3_METRICS)
    mm = mine.merge(ref, on=KEYS, how="outer", suffixes=("", "_ref"), indicator=True)
    num = [c for c in ref.columns if c not in KEYS and pd.api.types.is_numeric_dtype(ref[c])]
    diffs = {}
    for c in num:
        a, b = mm[c].to_numpy(float), mm[f"{c}_ref"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        diffs[c] = float(np.nanmax(np.where(np.isnan(d), np.inf, d))) if len(d) else 0.0
    ok = bool((mm._merge == "both").all() and len(mine) == len(ref) and max(diffs.values()) <= E0_TOL)
    return dict(n_rows=int(len(mine)), n_ref=int(len(ref)), unmatched=int((mm._merge != "both").sum()), max_abs=diffs,
                tol=E0_TOL, ok=ok), pred


def errors(long, model, set_name, target, sub=None):
    """(pair_id-sorted frame of one model / set / target) for paired comparisons"""
    f = long[(long.model == model) & (long.target == target)]
    f = f[M.set_mask(f, set_name)]
    if sub is not None:
        col, val = sub
        f = f[f[col] == val]
    return f.sort_values("pair_id", kind="mergesort")


def paired(long, mx, mb, set_name, target, sub=None):
    fx, fb = errors(long, mx, set_name, target, sub), errors(long, mb, set_name, target, sub)
    if list(fx.pair_id) != list(fb.pair_id):
        raise SystemExit(f"STOP: {mx} / {mb} {set_name} {target} {sub}: pair lists differ")
    ex, eb = (fx.yhat - fx.y).to_numpy(float), (fb.yhat - fb.y).to_numpy(float)
    d, lo, hi, p = M.bootstrap_mae_difference(ex, eb, fx.core.to_numpy(), C.BOOT_N, C.BOOT_SEED)
    return dict(n=int(len(fx)), n_cores=int(fx.core.nunique()), mae_model=float(np.abs(ex).mean()),
                mae_versus=float(np.abs(eb).mean()), d=d, ci_lo=lo, ci_hi=hi, p=p)


def add_holm(rows, scope):
    ps = [r["p"] for r in rows]
    for r, a in zip(rows, holm(ps)):
        r.update(p_holm=float(a), holm_scope=scope)
    return rows


def native(R):
    """E-A per-reaction test MAE (Espley's metric) and MAE as % of the test range, mean over folds"""
    rx = pd.read_parquet(C.RES / "C_reactions.parquet")
    rows = []
    for m in C.MODELS:
        d = rx[rx.model == m]
        wide = d.pivot_table(index=["rxn_id", "fold"], columns="target", values="yhat").reset_index()
        truth = R.loc[wide.rxn_id]
        cols = {"d1": ("d1", truth.d1_kcal.to_numpy()), "d2": ("d2", truth.d2_kcal.to_numpy()),
                "inter": ("inter", truth.inter.to_numpy()), "solv": ("solv", truth.solv.to_numpy()),
                "barrier": ("barrier", truth.barrier.to_numpy())}
        for name, (col, y) in list(cols.items()) + [("Int", (None, (truth.inter + truth.solv).to_numpy()))]:
            yhat = (wide["inter"] + wide["solv"]).to_numpy(float) if name == "Int" else wide[col].to_numpy(float)
            mae_f, pct_f = [], []
            for k in range(C.N_FOLDS):
                s = (wide.fold == k).to_numpy()
                e = np.abs(yhat[s] - y[s])
                mae_f.append(float(e.mean()))
                pct_f.append(float(100.0 * e.mean() / (y[s].max() - y[s].min())))
            rows.append(dict(model=f"EA-{m}", target=name, n=int(len(y)), mae=float(np.mean(mae_f)),
                             mae_pct_range=float(np.mean(pct_f)), fold_mae=json.dumps(mae_f), fold_pct=json.dumps(pct_f)))
    return pd.DataFrame(rows)


def preml(R, long):
    X = pd.read_parquet(C.X46).set_index("rxn_id")
    rows = []
    rx = R.index.to_numpy()
    for name, xcol, y in (("d1", "xtb_dist_dipole_kcal", R.d1_kcal), ("d2", "xtb_dist_dipolarophile_kcal", R.d2_kcal),
                          ("Int", "xtb_interaction_kcal", R.inter + R.solv), ("barrier", "xtb_e_barrier_kcal", R.barrier)):
        x = X.loc[rx, xcol].to_numpy(float)
        e = x - y.to_numpy(float)
        rows.append(dict(level="reaction", target=name, xtb=xcol, n=int(len(e)), mae=float(np.abs(e).mean()),
                         mean_signed=float(e.mean()), r=M.pearson(y.to_numpy(float), x)))
    g = long[(long.model == "GNN3") & (long.target == "barrier")]
    g = g[M.set_mask(g, "P_main")]
    P = pd.read_csv(C.PAIRS, usecols=["pair_id", "a_rxn", "b_rxn"]).set_index("pair_id").loc[g.pair_id]
    Rp = R
    for name, xcol, ycols in (("strain", "xtb_sum_distortion_kcal", ["strain"]),
                              ("Int", "xtb_interaction_kcal", ["inter", "solv"]),
                              ("barrier", "xtb_e_barrier_kcal", ["barrier"])):
        dx = X.loc[P.a_rxn, xcol].to_numpy(float) - X.loc[P.b_rxn, xcol].to_numpy(float)
        dy = (Rp.loc[P.a_rxn, ycols].sum(axis=1).to_numpy(float) - Rp.loc[P.b_rxn, ycols].sum(axis=1).to_numpy(float))
        e = dx - dy
        rows.append(dict(level="pair_P_main_test", target=name, xtb=f"delta {xcol}", n=int(len(e)),
                         mae=float(np.abs(e).mean()), mean_signed=float(e.mean()), r=M.pearson(dy, dx)))
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    E = gnn3_evaluate()
    truth, attrs = M.channel_truth(C.PAIRS)
    sets = M.pair_sets(C.S_FOLDS)
    e0, gnn3 = check_e0(E, truth, attrs, sets)
    if not e0["ok"]:
        C.write_json(C.RES / "E_report.json", dict(all_pass=False, failed=["E-0"], gates={"E-0": e0}))
        print(json.dumps(e0, indent=1))
        return 3
    cp = pd.read_parquet(C.RES / "C_predictions.parquet")
    espb = pd.read_parquet(C.RES / "C_espb.parquet")
    sel = pd.read_csv(C.RES / "C_selection.csv")
    chosen = sel[sel.chosen].set_index("target").candidate.to_dict()
    cols = ["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]
    models = {"GNN3": gnn3, "ESPB": espb[cols].assign(pair_id=espb.pair_id.astype(str))}
    cands = []
    for arm in C.ARMS:
        for m in C.MODELS:
            nm = C.model_name(arm, m)
            d = cp[(cp.arm == arm) & (cp.model == m)][cols]
            models[nm] = d.assign(pair_id=d.pair_id.astype(str))
            cands.append(nm)
    long = M.assemble(models, truth, attrs, sets)
    ids = sorted(long.pair_id.unique())
    names = list(models)
    C.write_atomic(C.RES / "E_metrics.csv", M.error_table(long, ids, names).to_csv(index=False))
    C.write_atomic(C.RES / "E_rank.csv", M.ranking_table(long, ids, names).to_csv(index=False))
    C.write_atomic(C.RES / "E_cause.csv",
                   M.cause_table([M.cause_agreement(long, ids, m) for m in names]).to_csv(index=False))
    C.write_atomic(C.RES / "E_breakdown.csv", pd.concat([M.breakdown_table(long, ids, m) for m in ("GNN3", "ESPB")],
                                                        ignore_index=True).to_csv(index=False))
    # primary (registered): P_main, all positions, ESPB vs GNN3, Holm over the 4 targets
    prim = add_holm([dict(target=t, espb_candidate=chosen[t], **paired(long, "ESPB", "GNN3", "P_main", t))
                     for t in C.CHANNELS], "primary: 4 targets")
    for r in prim:
        r["mae_gnn3"], r["mae_espb"] = r.pop("mae_versus"), r.pop("mae_model")
        r["reading"] = (("ESPB lower" if r["d"] < 0 else "GNN3 lower") if r["p_holm"] < 0.05     # registered Holm test
                        else "no clear difference") + f" (Holm p {r['p_holm']:.3g})"
    P_ = pd.DataFrame(prim)[["target", "n", "n_cores", "mae_gnn3", "mae_espb", "espb_candidate", "d", "ci_lo", "ci_hi",
                             "p", "p_holm", "holm_scope", "reading"]]
    C.write_atomic(C.RES / "E_primary.csv", P_.to_csv(index=False))
    # secondary (descriptive; Holm within each group of 4 targets)
    sec = []
    groups = [("ESPB vs GNN3", "P_main", ("pos", "p"), "para"), ("ESPB vs GNN3", "test_all", None, "all"),
              ("ESPB vs GNN3", "test_all", ("pos", "p"), "para")]
    groups += [("ESPB vs GNN3", "P_main", ("pos", p), p) for p in POSITIONS]
    groups += [("ESPB vs GNN3", "P_main", ("series", s), s) for s in SERIES]
    for comp, st, sub, label in groups:
        rows = [dict(comparison=comp, eval_set=st, subset=label, target=t, model="ESPB", versus="GNN3",
                     **paired(long, "ESPB", "GNN3", st, t, sub)) for t in C.CHANNELS]
        sec += add_holm(rows, f"{comp} / {st} / {label}: 4 targets")
    for nm in cands:
        for sub, label in ((None, "all"), (("pos", "p"), "para")):
            rows = [dict(comparison="candidate vs GNN3", eval_set="P_main", subset=label, target=t, model=nm, versus="GNN3",
                         **paired(long, nm, "GNN3", "P_main", t, sub)) for t in C.CHANNELS]
            sec += add_holm(rows, f"{nm} vs GNN3 / P_main / {label}: 4 targets")
    for nm in names:                                                   # zero predictor
        for sub, label in ((None, "all"), (("pos", "p"), "para")):
            for t in C.CHANNELS:
                f = errors(long, nm, "P_main", t, sub)
                mae, mae0 = float(np.abs(f.yhat - f.y).mean()), float(np.abs(f.y).mean())
                sec.append(dict(comparison="zero predictor", eval_set="P_main", subset=label, target=t, model=nm,
                                versus="zero", n=int(len(f)), mae_model=mae, mae_versus=mae0, d=mae - mae0,
                                improvement=mae0 - mae, holm_scope="none"))
    for nm in names:                                                   # shrinkage slope
        for label, sub in (("all", None), ("R1", ("series", "R1")), ("R2", ("series", "R2"))):
            f = errors(long, nm, "P_main", "barrier", sub)
            slope, intercept = np.polyfit(f.y.to_numpy(float), f.yhat.to_numpy(float), 1)
            sec.append(dict(comparison="shrinkage slope", eval_set="P_main", subset=label, target="barrier", model=nm,
                            versus="", n=int(len(f)), slope=float(slope), intercept=float(intercept), holm_scope="none"))
    C.write_atomic(C.RES / "E_secondary.csv", pd.DataFrame(sec).to_csv(index=False))
    R = pd.read_csv(C.P_REACTIONS).set_index("rxn_id")
    C.write_atomic(C.RES / "E_native.csv", native(R).to_csv(index=False))
    C.write_atomic(C.RES / "E_preml.csv", preml(R, long).to_csv(index=False))
    outs = ("E_primary.csv", "E_secondary.csv", "E_metrics.csv", "E_rank.csv", "E_cause.csv", "E_breakdown.csv",
            "E_native.csv", "E_preml.csv")
    rep = dict(spec="docs/specs/ESPB.md 8", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1), models=names,
               n_pairs=len(ids), chosen=chosen, gates={"E-0": e0}, primary=P_.to_dict("records"),
               inputs={str(p): C.sha256(p) for p in (C.PAIRS, C.S_FOLDS, C.GNN3_METRICS, C.RES / "C_predictions.parquet",
                                                    C.RES / "C_espb.parquet", C.RES / "C_selection.csv")},
               outputs={str(C.RES / n): C.sha256(C.RES / n) for n in outs}, all_pass=True, failed=[])
    C.write_json(C.RES / "E_report.json", rep)
    print(json.dumps(dict(E0=e0["ok"], n_pairs=len(ids), chosen=chosen, primary=rep["primary"]), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
