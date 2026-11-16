#!/usr/bin/env python3
"""ESPB Phase P (docs/specs/ESPB.md 3): data tables for E-A and E-B.

    python prep.py            PYENV=espley (jobs/py.slurm, submitted from analysis/espb), CPU

Writes $EB/P/X46.parquet (rxn_id + train_ml_single.ESPLEY46 columns from X182_lf.parquet), results/P_reactions.csv
(rxn_id, job_id, core, fold_T1, the 10 per-reaction cr2 labels under their d1pair_common.D1_TCOL names, the 4 channels =
gnn3.channels.labels_to_channels), results/P_inner_folds.csv (outer fold k, core, inner fold j: GroupKFold(4) over the
training pairs of fold k grouped by core = d1_train.inner_folds) and results/P_report.json (gates P-1 .. P-9).
An existing all-pass report with unchanged outputs is not rebuilt. Exit 0 pass, 3 = STOP.
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
import train_ml_single as T  # noqa: E402  (ESPLEY46, _make_pipe; read only)
from gnn3 import constants as GK  # noqa: E402
from gnn3 import metrics as M  # noqa: E402
from gnn3.channels import labels_to_channels  # noqa: E402

LABELS = list(GK.LABEL_CHANNELS)                         # d1, d2, elst, pauli, oi, disp, cpcm, cds, c_ghost, barrier
TOL = 1e-9
P9_TOL = 1e-10
N_P9 = 500
OUTPUTS = (C.X46, C.P_REACTIONS, C.P_INNER)


def current() -> bool:
    if not C.P_REPORT.is_file():
        return False
    rep = C.read_json(C.P_REPORT)
    return bool(rep.get("all_pass")) and all(Path(p).is_file() and C.sha256(p) == s
                                             for p, s in rep.get("outputs", {}).items())


def main() -> int:
    if current():
        print(f"Phase P done ({C.P_REPORT}); not rebuilt")
        return 0
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.svm import SVR
    t0 = time.time()
    assert list(GK.LABEL_CHANNELS) == list(C.D1_TCOL), "label order"
    P = pd.read_csv(C.PAIRS)
    S = pd.read_csv(C.S_FOLDS)
    pairs = P.merge(S[["pair_id", "fold_T1", "in_P_main"]], on="pair_id", how="left", validate="1:1")
    gates = {}
    # P-1 pairs and test sizes per fold
    sizes = {int(k): int(v) for k, v in pairs.groupby("fold_T1").size().items()}
    n_test = int((pairs.fold_T1 >= 0).sum())
    gates["P-1"] = dict(n_pairs=int(len(pairs)), sizes=sizes, want=C.TEST_SIZES, n_test=n_test,
                        ok=len(pairs) == C.N_PAIRS and sizes == C.TEST_SIZES and n_test == C.N_TEST
                        and not pairs.fold_T1.isna().any())
    # P-2 reactions, one core each, a and b in the pair's core
    ends = pd.concat([pairs[["a_rxn", "a_job", "core"]].set_axis(["rxn_id", "job_id", "core"], axis=1),
                      pairs[["b_rxn", "b_job", "core"]].set_axis(["rxn_id", "job_id", "core"], axis=1)])
    per = ends.groupby("rxn_id").agg(n_core=("core", "nunique"), n_job=("job_id", "nunique"), core=("core", "first"),
                                     job_id=("job_id", "first"))
    fold_of_core = pairs.groupby("core").fold_T1.agg(["nunique", "first"])
    gates["P-2"] = dict(n_reactions=int(len(per)), want=C.N_REACTIONS, max_cores_per_reaction=int(per.n_core.max()),
                        max_jobs_per_reaction=int(per.n_job.max()),
                        pair_core_is_a_core=True, max_folds_per_core=int(fold_of_core["nunique"].max()),
                        ok=len(per) == C.N_REACTIONS and per.n_core.max() == 1 and per.n_job.max() == 1
                        and fold_of_core["nunique"].max() == 1)
    rx = per.index.to_numpy()
    # P-3 ESPLEY46 columns, finite
    X = pd.read_parquet(C.X182)
    cols = list(T.ESPLEY46)
    missing_cols = [c for c in cols if c not in X.columns]
    X = X.set_index("rxn_id")
    have = [r for r in rx if r in X.index]
    X46 = X.loc[have, cols] if not missing_cols else pd.DataFrame()
    finite = bool(len(have) == len(rx) and not missing_cols and np.isfinite(X46.to_numpy(float)).all())
    gates["P-3"] = dict(n_columns=len(cols), missing_columns=missing_cols, n_reactions_in_x182=len(have), finite=finite,
                        ok=not missing_cols and len(have) == len(rx) and finite)
    if not gates["P-3"]["ok"] or not gates["P-2"]["ok"] or not gates["P-1"]["ok"]:
        C.write_json(C.P_REPORT, dict(all_pass=False, failed=[g for g, v in gates.items() if not v["ok"]], gates=gates))
        print(json.dumps(gates, indent=1, default=str))
        return 3
    # per-reaction labels (cr2) and channels
    cr2 = {d["job_id"]: d for d in json.loads(C.LABELS_CR2.read_text())}
    R = per[["job_id", "core"]].copy()
    R["fold_T1"] = R.core.map(fold_of_core["first"]).astype(int)
    for t in LABELS:
        R[C.D1_TCOL[t]] = [float(cr2[j][C.D1_TCOL[t]]) for j in R.job_id]
    ch = labels_to_channels(R[[C.D1_TCOL[t] for t in LABELS]].to_numpy(float))
    for i, c in enumerate(C.CHANNELS):
        R[c] = ch[:, i]
    R = R.reset_index().sort_values("rxn_id", kind="mergesort").reset_index(drop=True)
    Ri = R.set_index("rxn_id")
    # P-4 pair differences reproduce F_pairs_r.csv dd and gnn3.metrics.channel_truth
    a, b = pairs.a_rxn.to_numpy(), pairs.b_rxn.to_numpy()
    dd_max = {t: float(np.abs((Ri.loc[a, C.D1_TCOL[t]].to_numpy() - Ri.loc[b, C.D1_TCOL[t]].to_numpy())
                              - pairs[f"dd_{t}"].to_numpy(float)).max()) for t in LABELS}
    truth, _ = M.channel_truth(C.PAIRS)
    truth = truth.loc[pairs.pair_id.astype(str)]
    ch_max = {c: float(np.abs((Ri.loc[a, c].to_numpy() - Ri.loc[b, c].to_numpy()) - truth[c].to_numpy(float)).max())
              for c in C.CHANNELS}
    gates["P-4"] = dict(dd_max_abs=dd_max, channel_max_abs=ch_max, tol=TOL,
                        ok=max(dd_max.values()) <= TOL and max(ch_max.values()) <= TOL)
    # P-5 outer folds = the GNN3R2 split files; training and test reactions disjoint
    p5, inner_rows, p6 = {}, [], {}
    core_of = dict(zip(pairs.pair_id, pairs.core))
    ends_of = {q: (x, y) for q, x, y in zip(pairs.pair_id, pairs.a_rxn, pairs.b_rxn)}
    for k in range(C.N_FOLDS):
        unit = C.split_unit(k)
        test = sorted(pairs.loc[pairs.fold_T1 == k, "pair_id"])
        train = sorted(pairs.loc[pairs.fold_T1 != k, "pair_id"])
        rtr = {r for q in unit["train"] for r in ends_of[q]}
        rte = {r for q in unit["test"] for r in ends_of[q]}
        p5[k] = dict(train_equal=unit["train"] == train, test_equal=unit["test"] == test, n_train=len(train),
                     n_test=len(test), n_train_rxn=len(rtr), n_test_rxn=len(rte), rxn_overlap=len(rtr & rte))
        # P-6 GroupKFold(4) by core over the training pairs (split-file order) = d1_train.inner_folds
        tr = np.asarray(unit["train"])
        groups = np.array([core_of[q] for q in tr])
        splits = list(GroupKFold(C.N_INNER).split(tr, groups=groups))
        first_val = tr[splits[0][1]].tolist()
        cmap = {}
        for j, (_, vb) in enumerate(splits):
            for c in sorted(set(groups[vb])):
                cmap.setdefault(c, set()).add(j)
        p6[k] = dict(first_val_equals_inner_val=first_val == unit["inner_val"], n_inner_val=len(first_val),
                     n_cores=len(cmap), cores_in_one_inner_fold=all(len(v) == 1 for v in cmap.values()),
                     val_pairs_per_fold=[int(len(vb)) for _, vb in splits])
        inner_rows += [dict(outer_fold=k, core=c, inner_fold=int(next(iter(v)))) for c, v in sorted(cmap.items())]
    gates["P-5"] = dict(per_fold=p5, ok=all(v["train_equal"] and v["test_equal"] and v["rxn_overlap"] == 0
                                            for v in p5.values()))
    gates["P-6"] = dict(per_fold=p6, ok=all(v["first_val_equals_inner_val"] and v["cores_in_one_inner_fold"]
                                            for v in p6.values()))
    # P-7 label closure, P-8 xTB energy convention (reported)
    nine = [C.D1_TCOL[t] for t in LABELS if t != "barrier"]
    closure = np.abs(R.barrier_kcal - R[nine].sum(axis=1))
    gates["P-7"] = dict(max_abs=float(closure.max()), mean_abs=float(closure.mean()), ok=True, note="reported only")
    xe = X46
    conv = np.abs(xe.xtb_e_barrier_kcal - (xe.xtb_sum_distortion_kcal + xe.xtb_interaction_kcal))
    dsum = np.abs(xe.xtb_sum_distortion_kcal - (xe.xtb_dist_dipole_kcal + xe.xtb_dist_dipolarophile_kcal))
    gates["P-8"] = dict(max_abs_barrier_minus_dist_plus_int=float(conv.max()),
                        max_abs_sum_distortion_minus_parts=float(dsum.max()),
                        convention="xtb_e_barrier = xtb_sum_distortion + xtb_interaction; "
                                   "xtb_sum_distortion = xtb_dist_dipole + xtb_dist_dipolarophile",
                        ok=True, note="reported only (sign convention of the pre-ML table)")
    # P-9 no-effect hyper-parameters (fold 0 training reactions, 500 at random, target barrier)
    u0 = C.split_unit(0)
    tr0 = sorted({int(r) for q in u0["train"] for r in ends_of[q]})
    te0 = sorted({int(r) for q in u0["test"] for r in ends_of[q]})
    rng = np.random.default_rng(C.SEED)
    sample = np.sort(rng.choice(np.asarray(tr0), size=N_P9, replace=False))
    Xs, ys, Xt = X46.loc[sample].to_numpy(float), Ri.loc[sample, "barrier"].to_numpy(float), X46.loc[te0].to_numpy(float)
    rid = [T._make_pipe(Ridge(alpha=1.0, tol=tol)).fit(Xs, ys).predict(Xt) for tol in (0.1, 1.0)]
    svr = [T._make_pipe(SVR(kernel="rbf", coef0=c0, degree=dg)).fit(Xs, ys).predict(Xt) for c0 in (0, 1) for dg in (1, 3)]
    ridge_max = float(np.abs(rid[0] - rid[1]).max())
    svr_max = float(max(np.abs(p - svr[0]).max() for p in svr[1:]))
    gates["P-9"] = dict(n_sample=N_P9, sample_seed=C.SEED, n_predict=len(te0), ridge_tol_max_abs=ridge_max,
                        svr_coef0_degree_max_abs=svr_max, tol=P9_TOL, ok=ridge_max <= P9_TOL and svr_max <= P9_TOL)
    # outputs
    X46_out = X46.reset_index().rename(columns={"index": "rxn_id"})
    X46_out = X46_out[["rxn_id"] + cols].sort_values("rxn_id", kind="mergesort").reset_index(drop=True)
    C.write_atomic(C.X46, lambda t: X46_out.to_parquet(t, index=False))
    out_cols = ["rxn_id", "job_id", "core", "fold_T1"] + [C.D1_TCOL[t] for t in LABELS] + C.CHANNELS
    C.write_atomic(C.P_REACTIONS, R[out_cols].to_csv(index=False))
    C.write_atomic(C.P_INNER, pd.DataFrame(inner_rows, columns=["outer_fold", "core", "inner_fold"]).to_csv(index=False))
    from importlib.metadata import version
    bad = [g for g, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/ESPB.md 3", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1),
               versions={k: version(k) for k in ("scikit-learn", "numpy", "pandas")},
               inputs={str(p): C.sha256(p) for p in (C.PAIRS, C.S_FOLDS, C.X182, C.LABELS_CR2)
                       + tuple(C.SPLIT_DIR / f"T1__f{k}.json" for k in range(C.N_FOLDS))},
               espley46=cols, counts=dict(pairs=int(len(pairs)), reactions=int(len(R)), inner_rows=len(inner_rows)),
               gates=gates, outputs={str(p): C.sha256(p) for p in OUTPUTS}, all_pass=not bad, failed=bad)
    C.write_json(C.P_REPORT, rep)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{g: v["ok"] for g, v in gates.items()},
                          P7=gates["P-7"]["max_abs"], P8=[gates["P-8"]["max_abs_barrier_minus_dist_plus_int"],
                                                         gates["P-8"]["max_abs_sum_distortion_minus_parts"]],
                          P9=[ridge_max, svr_max]), indent=1, default=str))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
