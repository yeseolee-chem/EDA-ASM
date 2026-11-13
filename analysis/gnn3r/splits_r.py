#!/usr/bin/env python3
"""GNN3R Phase S (docs/specs/GNN3R.md 5): T1 splits and evaluation attributes of the GNN3R pairs.

    python analysis/gnn3r/splits_r.py            PYENV=espley (sklearn 1.3.1; analysis/d1_pair/jobs/py.slurm), CPU

core -> fold = the published T1 table (S_folds.csv core, fold_T1); -1 = always training (D-9). Fold k: test = pairs
of the cores with fold k, train = every other pair (both in ascending pair_id); inner split = the first split of
GroupKFold(4) over the training pairs grouped by core (= d1_train.inner_folds). Writes results/S_folds_r.csv,
results/splits/T1__f<k>.json, results/S_splits_r.json. Needs results/B_label_report.json all_pass for F_pairs_r.csv.
An existing passed S_splits_r.json with unchanged files is not rebuilt. Exit 0 pass, 3 = STOP.
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
import gr_common as C  # noqa: E402

ATTR = ["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "core", "dipole", "skeleton", "series", "cfg", "fam",
        "pos", "Xb", "sigma"]
N_FOLDS = 5


def unit_name(k: int) -> str:
    return f"T1__f{k}.json"


def current() -> bool:
    if not C.S_SPLITS_R.is_file():
        return False
    rep = C.read_json(C.S_SPLITS_R)
    files = [C.S_FOLDS_R] + [C.SPLIT_DIR / unit_name(k) for k in range(N_FOLDS)]
    return bool(rep.get("all_pass")) and all(p.is_file() for p in files) \
        and rep["pairs"]["sha256"] == C.sha256(C.PAIRS_R) and rep["s_folds_r"]["sha256"] == C.sha256(C.S_FOLDS_R) \
        and all(rep["units"][unit_name(k)]["sha256"] == C.sha256(C.SPLIT_DIR / unit_name(k)) for k in range(N_FOLDS))


def main() -> int:
    if current():
        print(f"Phase S done ({C.S_SPLITS_R}); not rebuilt")
        return 0
    import sklearn
    from sklearn.model_selection import GroupKFold
    lr = C.read_json(C.LABEL_REPORT)
    if not lr.get("all_pass") or lr["outputs"].get(str(C.PAIRS_R)) != C.sha256(C.PAIRS_R):
        print(f"STOP: {C.LABEL_REPORT} not all_pass or F_pairs_r.csv changed")
        return 3
    P = pd.read_csv(C.PAIRS_R, dtype={"pair_id": str})
    sf = pd.read_csv(C.S_FOLDS, usecols=["core", "fold_T1"])
    per_core = sf.groupby("core").fold_T1.nunique()
    table = sf.drop_duplicates().set_index("core").fold_T1
    gates = {"S-0": dict(n_cores=int(len(per_core)), want=C.N_FOLD_CORES, one_fold_per_core=bool((per_core == 1).all()),
                         ok=len(per_core) == C.N_FOLD_CORES and bool((per_core == 1).all()))}
    new = sorted(set(P.core) - set(table.index))
    gates["S-1"] = dict(n_pair_cores=int(P.core.nunique()), new_cores=new, ok=not new)
    if new or not gates["S-0"]["ok"]:
        C.write_json(C.S_SPLITS_R, dict(all_pass=False, failed=["S-0" if not gates["S-0"]["ok"] else "S-1"], gates=gates))
        print(json.dumps(gates, indent=1, default=str))
        return 3
    S = P[ATTR].copy()
    S["in_P_main"] = S.ref_kind == "reference"
    S["in_P_sens"] = S.ref_kind == "reference"
    S["in_T_main"] = True
    S["fold_T1"] = S.core.map(table).astype(int)
    S = S.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    C.write_atomic(C.S_FOLDS_R, S.to_csv(index=False))
    ids = S.pair_id.to_numpy()
    core = dict(zip(S.pair_id, S.core))
    fold = dict(zip(S.pair_id, S.fold_T1))
    ends = {q: (a, b) for q, a, b in zip(S.pair_id, S.a_job, S.b_job)}
    units, s2, s3, tests, trains = {}, {}, {}, {}, {}
    for k in range(N_FOLDS):
        test = sorted(q for q in ids if fold[q] == k)
        train = sorted(q for q in ids if fold[q] != k)
        tr = np.array(train)
        a, b = next(GroupKFold(C.N_INNER).split(tr, groups=np.array([core[q] for q in train])))
        inner_train, inner_val = tr[a].tolist(), tr[b].tolist()
        rec = {"split": "T1", "fold": k, "train": train, "test": test, "inner_train": inner_train, "inner_val": inner_val}
        path = C.SPLIT_DIR / unit_name(k)
        C.write_atomic(path, json.dumps(rec) + "\n")
        units[unit_name(k)] = dict(n_train=len(train), n_test=len(test), n_inner_train=len(inner_train),
                                   n_inner_val=len(inner_val), sha256=C.sha256(path))
        rx = lambda qs: {j for q in qs for j in ends[q]}                      # noqa: E731
        it, iv = set(inner_train), set(inner_val)
        s2[k] = dict(pairs=not (set(test) & set(train)), cores=not ({core[q] for q in test} & {core[q] for q in train}),
                     reactions=not (rx(test) & rx(train)))
        s3[k] = dict(union=it | iv == set(train), disjoint=not (it & iv),
                     val_cores_not_in_inner_train=not ({core[q] for q in iv} & {core[q] for q in it}),
                     train_order_kept=inner_train == [q for q in train if q in it]
                     and inner_val == [q for q in train if q in iv])
        tests[k], trains[k] = set(test), set(train)
    gates["S-2"] = dict(per_fold=s2, ok=all(all(v.values()) for v in s2.values()))
    gates["S-3"] = dict(per_fold=s3, ok=all(all(v.values()) for v in s3.values()))
    n_in_tests = {q: sum(q in tests[k] for k in range(N_FOLDS)) for q in ids}
    always = [q for q in ids if fold[q] == -1]
    s4 = dict(non_minus1_in_exactly_one_test=all(n_in_tests[q] == 1 for q in ids if fold[q] != -1),
              minus1_in_no_test=all(n_in_tests[q] == 0 for q in always),
              train_union_is_all=set().union(*trains.values()) == set(ids),
              in_T_main_is_all=bool(S.in_T_main.all()), folds_present=sorted(set(int(v) for v in fold.values())))
    gates["S-4"] = dict(s4, ok=all(v for v in s4.values() if isinstance(v, bool)))
    bad = [g for g, v in gates.items() if not v["ok"]]
    test_sizes = {str(k): len(tests[k]) for k in range(N_FOLDS)}
    test_sizes["-1"] = len(always)
    rep = dict(spec="docs/specs/GNN3R.md 5", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"),
               pairs=dict(path=str(C.PAIRS_R), sha256=C.sha256(C.PAIRS_R)), n_pairs=int(len(S)),
               s_folds_r=dict(path=str(C.S_FOLDS_R), sha256=C.sha256(C.S_FOLDS_R)),
               fold_table=dict(path=str(C.S_FOLDS), sha256=C.sha256(C.S_FOLDS), column="fold_T1"),
               sklearn=sklearn.__version__, rule="test = core fold k; train = the rest (fold -1 always train); inner = "
                                                  "first GroupKFold(4) split of train grouped by core",
               test_sizes=test_sizes, expected_test_sizes={str(k): v for k, v in C.EXPECT_TEST.items()},
               units=units, gates=gates, all_pass=not bad, failed=bad)
    C.write_json(C.S_SPLITS_R, rep)
    print(json.dumps(dict(all_pass=not bad, failed=bad, n_pairs=len(S), test_sizes=test_sizes,
                          units={k: {kk: vv for kk, vv in v.items() if kk != "sha256"} for k, v in units.items()}),
                     indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
