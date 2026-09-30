#!/usr/bin/env python3
"""verify_geom_dft.py — REV4 Phase 2-1 gate: `xtb_slice.py --geom dft` must reproduce the rev 3 features.

usage: python verify_geom_dft.py <new_parquet> <old_parquet> <out_json> [--rows-from-new]

Compared on the common rxn_ids: every feature column (union of train_ml_single.FEATURE_SETS) plus every other
numeric non-target column present in both (NaN on both sides = equal, NaN on one side = inf). The dft_* target
columns are reported separately and are not part of PASS.
PASS iff every compared max |diff| <= 1e-8, no FEATURE_SETS column is missing, xtb_status agrees on every
compared row and no rxn is missing on either side. --rows-from-new: only the rxns of <new_parquet> (e.g. one
re-run slice) must be present in <old_parquet>; old-only rxns are ignored.
A raw slice lacks the aggregate-derived columns; they are derived with aggregate.derive.
Exit code 0 = PASS, 3 = FAIL. The JSON is rewritten on every run (cheap and deterministic).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import derive  # noqa: E402
from train_ml_single import FEATURE_SETS  # noqa: E402

TOL = 1e-8


def absdiff(x, y):
    x = pd.to_numeric(x, errors="coerce").astype("float64").to_numpy()
    y = pd.to_numeric(y, errors="coerce").astype("float64").to_numpy()
    d = np.abs(x - y)
    d[(x == y) | (np.isnan(x) & np.isnan(y))] = 0.0
    d[np.isnan(d)] = np.inf
    return d


def compare(a, b, cols):
    out = {}
    for c in cols:
        d = absdiff(a[c], b[c])
        k = int(np.argmax(d)) if len(d) else 0
        out[c] = {"max_abs_diff": float(d.max()) if len(d) else 0.0, "n_over_tol": int((d > TOL).sum()),
                  "n_over_1e-6": int((d > 1e-6).sum()), "worst_rxn_id": int(a.index[k]) if len(d) else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_parquet", type=Path)
    ap.add_argument("old_parquet", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--rows-from-new", action="store_true")
    a = ap.parse_args()
    new, old = pd.read_parquet(a.new_parquet), pd.read_parquet(a.old_parquet)
    for df in (new, old):
        if "is_charged" not in df or "dft_c_ghost_kcal" not in df:
            derive(df)
    dup = {k: int(df.rxn_id.duplicated().sum()) for k, df in (("new", new), ("old", old))}
    new, old = new.drop_duplicates("rxn_id").set_index("rxn_id"), old.drop_duplicates("rxn_id").set_index("rxn_id")
    ids_new, ids_old = {int(i) for i in new.index}, {int(i) for i in old.index}
    miss_old, miss_new = sorted(ids_new - ids_old), sorted(ids_old - ids_new)
    rows = sorted(ids_new & ids_old)
    A, B = new.loc[rows], old.loc[rows]

    feats = sorted(set(sum(FEATURE_SETS.values(), [])))
    feat_missing = dict(new=[c for c in feats if c not in A], old=[c for c in feats if c not in B])
    num = lambda df, c: pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])  # noqa: E731
    common = [c for c in A.columns if c in B.columns]
    targets = [c for c in common if c.startswith("dft_")]
    others = [c for c in common if c not in feats and c not in targets and num(A, c) and num(B, c)]
    fcols = [c for c in feats if c in A and c in B]

    st_eq = (A["xtb_status"].astype(str) == B["xtb_status"].astype(str)).to_numpy()
    disagree = [dict(rxn_id=int(r), new=str(A.at[r, "xtb_status"]), old=str(B.at[r, "xtb_status"]))
                for r, e in zip(rows, st_eq) if not e]
    cmp_f, cmp_o, cmp_t = compare(A, B, fcols), compare(A, B, others), compare(A, B, targets)
    gated = {**cmp_f, **cmp_o}
    bad = sorted(c for c, v in gated.items() if not v["max_abs_diff"] <= TOL)
    missing_fail = bool(miss_old) or (bool(miss_new) and not a.rows_from_new)
    passed = (not bad and not disagree and not feat_missing["new"] and not feat_missing["old"]
              and not missing_fail and not any(dup.values()) and len(rows) > 0)

    rep = dict(verdict="PASS" if passed else "FAIL", tol=TOL, new=str(a.new_parquet), old=str(a.old_parquet),
               rows_from_new=a.rows_from_new, n_new=len(ids_new), n_old=len(ids_old), n_compared=len(rows),
               n_missing_in_old=len(miss_old), missing_in_old=miss_old[:50],
               n_missing_in_new=len(miss_new), missing_in_new=miss_new[:50] if not a.rows_from_new else "ignored",
               duplicate_rxn_ids=dup,
               n_ok=dict(new=int((A.xtb_status == "ok").sum()), old=int((B.xtb_status == "ok").sum())),
               xtb_status_agree=int(st_eq.sum()), xtb_status_disagree=len(disagree), status_disagreements=disagree[:50],
               feature_columns_missing=feat_missing, failing_columns=bad,
               n_feature_columns=len(fcols), n_other_numeric_columns=len(others),
               columns_only_in_new=sorted(set(new.columns) - set(old.columns)),
               columns_only_in_old=sorted(set(old.columns) - set(new.columns)),
               max_abs_diff_features=max((v["max_abs_diff"] for v in cmp_f.values()), default=0.0),
               max_abs_diff_other_numeric=max((v["max_abs_diff"] for v in cmp_o.values()), default=0.0),
               max_abs_diff_targets_not_gated=max((v["max_abs_diff"] for v in cmp_t.values()), default=0.0),
               features=cmp_f, other_numeric=cmp_o, targets_not_gated=cmp_t)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out_json.with_name(a.out_json.name + ".tmp")
    tmp.write_text(json.dumps(rep, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    os.replace(tmp, a.out_json)
    print(f"{rep['verdict']}: {len(rows)} rxns compared ({len(fcols)} feature + {len(others)} other numeric columns); "
          f"max|diff| features {rep['max_abs_diff_features']:.3g}, other {rep['max_abs_diff_other_numeric']:.3g}, "
          f"targets (not gated) {rep['max_abs_diff_targets_not_gated']:.3g}; xtb_status disagree {len(disagree)}; "
          f"missing in old {len(miss_old)}, in new {len(miss_new)}{' (ignored)' if a.rows_from_new else ''}")
    for c in bad:
        print(f"  FAIL column {c}: {gated[c]}")
    if feat_missing["new"] or feat_missing["old"]:
        print(f"  FAIL missing feature columns: {feat_missing}")
    print(f"-> {a.out_json}")
    sys.exit(0 if passed else 3)


if __name__ == "__main__":
    main()
