#!/usr/bin/env python3
"""abs_pack.py — ABS3 XGB input pack (docs/specs/ABS3_MODELS.md 3, 4), built in env_d1fu2 (AbsData needs torch) for
xgb_abs.py / eval_abs.py in env_espley (no torch).

  python abs_pack.py     -> $ABS3/pack/lf_abs.npz + lf_abs.json (exit 3 on a failed check)

Pack content
  reactions (token-row order = DeployData rids): rids, X (the 182 columns of $CH3/lf/X182_lf.parquet reindexed to the
    token rows, columns = T_report glob_cols: the LADDER XData rule, not the token glob), y10 / y4 (labels_d1_cr2.json,
    NaN = unlabelled token reaction), in_set (reaction of a usable pair of some fold list), dipole, skeleton, series,
    attach, core, role ('H' / 'sub' / '' outside the set);
  fold reaction sets f<k>_<train|test|inner_train|inner_val> (token rows, = AbsData.split(k));
  pairs (pair-table order = F_pairs_cr2 rows): pair_id, ref_kind, pos, a_rxn, b_rxn, ra, rb, has_tok, usable_pair,
    core (S_folds.csv), dd4 = dd_<10> @ M;
  split-file pair lists p<k>_<train|test|inner_train|inner_val> (original ids, unfiltered: the XGB inner GroupKFold(4) of
    xgb_abs.py starts from the unfiltered outer-train list, = LADDER XData.split / CH3M xgb_m.PackData).
Checks (STOP): X rows equal to the token glob (<= 1e-12 relative where finite), every set reaction has finite X / y4,
dd4 = DeployData y @ M, the pack arrays equal abs_data.pack_arrays (array sha256, = LS-5)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402
import abs_data as AD  # noqa: E402


def build():
    A = AD.get_data()
    D = A.D
    X, xpath = A.x182()
    fin = np.isfinite(X).all(1)
    dev = np.abs(X - A.glob) > 1e-12 * np.maximum(1.0, np.abs(A.glob))
    n_dev = int((dev & np.isfinite(X)).any(1).sum())
    rs = A.rxn_set
    arrays = dict(rids=np.asarray(A.rids, dtype=np.int64), X=X, y10=A.y10, y4=A.y4, in_set=A.in_set,
                  dipole=A.attr["dipole"].astype(str), skeleton=A.attr["skeleton"].astype(str),
                  series=A.attr["series"].astype(str), attach=A.attr["attach"].astype(str),
                  core=A.core_rxn.astype(str), role=A.role.astype(str),
                  pair_id=np.array([str(q) for q in D.P.index]), ref_kind=D.P.ref_kind.to_numpy().astype(str),
                  pos=D.P.pos.to_numpy().astype(str), a_rxn=D.P.a_rxn.to_numpy(np.int64), b_rxn=D.P.b_rxn.to_numpy(np.int64),
                  ra=D.ra.astype(np.int64), rb=D.rb.astype(np.int64), has_tok=D.has_tok, usable_pair=D.usable_pair,
                  pair_core=A.pair_core.astype(str), dd4=D.y @ C.M())
    for k in range(C.NF):
        for key in C.FOLD_KEYS:
            arrays[f"f{k}_{key}"] = A.sets[k][key].astype(np.int64)
            arrays[f"p{k}_{key}"] = np.array([str(q) for q in D.folds[k][key]])
    want = AD.pack_arrays(A)
    same = {k: C.sha256_array(arrays[k]) == C.sha256_array(v) for k, v in want.items()}
    checks = dict(X_rows_vs_token_glob_n_dev=n_dev, X_equals_token_glob=n_dev == 0,
                  set_X_finite=bool(fin[rs].all()), set_y4_finite=bool(np.isfinite(A.y4[rs]).all()),
                  dd4_eq_DeployData_y_at_M=bool(np.array_equal(arrays["dd4"], D.y @ C.M(), equal_nan=True)),
                  arrays_equal_abs_data=all(same.values()), n_reaction_set=int(len(rs)), n_pairs_P=len(D.pair_list),
                  pair_list_sha=D.pair_list_sha)
    ok = bool(checks["X_equals_token_glob"] and checks["set_X_finite"] and checks["set_y4_finite"]
              and checks["dd4_eq_DeployData_y_at_M"] and checks["arrays_equal_abs_data"])
    C.write_npz(C.PACK_NPZ, **arrays)
    info = dict(spec="docs/specs/ABS3_MODELS.md 3 (abs_pack.py), 4 (LS-5)", written=C.now(), ok=ok, checks=checks,
                glob_cols=list(A.trep["glob_cols"]), x182=xpath,
                sources_sha256=dict(A.sha(), x182_file=C.sha256(xpath), labels_file=C.sha256(C.LABELS)),
                array_sha256={k: C.sha256_array(v) for k, v in want.items()}, npz=str(C.PACK_NPZ),
                npz_sha256=C.sha256(C.PACK_NPZ), sets_fingerprint=A.sets_fingerprint(),
                n_tokens=int(len(A.rids)), usable_rule=D.usable_rule)
    C.write_json(C.PACK_JSON, info)
    print(json.dumps(C.clean(dict(ok=ok, checks=checks)), indent=1))
    return 0 if ok else 3


if __name__ == "__main__":
    if sys.argv[1:]:
        print(__doc__)
        sys.exit(2)
    sys.exit(build())
