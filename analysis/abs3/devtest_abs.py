#!/usr/bin/env python3
"""devtest_abs.py — ABS3 developer dry run helper (NOT a result; run only with ABS3_RESULTS / ABS3_SCRATCH pointing to
$ABS3/dev). numpy / pandas only.

  python devtest_abs.py fakek   the fold-0 smoke reaction predictions (TF, GNN: $SMOKE/nn, XGB: $SMOKE/xgb) as a
                                miniature KA_predictions / KA_pairs / KA_report in the dev results, so that
                                eval_abs.py run can be exercised end to end before K (fold 0 only, 1 seed / 2 epochs)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402
import abs_pairs as AP  # noqa: E402


def fake_k():
    if "dev" not in str(C.RES) or "dev" not in str(C.ABS3):
        C.stop(f"devtest_abs writes only dev paths (RES {C.RES}, ABS3 {C.ABS3})")
    parts = []
    for m, sub in (("TF", "nn"), ("GNN", "nn"), ("XGB", "xgb")):
        p = C.SMOKE / sub / f"{C.uname(m, 0)}.parquet"
        if not p.is_file():
            C.not_ready(f"{p} missing (smoke)")
        parts.append(pd.read_parquet(p))
    rx = pd.concat(parts, ignore_index=True)
    rx["model"] = rx.model.map(C.ABS_LABEL)
    z = np.load(C.PACK_NPZ, allow_pickle=False)
    row = {int(r): k for k, r in enumerate(z["rids"])}
    ri = np.array([row[int(r)] for r in rx.rxn_id])
    for c in ("dipole", "skeleton", "series", "attach"):
        rx[c] = z[c][ri].astype(str)
    KA = rx[C.RXN_COLS]
    C.write_parquet(C.RES / "KA_predictions.parquet", KA)
    pairs = AP.pack_test_pairs(z, 0)
    truth = AP.labels_truth(C.LABELS, C.M())
    pp, cp = [], {}
    for m in C.ABS_LABEL.values():
        q = rx[rx.model == m]
        a = AP.assemble(q, pairs, m, rung=C.PAIR_RUNG, arm=C.ARM)
        cp[m] = AP.cp_check(a, q, pairs, truth)
        pp.append(a)
    KP = pd.concat(pp, ignore_index=True)
    C.write_parquet(C.RES / "KA_pairs.parquet", KP)
    rep = dict(dev=True, note="developer dry run (fold 0 smoke units), not a result", C_P=cp,
               predictions_sha256=C.sha256(C.RES / "KA_predictions.parquet"),
               pairs_sha256=C.sha256(C.RES / "KA_pairs.parquet"))
    C.write_json(C.RES / "KA_report.json", rep)
    print(json.dumps(C.clean(dict(n_rx=len(KA), n_pairs=len(KP), C_P={m: v["ok"] for m, v in cp.items()})), indent=1))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["fakek"]:
        sys.exit(fake_k())
    print(__doc__)
    sys.exit(2)
