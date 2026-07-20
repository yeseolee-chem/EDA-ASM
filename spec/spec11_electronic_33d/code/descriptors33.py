"""SPEC_11 - 33-descriptor matrix builder.

X33 = hstack[X28(d1..d28), d29, d30, d31, d32, d33]

Rows where scf_ok=False (or column missing/NaN) are 0-filled; the returned
ok_mask is True only when all five d29..d33 succeeded AND all four d25..d28
already succeeded (report-only).

Reuses spec06.descriptors28.build_X28 verbatim, then attaches the five new
electronic descriptors from data/descriptors_d29_d33.parquet using the same
scf_ok policy as spec06 (attach_col).
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "spec/spec06_2step_xgb28_delta/code"))
from descriptors28 import build_X28, attach_col  # noqa: E402

D29_33_PQ = REPO / "spec/spec11_electronic_33d/data/descriptors_d29_d33.parquet"


def build_X33(rids, X24):
    """Return (X33, ok_mask). X33 dtype = float64 to match spec06."""
    X28, ok28 = build_X28(rids, X24)
    d29, ok29 = attach_col(rids, D29_33_PQ, "d29")
    d30, ok30 = attach_col(rids, D29_33_PQ, "d30")
    d31, ok31 = attach_col(rids, D29_33_PQ, "d31")
    d32, ok32 = attach_col(rids, D29_33_PQ, "d32")
    d33, ok33 = attach_col(rids, D29_33_PQ, "d33")
    X33 = np.hstack([
        np.asarray(X28, dtype=np.float64),
        d29[:, None], d30[:, None], d31[:, None], d32[:, None], d33[:, None],
    ])
    ok = ok28 & ok29 & ok30 & ok31 & ok32 & ok33
    return X33, ok


if __name__ == "__main__":
    import argparse, os, torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.environ.get(
        "BUNDLE_PT",
        "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
    args = ap.parse_args()
    b = torch.load(args.bundle, weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy().astype(np.float64)
    X33, ok = build_X33(rids, X24)
    print(f"rids={len(rids)}  X24={X24.shape}  X33={X33.shape}  ok={int(ok.sum())}/{len(ok)}")
    for i, col in enumerate(["d29", "d30", "d31", "d32", "d33"]):
        v = X33[:, 28 + i]
        nz = int((v != 0).sum())
        print(f"  {col}: nonzero={nz}/{len(v)}  mean={v.mean():+.3e}  std={v.std():+.3e}")
