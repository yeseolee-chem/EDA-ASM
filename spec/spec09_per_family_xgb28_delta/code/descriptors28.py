"""SPEC_06 — 28-descriptor matrix builder.

Composes the 24-d m3 descriptor block from the v9 bundle with the four
channel-proxy features d25..d28 from spec05, matching the exact spec05
`no_sum_28d` composition (the winning xgb_28d base).

Single source of truth: `build_X28(rids, X24) -> (X28, ok_mask)`.

d25       — spec/spec05_d25_sum/data/descriptors_d25_refR.parquet    (col d25, gate scf_ok)
d26..d28  — spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet (cols d26,d27,d28, gate scf_ok)

Rows where scf_ok=False (or column missing/NaN) are 0-filled; the returned
ok_mask is True only when all four proxies succeeded (used only for reporting).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
D25_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_d25_refR.parquet"
D26_28_PQ = REPO / "spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet"


def attach_col(rids, parquet_path: Path, col: str):
    df = pd.read_parquet(parquet_path).set_index("reaction_id")
    vals, ok = [], []
    for r in rids:
        if r in df.index and bool(df.loc[r, "scf_ok"]):
            v = df.loc[r, col]
            if pd.isna(v):
                vals.append(0.0); ok.append(False)
            else:
                vals.append(float(v)); ok.append(True)
        else:
            vals.append(0.0); ok.append(False)
    return np.array(vals, dtype=np.float64), np.array(ok, dtype=bool)


def build_X28(rids, X24):
    """Return (X28, ok_mask). X28 dtype = float64 to match spec05 xgb_28d.

    ok_mask is True where all four proxies were populated. Rows where any
    proxy is missing are still returned (0-filled) so that indices stay
    aligned with the bundle — the fold logic never drops rows.
    """
    d25, ok25 = attach_col(rids, D25_PQ, "d25")
    d26, ok26 = attach_col(rids, D26_28_PQ, "d26")
    d27, ok27 = attach_col(rids, D26_28_PQ, "d27")
    d28, ok28 = attach_col(rids, D26_28_PQ, "d28")
    X28 = np.hstack([
        np.asarray(X24, dtype=np.float64),
        d25[:, None], d26[:, None], d27[:, None], d28[:, None],
    ])
    ok = ok25 & ok26 & ok27 & ok28
    return X28, ok


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
    X28, ok = build_X28(rids, X24)
    print(f"rids={len(rids)}  X24={X24.shape}  X28={X28.shape}  ok={int(ok.sum())}/{len(ok)}")
