"""Arm A — `xgb_direct` OOF prediction on the 787-reaction cohort.

For each outer fold:
  - Fit XGB(X_train) per-channel (5 heads)
  - Predict on X_test → 5-channel prediction rows

Emit:
  results/oof_pred_A.parquet  columns = [reaction_id, fold, y*_c, y_c] for c=0..4

CPU-only. Deterministic seed = 42.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import fit_xgb  # noqa: E402

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
DEFAULT_BUNDLE = REPO / "pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.pt"
SPLITS = HERE / "splits" / "outer_folds.json"
OUT = HERE / "results" / "oof_pred_A.parquet"
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor-set", choices=["m1", "m2", "m3"], default="m3")
    ap.add_argument("--bundle", type=str, default=None,
                    help="Override bundle path (else derived from descriptor-set).")
    args = ap.parse_args()

    if args.bundle is None:
        args.bundle = str(REPO / f"pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_{args.descriptor_set}.pt")
    print(f"loading bundle={args.bundle}", flush=True)
    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    rids = list(b["reaction_ids"])
    X = b["descriptors"].numpy().astype(np.float32)
    Y = b["labels"].numpy().astype(np.float32)
    r2i = {r: i for i, r in enumerate(rids)}
    print(f"  X={X.shape} Y={Y.shape}", flush=True)

    splits = json.load(open(SPLITS))
    assert splits["seed"] == SEED, f"seed mismatch: {splits['seed']}"

    rows = []
    for fd in splits["folds"]:
        f = fd["fold"]
        tr = np.array([r2i[r] for r in fd["train_rids"]])
        te = np.array([r2i[r] for r in fd["test_rids"]])
        assert set(tr).isdisjoint(te), f"fold{f} leakage"
        t0 = time.time()
        m = fit_xgb(X[tr], Y[tr], seed=SEED + f)
        Yp = m.predict(X[te])
        print(f"  fold{f}: n_train={len(tr)} n_test={len(te)} "
              f"MAE_per_ch={np.abs(Yp - Y[te]).mean(axis=0).round(2).tolist()} "
              f"({time.time()-t0:.1f}s)", flush=True)
        for i, ridx in enumerate(te):
            rows.append({
                "reaction_id": rids[ridx],
                "fold": f,
                **{f"y_{CH[c]}": float(Y[ridx, c]) for c in range(5)},
                **{f"yhat_{CH[c]}": float(Yp[i, c]) for c in range(5)},
            })

    df = pd.DataFrame(rows).sort_values("reaction_id").reset_index(drop=True)
    assert len(df) == len(rids), f"OOF count {len(df)} != N {len(rids)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote → {OUT}", flush=True)


if __name__ == "__main__":
    main()
