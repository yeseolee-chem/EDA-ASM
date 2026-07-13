"""Arm A - xgb_direct: pooled OOF predictions from per-channel XGB on X only.

For each outer fold f, fit XGB on outer_train(f), predict on outer_test(f).
Concatenate across folds to produce pooled OOF -> spec/spec02_abc_ablation/oof/oof_A.parquet
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
from baselines import fit_xgb, predict_xgb  # noqa: E402

BUNDLE_PT = Path(os.environ.get("BUNDLE_PT", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
FOLDS_JSON = REPO / "spec/spec02_abc_ablation/splits/outer_folds.json"
OUT = REPO / "spec/spec02_abc_ablation/oof/oof_A.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]


def main():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X = b["descriptors"].numpy()
    Y = b["labels"].numpy()
    r2i = {r: i for i, r in enumerate(rids)}
    folds = json.load(open(FOLDS_JSON))
    folds = {int(k): v for k, v in folds.items()}

    oof_rows = []
    for f_i in sorted(folds):
        tr = np.array([r2i[r] for r in folds[f_i]["train"]])
        te = np.array([r2i[r] for r in folds[f_i]["test"]])
        print(f"[A] fold{f_i}: train={len(tr)} test={len(te)}", flush=True)
        m = fit_xgb(X[tr], Y[tr])
        yp = predict_xgb(m, X[te])
        for i_te, idx in enumerate(te):
            oof_rows.append({
                "fold": f_i, "reaction_id": rids[idx],
                **{f"y_true_{c}": float(Y[idx, i]) for i, c in enumerate(CHANNELS)},
                **{f"y_pred_{c}": float(yp[i_te, i]) for i, c in enumerate(CHANNELS)},
            })
    df = pd.DataFrame(oof_rows)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
