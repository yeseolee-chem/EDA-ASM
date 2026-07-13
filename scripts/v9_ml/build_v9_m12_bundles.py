"""Build v9 m1 (d1..d6) and v9 m2 (d1..d21) bundles by reusing v8 MACE features + v8 descriptors.
Splits already produced by build_v9_m3_bundle.py — same folds reused."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
LABELS = REPO/"outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
MACE_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")
DESC = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8.parquet")
BUNDLES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9")

LABEL_COLS = ["strain_kcal","pauli_kcal","elst_kcal","orb_kcal","disp_kcal"]

def _t(x):
    return torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float()

def main():
    BUNDLES.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(LABELS).set_index("reaction_id")
    desc = pd.read_parquet(DESC).set_index("reaction_id")

    common = [rid for rid in labels.index
              if (MACE_DIR/f"{rid}.pt").exists() and rid in desc.index]
    print(f"common rids: {len(common)}")

    df = labels.loc[common]
    R_feats=[]; TS_feats=[]; P_feats=[]
    for rid in common:
        b = torch.load(MACE_DIR/f"{rid}.pt", weights_only=False, map_location="cpu")
        R_feats.append(_t(b["R"]["feat"]))
        TS_feats.append(_t(b["TS"]["feat"]))
        P_feats.append(_t(b["P"]["feat"]))

    labels_np = df[LABEL_COLS].values.astype(np.float32)
    labels_t = torch.tensor(labels_np)

    for mk, cols in [("m1", [f"d{i}" for i in range(1,7)]),
                     ("m2", [f"d{i}" for i in range(1,22)])]:
        D = desc.loc[common, cols].values.astype(np.float32)
        bundle = dict(
            reaction_ids=common,
            family=df["family"].tolist(),
            R_features=R_feats,
            TS_features=TS_feats,
            P_features=P_feats,
            labels=labels_t,
            descriptors=torch.tensor(D),
            feature_dim=256,
            label_cols=LABEL_COLS,
        )
        torch.save(bundle, BUNDLES/f"features_v6_delta_{mk}.pt")
        fams = {rid: fam for rid, fam in zip(common, df["family"].tolist())}
        (BUNDLES/f"features_v6_delta_{mk}.families.json").write_text(json.dumps(fams))
        print(f"wrote {mk}: n={len(common)} descriptor_dim={D.shape[1]}")

if __name__ == "__main__":
    main()
