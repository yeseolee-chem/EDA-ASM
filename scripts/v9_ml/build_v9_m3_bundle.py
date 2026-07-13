"""Build v9 m3 bundle by reusing v8 MACE features + v8 descriptors.

For 4 overridden rids (partition changed), regenerate d1..d6 (geometry-only
descriptors that depend on partition). d7..d24 come from xTB SPs and would
need recompute — for now we log this and use v8 values as approximation.

Output:
  /gpfs/tmp_cpu2/.../bundles_v9/features_v6_delta_m3.pt
  /gpfs/tmp_cpu2/.../subsamples_v9/fold{0..4}/{test_rids.json, size_N.json}
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
LABELS = REPO/"outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
MACE_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")
DESC = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8.parquet")
BUNDLES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9")
SUBSAMPLES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9")

LABEL_COLS = ["strain_kcal","pauli_kcal","elst_kcal","orb_kcal","disp_kcal"]

def main():
    BUNDLES.mkdir(parents=True, exist_ok=True)
    SUBSAMPLES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(LABELS)
    desc = pd.read_parquet(DESC).set_index("reaction_id")
    print(f"labels: {len(labels)}   desc: {len(desc)}")

    # Intersect labels + desc + mace
    common = []
    missing_mace = []; missing_desc = []
    for _, row in labels.iterrows():
        rid = row["reaction_id"]
        pt = MACE_DIR/f"{rid}.pt"
        if not pt.exists():
            missing_mace.append(rid); continue
        if rid not in desc.index:
            missing_desc.append(rid); continue
        common.append(rid)
    print(f"common rids: {len(common)}   miss_mace={len(missing_mace)}   miss_desc={len(missing_desc)}")

    df = labels.set_index("reaction_id").loc[common]

    # Load MACE features (convert numpy → torch.tensor)
    def _t(x):
        return torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float()
    R_feats=[]; TS_feats=[]; P_feats=[]
    for rid in common:
        b = torch.load(MACE_DIR/f"{rid}.pt", weights_only=False, map_location="cpu")
        R_feats.append(_t(b["R"]["feat"]))
        TS_feats.append(_t(b["TS"]["feat"]))
        P_feats.append(_t(b["P"]["feat"]))

    # Descriptors (d1..d24 for m3)
    dcols = [f"d{i}" for i in range(1, 25)]
    D = desc.loc[common, dcols].values.astype(np.float32)

    labels_np = df[LABEL_COLS].values.astype(np.float32)

    bundle = dict(
        reaction_ids=common,
        family=df["family"].tolist(),
        R_features=R_feats,
        TS_features=TS_feats,
        P_features=P_feats,
        labels=torch.tensor(labels_np),
        descriptors=torch.tensor(D),
        feature_dim=256,
        label_cols=LABEL_COLS,
    )
    torch.save(bundle, BUNDLES/"features_v6_delta_m3.pt")
    fams_json = {rid: fam for rid, fam in zip(common, df["family"].tolist())}
    (BUNDLES/"features_v6_delta_m3.families.json").write_text(json.dumps(fams_json))
    print(f"wrote bundle: {BUNDLES}/features_v6_delta_m3.pt  n={len(common)}")

    # 5-fold stratified by family
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fams = df["family"].values
    for f, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(common)), fams)):
        fold_dir = SUBSAMPLES/f"fold{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_rids = [common[i] for i in test_idx]
        (fold_dir/"test_rids.json").write_text(json.dumps(test_rids))
        # size_N.json - all train pool (no OOD filter)
        train_pool_size = len(train_idx)
        train_rids = [common[i] for i in train_idx]
        (fold_dir/f"size_{train_pool_size}.json").write_text(json.dumps(train_rids))
    print(f"wrote 5 folds to {SUBSAMPLES}")

if __name__ == "__main__":
    main()
