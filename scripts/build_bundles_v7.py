"""Rebuild m1/m2/m3 bundles using the NEW 5-channel labels from ORCA v7.

Wraps pipeline_rebuild/spec_v1/stage4_build_bundles.py logic but points
at labels/orca/orca_eda_labels_v7.parquet (produced by parse_orca_5channel.py)
and writes bundles to a fresh location so nothing overlaps with old runs.

Bundles written to:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v7/{m1,m2,m3}.pt
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
DESC_PQ = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1.parquet")
LABELS_V7 = REPO / "labels/orca/orca_eda_labels_v7.parquet"
BUNDLE_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v7")
SPLIT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v7/trackB_no_ood")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["E_strain_kcal", "Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal"]
SEED = 42
N_FOLDS = 5

DESC_M1 = [f"d{i}" for i in range(1, 7)]
DESC_M2 = [f"d{i}" for i in range(1, 22)]
DESC_M3 = [f"d{i}" for i in range(1, 25)]


def main():
    if not LABELS_V7.exists():
        raise FileNotFoundError(f"labels_v7 not built yet: {LABELS_V7}")
    labels = pd.read_parquet(LABELS_V7)
    labels = labels.dropna(subset=LABEL_COLS)
    labels = labels.sort_values("reaction_id").reset_index(drop=True)
    print(f"labels: {len(labels)} rows with all 5 channels", flush=True)

    desc = pd.read_parquet(DESC_PQ)
    if "error" in desc.columns:
        desc = desc[desc["error"].isna()]
    desc = desc.set_index("reaction_id")

    common = labels[labels.reaction_id.isin(desc.index)].reset_index(drop=True)
    print(f"labels ∩ descriptors: {len(common)}", flush=True)

    y = common[LABEL_COLS].to_numpy(dtype=np.float32)
    families = common["family"].to_numpy()

    # Load MACE features per reaction
    def load_feat(rid: str, split: str):
        d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
        if split not in d or "feat" not in d[split]:
            return None
        return torch.as_tensor(d[split]["feat"], dtype=torch.float32)

    feats_R, feats_TS, feats_P = [], [], []
    valid = []
    for i, rid in enumerate(common.reaction_id):
        fR = load_feat(rid, "R"); fTS = load_feat(rid, "TS"); fP = load_feat(rid, "P")
        if any(x is None for x in (fR, fTS, fP)):
            continue
        feats_R.append(fR); feats_TS.append(fTS); feats_P.append(fP)
        valid.append(i)
    valid = np.array(valid, int)
    print(f"with full R/TS/P MACE features: {len(valid)}", flush=True)

    y = y[valid]; families = families[valid]
    rids = common.reaction_id.to_numpy()[valid]

    # Build per-model bundles
    def build(desc_cols, tag):
        # Get descriptor rows in same order
        X = desc.loc[rids, desc_cols].to_numpy(dtype=np.float32)
        bundle = dict(
            reaction_id=list(rids),
            family=list(families),
            desc=torch.as_tensor(X),
            y=torch.as_tensor(y),
            feat_R=feats_R,
            feat_TS=feats_TS,
            feat_P=feats_P,
            feature_dim=256,
            desc_cols=desc_cols,
            label_cols=LABEL_COLS,
        )
        out = BUNDLE_DIR / f"features_v7_delta_{tag}.pt"
        torch.save(bundle, out)
        # Compatibility: also write ".families.json" that some runners load.
        fam_json = BUNDLE_DIR / f"features_v7_delta_{tag}.families.json"
        fam_json.write_text(json.dumps(dict(zip(rids.tolist(), families.tolist()))))
        print(f"  wrote {out.name}  ({len(rids)} rxns, desc={X.shape[1]})")

    build(DESC_M1, "m1")
    build(DESC_M2, "m2")
    build(DESC_M3, "m3")

    # Build stratified 5-fold splits
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(skf.split(rids, families)):
        fold_dir = SPLIT_DIR / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "test_rids.json").write_text(json.dumps(rids[test_idx].tolist()))
        # size_N.json = the full training pool
        pool = rids[train_idx].tolist()
        (fold_dir / f"size_{len(pool)}.json").write_text(json.dumps(pool))
    print(f"splits: {N_FOLDS} folds → {SPLIT_DIR}")


if __name__ == "__main__":
    main()
