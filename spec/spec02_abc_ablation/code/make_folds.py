"""T2 - build family-stratified reaction-level 5-fold split and save
outer_folds.json. All three arms (A/B/C) share this file (gate #1).

Inputs:
  - v7 m3 bundle (776 rxns) -> reaction_ids + family

Output:
  spec/spec02_abc_ablation/splits/outer_folds.json  (dict: fold -> {train, test})
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v7/features_v7_delta_m3.pt")
OUT = REPO / "spec/spec02_abc_ablation/splits/outer_folds.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5


def main():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    labels = pd.read_parquet(REPO / "labels/orca/orca_eda_labels_v7.parquet")
    fam_map = dict(zip(labels.reaction_id, labels.family))
    fams = np.array([fam_map[r] for r in rids])

    print(f"cohort: {len(rids)} rxns, families: {dict(pd.Series(fams).value_counts())}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = {}
    for i, (tr, te) in enumerate(skf.split(rids, fams)):
        assert set(tr).isdisjoint(set(te)), f"fold{i}: train/test overlap!"
        folds[i] = {"train": rids[tr].tolist(), "test": rids[te].tolist(),
                    "family_train": {k: int(v) for k, v in pd.Series(fams[tr]).value_counts().items()},
                    "family_test":  {k: int(v) for k, v in pd.Series(fams[te]).value_counts().items()}}
        print(f"fold{i}: train={len(tr)} test={len(te)}  fam_test={folds[i]['family_test']}")

    all_test_rids = [r for fold in folds.values() for r in fold["test"]]
    assert len(all_test_rids) == len(set(all_test_rids)), "reaction appears in multiple test folds!"
    assert set(all_test_rids) == set(rids), "test folds do not cover all rxns"

    OUT.write_text(json.dumps(folds, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
