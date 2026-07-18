"""SPEC_06 B1 — build 5-fold splits per family (simple KFold, seed=42).

Reads:
  labels_v9_5channel.LOCKED_783.parquet  (reaction_id, family)
  v9 m3 bundle                            (reaction_ids)

Writes:
  splits/family_folds/{family}_outer_folds.json
     { "family": name,
       "all_rids": [...],
       "n": N,
       "folds": { "0": {"train": [...], "test": [...]}, ... "4": {...} } }
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE_PT = Path(os.environ.get(
    "BUNDLE_PT",
    "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
LABELS_PQ = REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
OUT_DIR = REPO / "spec/spec09_per_family_xgb28_delta/splits/family_folds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5


def main():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    all_rids = np.asarray(b["reaction_ids"])
    lbl = pd.read_parquet(LABELS_PQ)
    fam_map = dict(zip(lbl.reaction_id, lbl.family))

    families = sorted(set(fam_map[r] for r in all_rids if r in fam_map))
    print(f"[make_family_folds] cohort={len(all_rids)}  families={families}")

    for fam in families:
        fam_rids = np.array([r for r in all_rids if fam_map.get(r) == fam])
        n = len(fam_rids)
        rng = np.random.default_rng(SEED)
        perm = fam_rids[rng.permutation(n)]  # deterministic seed-42 shuffle
        kf = KFold(n_splits=N_FOLDS, shuffle=False)  # already shuffled
        folds = {}
        seen_test = set()
        for i, (tr, te) in enumerate(kf.split(perm)):
            folds[str(i)] = {"train": perm[tr].tolist(), "test": perm[te].tolist()}
            assert set(tr).isdisjoint(set(te)), f"{fam} fold{i}: overlap"
            for r in perm[te]:
                assert r not in seen_test, f"{fam} rxn {r} in multiple test folds"
                seen_test.add(r)
        assert seen_test == set(fam_rids), f"{fam}: test folds do not cover all rxns"

        out = OUT_DIR / f"{fam}_outer_folds.json"
        out.write_text(json.dumps({
            "family": fam, "n": int(n), "seed": SEED, "n_folds": N_FOLDS,
            "all_rids": fam_rids.tolist(),
            "folds": folds,
        }, indent=2))
        sizes = [len(folds[str(i)]["test"]) for i in range(N_FOLDS)]
        print(f"  {fam:<12s} n={n}  test-sizes={sizes}  → {out.name}")


if __name__ == "__main__":
    main()
