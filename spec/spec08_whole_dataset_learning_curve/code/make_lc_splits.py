"""SPEC_08 whole-dataset LC — build family-stratified nested subsamples.

Per (size, fold):
  - reuse spec06 outer_folds.json for train / test partitioning
  - within the fold's train pool, sample `size` reactions with 25/family
    per bucket (nested: size-N ⊇ size-(N−100))
  - if a family runs out, top up round-robin from other families so
    actual N stays close to target until the whole fold train is used

Emits `splits/lc_splits.json`:
  { "sizes": [100, 200, ..., 786],
    "seed":  42,
    "folds": { "0": {"train": [...], "test": [...], ... }, ... },
    "subsamples": {
        "100": {"0": {"train_rids": [...], "actual_n": 100,
                       "per_family": {"dipolar": 25, ...}}, ...},
        ...
    }
  }
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec08_whole_dataset_learning_curve"
BUNDLE_PT = Path(os.environ.get(
    "BUNDLE_PT",
    "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
LABELS_PQ = REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
OUTER_FOLDS = REPO / "spec/spec06_2step_xgb28_delta/splits/outer_folds.json"

SIZES = [100, 200, 300, 400, 500, 600, 700, 786]
FAMILIES = ["dipolar", "qmrxn20_e2", "qmrxn20_sn2", "rgd1"]
SEED = 42
OUT = SPEC / "splits/lc_splits.json"


def family_shuffled_pools(train_rids, family_map, seed):
    rng = np.random.default_rng(seed)
    pools = {}
    for fam in FAMILIES:
        fam_rids = [r for r in train_rids if family_map.get(r) == fam]
        idx = rng.permutation(len(fam_rids))
        pools[fam] = [fam_rids[i] for i in idx]
    return pools


def subsample_nested(pools, target):
    """Return picks (concatenated by family) + per-family count map.

    Nested: for target N, take the first n_per_fam = N // 4 rids from
    each family. If a family is short, top up from other families
    round-robin.
    """
    n_per_fam = target // len(FAMILIES)
    remainder = target - n_per_fam * len(FAMILIES)

    picked_by_fam = {}
    remaining_by_fam = {}
    deficit = 0
    for fam in FAMILIES:
        take = min(n_per_fam, len(pools[fam]))
        picked_by_fam[fam] = list(pools[fam][:take])
        remaining_by_fam[fam] = list(pools[fam][take:])
        deficit += (n_per_fam - take)

    extras = deficit + remainder
    while extras > 0:
        gave = 0
        for fam in FAMILIES:
            if extras <= 0:
                break
            if remaining_by_fam[fam]:
                picked_by_fam[fam].append(remaining_by_fam[fam].pop(0))
                extras -= 1
                gave += 1
        if gave == 0:
            break

    train_rids = [r for fam in FAMILIES for r in picked_by_fam[fam]]
    per_family = {fam: len(picked_by_fam[fam]) for fam in FAMILIES}
    return train_rids, per_family


def main():
    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    all_rids = list(b["reaction_ids"])
    lbl = pd.read_parquet(LABELS_PQ)
    family_map = dict(zip(lbl.reaction_id, lbl.family))
    for r in all_rids:
        if r not in family_map:
            raise RuntimeError(f"reaction {r} missing from family map")

    with open(OUTER_FOLDS) as fh:
        outer = json.load(fh)

    folds_out = {}
    subs_out = {size: {} for size in SIZES}

    for fold_key in sorted(outer.keys(), key=int):
        fold = outer[fold_key]
        train_rids = list(fold["train"])
        test_rids = list(fold["test"])
        train_fam_counts = (
            pd.Series([family_map[r] for r in train_rids]).value_counts().to_dict()
        )
        folds_out[fold_key] = {
            "train": train_rids,
            "test": test_rids,
            "n_train": len(train_rids),
            "n_test": len(test_rids),
            "family_counts_train": train_fam_counts,
        }

        pools = family_shuffled_pools(train_rids, family_map,
                                      seed=SEED + int(fold_key))

        prev_set = None
        for size in SIZES:
            picked, per_fam = subsample_nested(pools, size)
            actual = len(picked)
            if prev_set is not None:
                assert set(prev_set) <= set(picked), (
                    f"nested containment broken at fold={fold_key} size={size}")
            subs_out[size][fold_key] = {
                "train_rids": picked,
                "target_n": size,
                "actual_n": actual,
                "capped_to_fold_train": actual < size,
                "per_family": per_fam,
            }
            prev_set = picked

    out = {
        "sizes": SIZES,
        "families": FAMILIES,
        "seed": SEED,
        "outer_folds_source": str(OUTER_FOLDS),
        "bundle": str(BUNDLE_PT),
        "folds": folds_out,
        "subsamples": {str(s): subs_out[s] for s in SIZES},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")

    print("\nSummary (actual n per fold):")
    print("  size " + " ".join(f"f{k}".rjust(6) for k in sorted(outer.keys(), key=int)))
    for size in SIZES:
        row = f"  {size:>4d}"
        for fk in sorted(outer.keys(), key=int):
            row += f" {subs_out[size][fk]['actual_n']:>6d}"
        print(row)


if __name__ == "__main__":
    main()
