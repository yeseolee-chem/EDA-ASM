"""SPEC_10 — build within-family nested subsamples for the learning curve.

Reuses `spec/spec09_per_family_xgb28_delta/splits/family_folds/{family}_outer_folds.json`
(5-fold KFold splits inside each family, seed=42).

Per (family, fold):
  - deterministic-shuffle the fold's family-train roster once
    (seed = 42 + family_idx*10 + fold)
  - for target N in {50, 100, 150, 200}, take first min(N, family_train_size)
    rids from the shuffled roster

Emits `splits/lc_family_splits.json`:
  { "sizes":   [50, 100, 150, 200],
    "families": ["dipolar", ...],
    "seed_base": 42,
    "folds": {
        "<family>": {
            "0": {"train_pool": [...], "test": [...], "n_train": N, "n_test": M},
            ...
        }, ...
    },
    "subsamples": {
        "<family>": {
            "50":  { "0": {"train_rids": [...], "actual_n": 50}, ... },
            "100": {...}, ...
        }, ...
    }
  }
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec10_family_learning_curve"
FAM_SPLITS_DIR = REPO / "spec/spec09_per_family_xgb28_delta/splits/family_folds"

SIZES = [50, 100, 150]
FAMILIES = ["dipolar", "qmrxn20_e2", "qmrxn20_sn2", "rgd1"]
SEED_BASE = 42
OUT = SPEC / "splits/lc_family_splits.json"


def subsample_nested(shuffled_train_rids, target):
    actual = min(target, len(shuffled_train_rids))
    return list(shuffled_train_rids[:actual]), actual


def main():
    folds_out = {}
    subs_out = {fam: {str(s): {} for s in SIZES} for fam in FAMILIES}

    for fam_idx, fam in enumerate(FAMILIES):
        with open(FAM_SPLITS_DIR / f"{fam}_outer_folds.json") as fh:
            fam_folds = json.load(fh)
        folds_out[fam] = {}

        for fold_key in sorted(fam_folds["folds"].keys(), key=int):
            fold = fam_folds["folds"][fold_key]
            train_pool = list(fold["train"])
            test_pool = list(fold["test"])
            folds_out[fam][fold_key] = {
                "train_pool": train_pool,
                "test": test_pool,
                "n_train": len(train_pool),
                "n_test": len(test_pool),
            }

            seed = SEED_BASE + fam_idx * 10 + int(fold_key)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(train_pool))
            shuffled = [train_pool[i] for i in perm]

            prev_set = None
            for size in SIZES:
                picked, actual = subsample_nested(shuffled, size)
                if prev_set is not None:
                    assert set(prev_set) <= set(picked), (
                        f"nested containment broken at {fam} fold={fold_key} size={size}")
                subs_out[fam][str(size)][fold_key] = {
                    "train_rids": picked,
                    "target_n": size,
                    "actual_n": actual,
                    "capped_to_fold_train": actual < size,
                    "seed": seed,
                }
                prev_set = picked

    out = {
        "sizes": SIZES,
        "families": FAMILIES,
        "seed_base": SEED_BASE,
        "family_folds_source": str(FAM_SPLITS_DIR),
        "folds": folds_out,
        "subsamples": subs_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")

    print("\nSummary (actual n per fold):")
    header = "  family        size " + " ".join(f"f{k}".rjust(5) for k in ["0", "1", "2", "3", "4"])
    print(header)
    for fam in FAMILIES:
        for size in SIZES:
            row = f"  {fam:<14s} {size:>4d}"
            for fk in ["0", "1", "2", "3", "4"]:
                row += f" {subs_out[fam][str(size)][fk]['actual_n']:>5d}"
            print(row)


if __name__ == "__main__":
    main()
