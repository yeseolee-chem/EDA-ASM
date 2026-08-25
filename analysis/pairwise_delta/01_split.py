#!/usr/bin/env python3
"""Step 1: build 2 split schemes.

Split-A: GroupKFold by connected component (162 groups, greedy bin-pack).
Split-B: swap_type holdout (30 types × 5 folds; each type in test once).
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta")
OUT = BASE / "artifacts"

N_FOLDS = 5
SEED_ROOT = 42


def bin_pack_groups(groups_sizes, n_folds, seed):
    """Sort groups by size desc (jittered), assign each to currently smallest fold."""
    rng = np.random.default_rng(seed)
    items = [(sz, rng.random(), gid) for gid, sz in groups_sizes.items()]
    items.sort(key=lambda x: (-x[0], x[1]))
    fold_of = {}
    fold_sizes = [0] * n_folds
    for sz, _, gid in items:
        f = int(np.argmin(fold_sizes))
        fold_of[gid] = f
        fold_sizes[f] += sz
    return fold_of, fold_sizes


def main():
    with open(OUT / "pair_dataset.pkl", "rb") as f:
        pd_ = pickle.load(f)
    pairs = pd_["pairs"]
    n = len(pairs)

    # Split-A: by comp_id
    comp_sizes = pairs.groupby("comp_id").size().to_dict()
    print(f"Components involved in pairs: {len(comp_sizes)}  max size={max(comp_sizes.values())}")
    comp_fold_A, sizes_A = bin_pack_groups(comp_sizes, N_FOLDS, SEED_ROOT)
    pairs["fold_A"] = pairs["comp_id"].map(comp_fold_A).astype(int)
    print(f"Split-A fold sizes (pairs): {sizes_A}")

    # Verify: no comp spans two folds (assured by design)
    check = pairs.groupby("comp_id")["fold_A"].nunique()
    assert (check == 1).all(), "Split-A leakage: some comp spans folds"

    # Split-B: by swap_type
    sw_sizes = pairs.groupby("swap_type").size().to_dict()
    print(f"Swap types: {len(sw_sizes)}  sizes: {sorted(sw_sizes.values(), reverse=True)[:5]}... "
          f"{sorted(sw_sizes.values())[:5]}")
    sw_fold, sw_sz = bin_pack_groups(sw_sizes, N_FOLDS, SEED_ROOT)
    pairs["fold_B"] = pairs["swap_type"].map(sw_fold).astype(int)
    print(f"Split-B fold sizes (pairs): {sw_sz}")
    # Verify: swap_type unique per fold
    check = pairs.groupby("swap_type")["fold_B"].nunique()
    assert (check == 1).all(), "Split-B leakage: swap type in multiple folds"

    # Persist
    with open(OUT / "pair_dataset.pkl", "wb") as f:
        pd_["pairs"] = pairs
        pickle.dump(pd_, f)

    fold_summary = {
        "split_A_component": sizes_A,
        "split_B_swaptype": sw_sz,
        "n_pairs": n,
        "n_components": len(comp_sizes),
        "n_swaptypes": len(sw_sizes),
    }
    with open(OUT / "splits_summary.txt", "w") as f:
        for k, v in fold_summary.items():
            f.write(f"{k}: {v}\n")
    print("=== GATE-1 checks passed (no leakage in either split) ===")
    with open(OUT / "GATE1_STATUS.txt", "w") as f:
        f.write(f"PASS split_A_sizes={sizes_A} split_B_sizes={sw_sz}\n")


if __name__ == "__main__":
    main()
