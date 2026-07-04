"""Stage 4 — assemble spec-compliant bundles for m1/m2/m3 and 5-fold splits.

- m1 bundle: descriptors = d1..d6 (6-d)      → features_v6_delta_m1.pt
- m2 bundle: descriptors = d1..d21 (21-d)    → features_v6_delta_m2.pt
- m3 bundle: descriptors = d1..d24 (24-d)    → features_v6_delta_m3.pt

Each bundle also carries the reaction_id list, R/TS/P MACE feature tensors
(256-d), 5-channel labels, and feature_dim (256).

Fold splits: stratified 5-fold on family, seed=42. Per fold, train pool =
789 − test; 15% of train_pool is carved off in-runner as val (with a
member-dependent seed). We write both test_rids.json and size_{N}.json
where N = |train_pool| so the runner's `size_{SIZE_FULL}` lookup finds
the whole training pool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
DESC_PQ = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1.parquet")
BUNDLE_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1")
SPLIT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v1/trackB_no_ood")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["E_strain_kcal", "Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal"]
SEED = 42
N_FOLDS = 5

DESC_M1 = [f"d{i}" for i in range(1, 7)]
DESC_M2 = [f"d{i}" for i in range(1, 22)]
DESC_M3 = [f"d{i}" for i in range(1, 25)]


def main():
    labels = pd.read_parquet(REPO / "labels/adf/adf_labels_v6_multifamily.parquet")
    labels = labels.sort_values("reaction_id").reset_index(drop=True)

    desc = pd.read_parquet(DESC_PQ)
    # Filter out rows with an "error" column set (partial rebuilds may leave those)
    if "error" in desc.columns:
        n_before = len(desc)
        desc = desc[desc["error"].isna()] if desc["error"].dtype == object else desc
        # Drop rows where any required column missing
        for col in DESC_M3:
            desc = desc[desc[col].notna()] if col in desc.columns else desc
        print(f"filtered descriptors: {n_before} -> {len(desc)}")
    desc = desc.set_index("reaction_id")

    # Take the intersection so bundles are aligned.
    have = [rid for rid in labels.reaction_id if rid in desc.index]
    missing = [rid for rid in labels.reaction_id if rid not in desc.index]
    print(f"cohort with valid descriptors: {len(have)}   missing: {len(missing)}")
    if missing[:5]:
        print("  first missing:", missing[:5])
    labels = labels[labels.reaction_id.isin(have)].reset_index(drop=True)

    reaction_ids = labels.reaction_id.tolist()
    families = labels.family.tolist()

    R_feats, TS_feats, P_feats = [], [], []
    for rid in reaction_ids:
        d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
        R_feats.append(torch.from_numpy(d["R"]["feat"]).float())
        TS_feats.append(torch.from_numpy(d["TS"]["feat"]).float())
        P_feats.append(torch.from_numpy(d["P"]["feat"]).float())
    feature_dim = int(R_feats[0].shape[1])
    labels_t = torch.tensor(labels[LABEL_COLS].to_numpy(dtype=np.float32))

    for name, cols in [("m1", DESC_M1), ("m2", DESC_M2), ("m3", DESC_M3)]:
        D = np.stack([desc.loc[rid, cols].to_numpy(dtype=np.float32)
                      for rid in reaction_ids], axis=0)
        obj = {
            "reaction_ids": reaction_ids,
            "R_features": R_feats, "TS_features": TS_feats, "P_features": P_feats,
            "labels": labels_t,
            "descriptors": torch.from_numpy(D),
            "feature_dim": feature_dim,
        }
        out = BUNDLE_DIR / f"features_v6_delta_{name}.pt"
        torch.save(obj, out)
        print(f"wrote {out.name}   D={D.shape[1]}   {out.stat().st_size/1e6:.1f} MB")

        fam = BUNDLE_DIR / f"features_v6_delta_{name}.families.json"
        fam.write_text(json.dumps(dict(zip(reaction_ids, families))))

    # 5-fold stratified splits
    rng = np.random.default_rng(SEED)
    per_fold_test = [[] for _ in range(N_FOLDS)]
    for f in sorted(set(families)):
        sub = labels[labels.family == f].reaction_id.tolist()
        rng.shuffle(sub)
        for i, rid in enumerate(sub):
            per_fold_test[i % N_FOLDS].append(rid)

    all_rids = set(reaction_ids)
    # Runner expects size_509.json (SIZE_FULL = 509 hardcoded). Subsample the
    # train_pool down to 509 per fold with a deterministic per-fold seed so the
    # runs are reproducible while satisfying that path convention.
    SIZE_FULL = 509
    for f in range(N_FOLDS):
        test = sorted(per_fold_test[f])
        train_pool = sorted(all_rids - set(test))
        fdir = SPLIT_DIR / f"fold{f}"
        fdir.mkdir(exist_ok=True)
        (fdir / "test_rids.json").write_text(json.dumps(test))
        if len(train_pool) >= SIZE_FULL:
            pool_rng = np.random.default_rng(SEED * 1000 + f)
            idx = pool_rng.permutation(len(train_pool))[:SIZE_FULL]
            train = sorted([train_pool[i] for i in idx])
        else:
            train = train_pool
        (fdir / f"size_{SIZE_FULL}.json").write_text(json.dumps(train))
        (fdir / f"size_{len(train_pool)}.json").write_text(json.dumps(train_pool))
        print(f"fold{f}: test={len(test)} train_pool={len(train_pool)} "
              f"train_used={len(train)}")

    (BUNDLE_DIR / "stage4_manifest.json").write_text(json.dumps({
        "n_reactions": len(reaction_ids),
        "family_counts": labels.family.value_counts().to_dict(),
        "feature_dim": feature_dim,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "bundles": ["m1", "m2", "m3"],
        "descriptor_dims": {"m1": 6, "m2": 21, "m3": 24},
    }, indent=2))
    print("Stage 4 done.")


if __name__ == "__main__":
    main()
