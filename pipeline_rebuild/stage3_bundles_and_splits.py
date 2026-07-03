"""Stage 3 — assemble the CachedFeatureBundleDelta bundle for m1 (geom6) and
regenerate the 5-fold stratified splits the runner expects.

Inputs:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt  (789 files)
  labels/adf/adf_labels_v6_multifamily.parquet                          (targets)

Outputs:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles/
      features_v6_delta_geom6.pt
      features_v6_delta_geom6.families.json
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples/trackB_no_ood/
      fold0/test_rids.json  size_509.json
      fold1/... fold4/...

The bundle layout matches CachedFeatureBundleDelta:
  reaction_ids  : list[str]           len N
  R_features    : list[Tensor (n_i, 256)]
  TS_features   : list[Tensor (n_i, 256)]
  P_features    : list[Tensor (n_i, 256)]
  labels        : Tensor (N, 5)       (E_strain, Pauli, V_elst, E_orb, E_disp)
  descriptors   : Tensor (N, 6)       (geom6 = d1..d6)
  feature_dim   : int                 256

Splits: stratified 5-fold on family; per fold pick 509 train from the
remaining 631 (deterministic seed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
from eda_asm.asr_v1.baseline_physics import compute_descriptors
from ase import Atoms

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
BUNDLE_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles")
SPLIT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples/trackB_no_ood")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Label columns in the parquet (must match ASR_COMPONENTS order in data.py)
LABEL_COLS = ["E_strain_kcal", "Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal"]

SEED = 42
SIZE_FULL = 509
N_FOLDS = 5


def load_all():
    df = pd.read_parquet(REPO / "labels/adf/adf_labels_v6_multifamily.parquet")
    df = df.sort_values("reaction_id").reset_index(drop=True)
    print(f"[{time.strftime('%H:%M:%S')}] {len(df)} cohort reactions")
    return df


def assemble_bundle(df: pd.DataFrame):
    reaction_ids, R_feats, TS_feats, P_feats = [], [], [], []
    descriptors, labels_rows, families = [], [], []
    t0 = time.time()
    for i, row in df.iterrows():
        rid = row.reaction_id
        pt = FEAT_DIR / f"{rid}.pt"
        if not pt.exists():
            raise FileNotFoundError(pt)
        d = torch.load(str(pt), map_location="cpu", weights_only=False)
        reaction_ids.append(rid)
        R_feats.append(torch.from_numpy(d["R"]["feat"]).float())
        TS_feats.append(torch.from_numpy(d["TS"]["feat"]).float())
        P_feats.append(torch.from_numpy(d["P"]["feat"]).float())
        # geom6 descriptors
        R_at = Atoms(numbers=d["R"]["z"], positions=d["R"]["pos"])
        TS_at = Atoms(numbers=d["TS"]["z"], positions=d["TS"]["pos"])
        # For qmrxn20 e2/sn2, the product XYZ loses atoms (leaving group +
        # transferred proton). d2 = RMSD(P, TS) needs matching shapes, so we
        # fall back to using TS as the P surrogate — the R-side strain (d1) is
        # still captured, and the product-side term collapses to 0.
        p_pos = np.asarray(d["P"]["pos"])
        p_z = d["P"]["z"]
        if len(p_z) != len(d["TS"]["z"]):
            p_pos = np.asarray(d["TS"]["pos"])
            p_z = d["TS"]["z"]
        P_at = Atoms(numbers=p_z, positions=p_pos)
        descriptors.append(compute_descriptors(R_at, TS_at, P_at))
        labels_rows.append(row[LABEL_COLS].to_numpy(dtype=np.float32))
        families.append(row.family)
        if (i + 1) % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] assembled {i+1}/{len(df)}")

    labels = torch.tensor(np.stack(labels_rows), dtype=torch.float32)
    descriptors_t = torch.tensor(np.stack(descriptors), dtype=torch.float32)
    feature_dim = int(R_feats[0].shape[1])
    print(f"[{time.strftime('%H:%M:%S')}] bundle assembly done in "
          f"{time.time()-t0:.1f}s   feature_dim={feature_dim}")
    return reaction_ids, R_feats, TS_feats, P_feats, labels, descriptors_t, feature_dim, families


def save_bundle(rids, Rf, Tf, Pf, labels, descs, fdim, families, baseline_tag: str):
    obj = {
        "reaction_ids": rids,
        "R_features": Rf, "TS_features": Tf, "P_features": Pf,
        "labels": labels, "descriptors": descs,
        "feature_dim": fdim,
    }
    out = BUNDLE_DIR / f"features_v6_delta_{baseline_tag}.pt"
    torch.save(obj, out)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    fam_json = BUNDLE_DIR / f"features_v6_delta_{baseline_tag}.families.json"
    with open(fam_json, "w") as f:
        json.dump({rid: fam for rid, fam in zip(rids, families)}, f)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {fam_json}")


def stratified_5fold(df, seed=SEED):
    """Return list of 5 (test_rids, train_rids) tuples. Stratified by family."""
    rng = np.random.default_rng(seed)
    per_fold_test = [[] for _ in range(N_FOLDS)]
    for fam in sorted(df.family.unique()):
        sub = df[df.family == fam].reaction_id.tolist()
        rng.shuffle(sub)
        # round-robin distribution across folds
        for i, rid in enumerate(sub):
            per_fold_test[i % N_FOLDS].append(rid)

    splits = []
    all_rids = set(df.reaction_id)
    for f in range(N_FOLDS):
        test = sorted(per_fold_test[f])
        train_pool = sorted(all_rids - set(test))
        # subsample train_pool → SIZE_FULL with a fold-specific seed
        pool_rng = np.random.default_rng(seed * 1000 + f)
        idx = pool_rng.permutation(len(train_pool))[:SIZE_FULL]
        train = sorted([train_pool[i] for i in idx])
        splits.append((test, train))
    return splits


def save_splits(splits):
    for f, (test, train) in enumerate(splits):
        fdir = SPLIT_DIR / f"fold{f}"
        fdir.mkdir(exist_ok=True)
        with open(fdir / "test_rids.json", "w") as fh:
            json.dump(test, fh)
        with open(fdir / f"size_{SIZE_FULL}.json", "w") as fh:
            json.dump(train, fh)
        print(f"  fold{f}: test={len(test)} train={len(train)}")


def main():
    df = load_all()
    rids, Rf, Tf, Pf, labels, descs, fdim, families = assemble_bundle(df)
    save_bundle(rids, Rf, Tf, Pf, labels, descs, fdim, families, "geom6")

    print(f"[{time.strftime('%H:%M:%S')}] generating 5-fold stratified splits (seed={SEED})")
    splits = stratified_5fold(df)
    save_splits(splits)

    manifest = {
        "n_reactions": len(rids),
        "family_counts": df.family.value_counts().to_dict(),
        "feature_dim": fdim,
        "seed": SEED,
        "size_full": SIZE_FULL,
        "n_folds": N_FOLDS,
        "baseline_tags_built": ["geom6"],
    }
    (BUNDLE_DIR / "stage3_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] Stage 3 done.")


if __name__ == "__main__":
    main()
