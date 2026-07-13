"""Stage 4 (v8) — assemble spec-compliant bundles for m1/m2/m3 and 5-fold splits
for the 799-reaction v8 cohort.

**No OOD filtering.** All 799 reactions in
outputs/v8_review/labels/labels_v8_5channel.parquet are used.

Bundles (one per model):
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8/features_v6_delta_m1.pt   (D=6)
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8/features_v6_delta_m2.pt   (D=21)
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8/features_v6_delta_m3.pt   (D=24)

Each bundle carries:
  reaction_ids, family, R_features, TS_features, P_features, labels[N,5],
  descriptors[N,D], feature_dim=256, label_cols

Label channel order (matches ASR_COMPONENTS in the runner and the plot code in
spec_v1 stage6):
  [strain_kcal, pauli_kcal, elst_kcal, orb_kcal, disp_kcal]

Splits (stratified 5-fold on family, seed=42):
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v8/fold{F}/
      test_rids.json        — fold-out reaction ids
      size_full.json        — full training pool (~640 rids; NO subsampling)
Also:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v8/all_rids.json
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8/stage4_manifest.json
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

LABELS_PARQUET = REPO / "outputs/v8_review/labels/labels_v8_5channel.parquet"
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")
DESC_PQ = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8.parquet")
BUNDLE_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8")
SPLIT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v8")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Channel order (strain first, then Pauli/elst/orb/disp).
# NOTE: This must match ASR_COMPONENTS in eda_asm.asr_v1.data. The runner
# assumes labels[:,0]=strain, [:,1]=Pauli, [:,2]=Velst, [:,3]=oi, [:,4]=disp.
LABEL_COLS = ["strain_kcal", "pauli_kcal", "elst_kcal", "orb_kcal", "disp_kcal"]

SEED = 42
N_FOLDS = 5

DESC_M1 = [f"d{i}" for i in range(1, 7)]
DESC_M2 = [f"d{i}" for i in range(1, 22)]
DESC_M3 = [f"d{i}" for i in range(1, 25)]


def main():
    # -------------------------------------------------------------------
    # 1) Load labels — all 799, no OOD filtering
    # -------------------------------------------------------------------
    labels = pd.read_parquet(LABELS_PARQUET)
    labels = labels.sort_values("reaction_id").reset_index(drop=True)
    print(f"[labels] loaded {len(labels)} rxns from {LABELS_PARQUET.name}")
    for c in LABEL_COLS + ["family", "reaction_id"]:
        if c not in labels.columns:
            raise KeyError(f"labels parquet missing column: {c}")

    # -------------------------------------------------------------------
    # 2) Load descriptors (d1..d24)
    # -------------------------------------------------------------------
    desc = pd.read_parquet(DESC_PQ)
    print(f"[desc] loaded {len(desc)} rows, columns: {list(desc.columns)[:6]}...")
    if "error" in desc.columns:
        n_before = len(desc)
        try:
            desc = desc[desc["error"].isna()]
        except Exception:
            pass
        for col in DESC_M3:
            if col in desc.columns:
                desc = desc[desc[col].notna()]
        print(f"[desc] filtered {n_before} -> {len(desc)} (dropped errors / NaN)")
    else:
        for col in DESC_M3:
            if col in desc.columns:
                desc = desc[desc[col].notna()]
    desc = desc.set_index("reaction_id")

    # -------------------------------------------------------------------
    # 3) Intersect labels + descriptors + MACE features (all 3 must exist)
    # -------------------------------------------------------------------
    have = []
    missing_desc = []
    missing_feat = []
    for rid in labels.reaction_id:
        if rid not in desc.index:
            missing_desc.append(rid)
            continue
        if not (FEAT_DIR / f"{rid}.pt").exists():
            missing_feat.append(rid)
            continue
        have.append(rid)
    print(f"[cohort] usable={len(have)}   missing_desc={len(missing_desc)}   "
          f"missing_feat={len(missing_feat)}")
    if missing_desc[:5]:
        print(f"  first missing descriptors: {missing_desc[:5]}")
    if missing_feat[:5]:
        print(f"  first missing MACE feats: {missing_feat[:5]}")
    labels = labels[labels.reaction_id.isin(have)].reset_index(drop=True)

    reaction_ids = labels.reaction_id.tolist()
    families = labels.family.tolist()

    # -------------------------------------------------------------------
    # 4) Load MACE features into per-state lists
    # -------------------------------------------------------------------
    R_feats, TS_feats, P_feats = [], [], []
    for i, rid in enumerate(reaction_ids):
        d = torch.load(str(FEAT_DIR / f"{rid}.pt"),
                       map_location="cpu", weights_only=False)
        R_feats.append(torch.from_numpy(d["R"]["feat"]).float())
        TS_feats.append(torch.from_numpy(d["TS"]["feat"]).float())
        P_feats.append(torch.from_numpy(d["P"]["feat"]).float())
        if (i + 1) % 100 == 0:
            print(f"  loaded {i+1}/{len(reaction_ids)} MACE tensors", flush=True)
    feature_dim = int(R_feats[0].shape[1])
    print(f"[feat] loaded {len(R_feats)} tensors, feature_dim={feature_dim}")

    labels_t = torch.tensor(labels[LABEL_COLS].to_numpy(dtype=np.float32))
    families_map = dict(zip(reaction_ids, families))

    # -------------------------------------------------------------------
    # 5) Emit one bundle per model with the appropriate descriptor slice
    # -------------------------------------------------------------------
    for name, cols in [("m1", DESC_M1), ("m2", DESC_M2), ("m3", DESC_M3)]:
        D = np.stack([desc.loc[rid, cols].to_numpy(dtype=np.float32)
                      for rid in reaction_ids], axis=0)
        obj = {
            "reaction_ids": reaction_ids,
            "family": families,
            "R_features": R_feats,
            "TS_features": TS_feats,
            "P_features": P_feats,
            "labels": labels_t,
            "descriptors": torch.from_numpy(D),
            "feature_dim": feature_dim,
            "label_cols": LABEL_COLS,
        }
        out = BUNDLE_DIR / f"features_v6_delta_{name}.pt"
        torch.save(obj, out)
        print(f"[bundle] wrote {out.name}   D={D.shape[1]}   "
              f"{out.stat().st_size/1e6:.1f} MB")

        # Also write the families sidecar (spec_v1 runner reads this if present).
        fam = BUNDLE_DIR / f"features_v6_delta_{name}.families.json"
        fam.write_text(json.dumps(families_map))

    # -------------------------------------------------------------------
    # 6) 5-fold stratified splits by family. NO OOD filtering; NO subsample.
    # -------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    per_fold_test: list[list[str]] = [[] for _ in range(N_FOLDS)]
    for f in sorted(set(families)):
        sub = labels[labels.family == f].reaction_id.tolist()
        rng.shuffle(sub)
        for i, rid in enumerate(sub):
            per_fold_test[i % N_FOLDS].append(rid)

    all_rids = set(reaction_ids)
    (SPLIT_DIR / "all_rids.json").write_text(json.dumps(sorted(all_rids)))
    print(f"[splits] wrote all_rids.json ({len(all_rids)} rxns)")

    fold_summary = []
    for f in range(N_FOLDS):
        test = sorted(per_fold_test[f])
        train_pool = sorted(all_rids - set(test))
        fdir = SPLIT_DIR / f"fold{f}"
        fdir.mkdir(exist_ok=True)
        (fdir / "test_rids.json").write_text(json.dumps(test))
        # Full training pool — no SIZE_FULL subsampling.
        # We emit size_{N}.json (numeric N == train_pool size). The runner reads
        # this via its "pick the largest size_*.json" fallback when SIZE_FULL
        # env var doesn't match. NOTE: filename must be numeric because the
        # runner's fallback does int(p.stem.split("_")[1]).
        n_pool = len(train_pool)
        (fdir / f"size_{n_pool}.json").write_text(json.dumps(train_pool))
        # Human-readable copy under a distinct name (JSON, not size_*.json —
        # keeps the runner's glob from stumbling over a non-numeric size_).
        (fdir / "train_pool.json").write_text(json.dumps(train_pool))
        fold_summary.append({"fold": f, "test": len(test), "train_pool": len(train_pool)})
        print(f"  fold{f}: test={len(test)}  train_pool={len(train_pool)}")

    # -------------------------------------------------------------------
    # 7) Manifest
    # -------------------------------------------------------------------
    manifest = {
        "cohort_size": len(reaction_ids),
        "family_counts": labels.family.value_counts().to_dict(),
        "feature_dim": feature_dim,
        "label_cols": LABEL_COLS,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "bundles": ["m1", "m2", "m3"],
        "descriptor_dims": {"m1": 6, "m2": 21, "m3": 24},
        "ood_filtering": "NONE (all 799 reactions retained)",
        "fold_summary": fold_summary,
        "paths": {
            "labels": str(LABELS_PARQUET),
            "descriptors": str(DESC_PQ),
            "mace_features": str(FEAT_DIR),
            "bundle_dir": str(BUNDLE_DIR),
            "split_dir": str(SPLIT_DIR),
        },
    }
    (BUNDLE_DIR / "stage4_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[stage4] done. Manifest -> {BUNDLE_DIR / 'stage4_manifest.json'}")


if __name__ == "__main__":
    main()
