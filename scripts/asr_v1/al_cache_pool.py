"""Cache MACE-OFF (R, TS, P) features + 6-descriptor physics vector for the
entire UNLABELED candidate pool, across BOTH dipolar and qmrxn20 (e2/sn2).

Source of truth = the seed CSV under ADF_250/seed_selection/initial_seed_v1/.
A reaction is included in the pool if (a) its row exists in the seed CSV and
(b) its reaction_id is NOT in the current labels parquet.

Output: outputs/asr_v1/al/pool_features.pt + .manifest.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor
from eda_asm.asr_v1.baseline_physics import DESCRIPTOR_NAMES, compute_descriptors
from eda_asm.asr_v1.data_multi import (
    iter_seed_rows,
    load_seed_csv,
    normalize_family,
)
from eda_asm.asr_v1.training_delta import CachedFeatureBundleDelta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1_maceoff_delta_n250.yaml")
    ap.add_argument("--seed-csv",
                    default="ADF_250/seed_selection/initial_seed_v1/selected_reactions.csv")
    ap.add_argument("--families", nargs="+", default=["dipolar", "e2", "sn2"])
    ap.add_argument("--model-size", default="medium",
                    choices=["small", "medium", "large"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--default-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--num-layers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/asr_v1/al/pool_features.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo = Path.cwd()
    labels_path = repo / cfg["labels_parquet"]
    seed_path = repo / args.seed_csv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_parquet(labels_path)
    # Build the set of already-labeled reaction_ids across all families.
    labeled_ids = set(labels_df["reaction_id"])
    print(f"[pool] labels parquet: {labels_path}  ({len(labels_df)} rows, "
          f"families={labels_df['family'].value_counts().to_dict()})")

    seed = load_seed_csv(seed_path)
    print(f"[pool] seed CSV: {len(seed)} rows  "
          f"families={seed['family'].value_counts().to_dict()}")

    # Filter to selected families AND unlabeled.
    seed = seed[seed["family"].isin(args.families)].copy()
    seed = seed[~seed["reaction_id"].isin(labeled_ids)].copy().reset_index(drop=True)
    if args.limit:
        seed = seed.head(args.limit).copy()
    print(f"[pool] candidate pool (unlabeled): {len(seed)} reactions  "
          f"by family: {seed['family'].value_counts().to_dict()}")

    backbone = MACEOFFFeatureExtractor(
        model_size=args.model_size, device=args.device,
        default_dtype=args.default_dtype, num_layers=args.num_layers,
    )
    print(f"[pool] feature_dim: {backbone.feature_dim}")

    reaction_ids: list[str] = []
    families: list[str] = []
    R_feats: list[torch.Tensor] = []
    TS_feats: list[torch.Tensor] = []
    P_feats: list[torch.Tensor] = []
    descriptors: list[np.ndarray] = []
    failed: list[tuple[str, str]] = []

    t0 = time.time()
    n_total = len(seed)
    for i, sample in enumerate(iter_seed_rows(seed, labels_df=None)):
        try:
            R = sample.R_atoms; TS = sample.TS_atoms; P = sample.P_atoms
            R_f = backbone.extract(R)
            TS_f = backbone.extract(TS)
            P_f = backbone.extract(P)
            desc = compute_descriptors(R, TS, P)
        except Exception as exc:                                   # noqa: BLE001
            failed.append((sample.reaction_id, repr(exc)))
            continue
        reaction_ids.append(sample.reaction_id)
        families.append(sample.family)
        R_feats.append(R_f)
        TS_feats.append(TS_f)
        P_feats.append(P_f)
        descriptors.append(desc)
        if (i + 1) % 50 == 0 or i == 0:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            print(f"  [{i+1:4d}/{n_total}] {sample.reaction_id} "
                  f"({sample.family})  R={tuple(R_f.shape)}  ({dt:.0f}s, {rate:.2f}/s)")

    bundle = CachedFeatureBundleDelta(
        reaction_ids=reaction_ids,
        R_features=R_feats,
        TS_features=TS_feats,
        P_features=P_feats,
        labels=torch.zeros((len(reaction_ids), 5), dtype=torch.float32),
        descriptors=torch.from_numpy(np.stack(descriptors, axis=0)).float(),
        feature_dim=backbone.feature_dim,
    )
    bundle.save(out_path)

    # Sidecar: families per pool index (needed for ADF input generation later).
    families_path = out_path.with_suffix(".families.json")
    families_path.write_text(json.dumps({
        "reaction_ids": reaction_ids,
        "families": families,
    }))

    manifest = {
        "kind": "al_pool",
        "backbone": "mace-off23",
        "model_size": args.model_size,
        "feature_dim": backbone.feature_dim,
        "n_reactions": len(bundle),
        "n_failed": len(failed),
        "labels_parquet_at_cache": str(labels_path),
        "n_labeled_at_cache": int(len(labeled_ids)),
        "families_included": args.families,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "elapsed_seconds": time.time() - t0,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[pool] wrote {len(bundle)} pool reactions → {out_path}")
    if failed:
        print(f"[pool] {len(failed)} failures (first 5):")
        for rid, msg in failed[:5]:
            print(f"  - {rid}: {msg}")


if __name__ == "__main__":
    main()
