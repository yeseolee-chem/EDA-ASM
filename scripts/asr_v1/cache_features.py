"""Precompute frozen NequIP per-atom features for the labeled dipolar set.

One-time job. Output is a .pt blob consumed by training/smoke tests.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from eda_asm.asr_v1.backbone import NequIPFeatureExtractor
from eda_asm.asr_v1.data import (
    ASR_COMPONENTS,
    iter_reaction_pairs,
    load_label_table,
)
from eda_asm.asr_v1.training import CachedFeatureBundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit to first N reactions (for smoke testing)")
    ap.add_argument("--output", default=None,
                    help="override feature_cache path from config")
    ap.add_argument("--device", default=None,
                    help="override backbone device (cuda|cpu)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path.cwd()
    labels_path = repo_root / cfg["labels_parquet"]
    dipolar_root = repo_root / cfg["dipolar_root"]
    out_path = Path(args.output) if args.output else repo_root / cfg["feature_cache"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache_features] labels  : {labels_path}")
    print(f"[cache_features] dipolar : {dipolar_root}")
    print(f"[cache_features] output  : {out_path}")

    df = load_label_table(labels_path, family=cfg["family"])
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"[cache_features] loaded {len(df)} labeled reactions ({cfg['family']})")

    backbone = NequIPFeatureExtractor(
        config_path=repo_root / cfg["backbone"]["config"],
        checkpoint_path=repo_root / cfg["backbone"]["checkpoint"],
        device=args.device,
    )
    print(f"[cache_features] backbone feature_dim = {backbone.feature_dim}")

    reaction_ids: list[str] = []
    R_features: list[torch.Tensor] = []
    P_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    t0 = time.time()
    failed: list[tuple[str, str]] = []
    for i, sample in enumerate(iter_reaction_pairs(df, dipolar_root=dipolar_root,
                                                   skip_errors=False)):
        try:
            R_feat = backbone.extract(sample.R_atoms)
            P_feat = backbone.extract(sample.P_atoms)
        except Exception as exc:                                # noqa: BLE001
            failed.append((sample.reaction_id, repr(exc)))
            continue
        reaction_ids.append(sample.reaction_id)
        R_features.append(R_feat)
        P_features.append(P_feat)
        labels.append(torch.from_numpy(sample.label))
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1:3d}/{len(df)}] {sample.reaction_id} "
                  f"R={tuple(R_feat.shape)} P={tuple(P_feat.shape)}  "
                  f"({dt:.1f}s elapsed)")

    bundle = CachedFeatureBundle(
        reaction_ids=reaction_ids,
        R_features=R_features,
        P_features=P_features,
        labels=torch.stack(labels, dim=0).float(),
        feature_dim=backbone.feature_dim,
    )
    bundle.save(out_path)
    print(f"[cache_features] wrote {len(bundle)} reactions to {out_path}")
    print(f"[cache_features] component order: {list(ASR_COMPONENTS)}")
    if failed:
        print(f"[cache_features] {len(failed)} failures:")
        for rid, msg in failed[:10]:
            print(f"  - {rid}: {msg}")


if __name__ == "__main__":
    main()
