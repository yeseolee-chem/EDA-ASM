"""Cache MACE-OFF (R, TS, P) features + physics descriptors for Δ-learning.

For each labelled dipolar reaction, computes:
  - per-atom MACE-OFF invariant features for R, TS, P (3 tensors)
  - a deterministic 6-vector physics descriptor
    (compute_descriptors from baseline_physics)

Output schema matches ``training_delta.CachedFeatureBundleDelta``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor
from eda_asm.asr_v1.baseline_physics import (
    DESCRIPTOR_NAMES,
    compute_descriptors,
)
from eda_asm.asr_v1.data import (
    ASR_COMPONENTS,
    iter_reaction_triples,
    load_label_table,
)
from eda_asm.asr_v1.training_delta import CachedFeatureBundleDelta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml")
    ap.add_argument("--model-size", default="medium",
                    choices=["small", "medium", "large"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--default-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--num-layers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path.cwd()
    labels_path = repo_root / cfg["labels_parquet"]
    dipolar_root = repo_root / cfg["dipolar_root"]
    default_out = f"outputs/asr_v1/features_dipolar_maceoff_{args.model_size}_delta.pt"
    out_path = Path(args.out) if args.out else repo_root / default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache_delta] labels  : {labels_path}")
    print(f"[cache_delta] dipolar : {dipolar_root}")
    print(f"[cache_delta] output  : {out_path}")
    print(f"[cache_delta] model   : MACE-OFF23 {args.model_size}")

    df = load_label_table(labels_path, family=cfg["family"])
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"[cache_delta] loaded {len(df)} labeled reactions ({cfg['family']})")

    backbone = MACEOFFFeatureExtractor(
        model_size=args.model_size,
        device=args.device,
        default_dtype=args.default_dtype,
        num_layers=args.num_layers,
    )
    print(f"[cache_delta] feature_dim = {backbone.feature_dim}")

    try:
        import mace
        mace_version = getattr(mace, "__version__", "unknown")
    except Exception:                                              # noqa: BLE001
        mace_version = "unknown"

    reaction_ids: list[str] = []
    R_features: list[torch.Tensor] = []
    TS_features: list[torch.Tensor] = []
    P_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    descriptors: list[np.ndarray] = []
    failed: list[tuple[str, str]] = []

    t0 = time.time()
    for i, sample in enumerate(iter_reaction_triples(df, dipolar_root=dipolar_root,
                                                     skip_errors=False)):
        try:
            R_feat = backbone.extract(sample.R_atoms)
            TS_feat = backbone.extract(sample.TS_atoms)
            P_feat = backbone.extract(sample.P_atoms)
            desc = compute_descriptors(sample.R_atoms, sample.TS_atoms,
                                       sample.P_atoms)
        except Exception as exc:                                   # noqa: BLE001
            failed.append((sample.reaction_id, repr(exc)))
            continue
        reaction_ids.append(sample.reaction_id)
        R_features.append(R_feat)
        TS_features.append(TS_feat)
        P_features.append(P_feat)
        labels.append(torch.from_numpy(sample.label))
        descriptors.append(desc)
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1:3d}/{len(df)}] {sample.reaction_id}  "
                  f"desc={tuple(desc.shape)}  ({dt:.1f}s elapsed)")

    bundle = CachedFeatureBundleDelta(
        reaction_ids=reaction_ids,
        R_features=R_features,
        TS_features=TS_features,
        P_features=P_features,
        labels=torch.stack(labels, dim=0).float(),
        descriptors=torch.from_numpy(np.stack(descriptors, axis=0)).float(),
        feature_dim=backbone.feature_dim,
    )
    bundle.save(out_path)
    manifest = {
        "backbone": "mace-off23",
        "model_size": args.model_size,
        "default_dtype": args.default_dtype,
        "num_layers": args.num_layers,
        "device": args.device or "auto",
        "feature_dim": backbone.feature_dim,
        "n_reactions": len(bundle),
        "n_failed": len(failed),
        "mace_torch_version": mace_version,
        "labels_parquet": str(labels_path),
        "family": cfg["family"],
        "elapsed_seconds": time.time() - t0,
        "input_mode": "delta_rtsp",
        "descriptor_names": list(DESCRIPTOR_NAMES),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[cache_delta] wrote {len(bundle)} reactions → {out_path}")
    print(f"[cache_delta] components : {list(ASR_COMPONENTS)}")
    print(f"[cache_delta] descriptors: {list(DESCRIPTOR_NAMES)}")
    if failed:
        print(f"[cache_delta] {len(failed)} failures:")
        for rid, msg in failed[:10]:
            print(f"  - {rid}: {msg}")


if __name__ == "__main__":
    main()
