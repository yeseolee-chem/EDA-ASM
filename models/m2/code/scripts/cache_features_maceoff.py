"""Precompute frozen MACE-OFF per-atom features for the labelled dipolar set.

Drop-in replacement for ``cache_features.py`` (the ep29 NequIP cache) — same
input list of reactions, same output schema (``CachedFeatureBundle``), but the
backbone is the pretrained MACE-OFF23 model instead of the 29-epoch NequIP
checkpoint. This is the cache used by the backbone-comparison spec
(ASR_Backbone_Comparison_Spec_v1.0).

Output: a ``.pt`` blob compatible with ``training.CachedFeatureBundle.load``.
Fold ids / reaction order are determined by the labels parquet and the data
loader — identical across NequIP and MACE-OFF caches.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor
from eda_asm.asr_v1.data import (
    ASR_COMPONENTS,
    iter_reaction_pairs,
    load_label_table,
)
from eda_asm.asr_v1.training import CachedFeatureBundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml",
                    help="reuses asr_v1.yaml for label/data paths; the "
                         "backbone block is ignored")
    ap.add_argument("--model-size", default="medium",
                    choices=["small", "medium", "large"],
                    help="MACE-OFF23 model size (default: medium)")
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--default-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--num-layers", type=int, default=-1,
                    help="get_descriptors num_layers; -1 = all (default)")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit to first N reactions (smoke test)")
    ap.add_argument("--out", default=None,
                    help="output .pt path (default: outputs/asr_v1/"
                         "features_dipolar_maceoff_<size>.pt)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path.cwd()
    labels_path = repo_root / cfg["labels_parquet"]
    dipolar_root = repo_root / cfg["dipolar_root"]

    default_out = f"outputs/asr_v1/features_dipolar_maceoff_{args.model_size}.pt"
    out_path = Path(args.out) if args.out else repo_root / default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache_features_maceoff] labels   : {labels_path}")
    print(f"[cache_features_maceoff] dipolar  : {dipolar_root}")
    print(f"[cache_features_maceoff] output   : {out_path}")
    print(f"[cache_features_maceoff] model    : MACE-OFF23 {args.model_size}")
    print(f"[cache_features_maceoff] dtype    : {args.default_dtype}")
    print(f"[cache_features_maceoff] device   : {args.device or 'auto'}")
    print(f"[cache_features_maceoff] layers   : {args.num_layers} "
          f"(-1 = all interaction blocks)")

    df = load_label_table(labels_path, family=cfg["family"])
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"[cache_features_maceoff] loaded {len(df)} labeled reactions "
          f"({cfg['family']})")

    backbone = MACEOFFFeatureExtractor(
        model_size=args.model_size,
        device=args.device,
        default_dtype=args.default_dtype,
        num_layers=args.num_layers,
    )
    print(f"[cache_features_maceoff] feature_dim = {backbone.feature_dim}")

    # Record mace-torch version for the run manifest.
    try:
        import mace
        mace_version = getattr(mace, "__version__", "unknown")
    except Exception:                                              # noqa: BLE001
        mace_version = "unknown"

    reaction_ids: list[str] = []
    R_features: list[torch.Tensor] = []
    P_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    failed: list[tuple[str, str]] = []

    t0 = time.time()
    for i, sample in enumerate(iter_reaction_pairs(df, dipolar_root=dipolar_root,
                                                   skip_errors=False)):
        try:
            R_feat = backbone.extract(sample.R_atoms)
            P_feat = backbone.extract(sample.P_atoms)
        except Exception as exc:                                   # noqa: BLE001
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
    # Save bundle + a sidecar manifest so the comparison report can pick up
    # backbone provenance without re-loading the .pt.
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
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    import json
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[cache_features_maceoff] wrote {len(bundle)} reactions → {out_path}")
    print(f"[cache_features_maceoff] manifest → {manifest_path}")
    print(f"[cache_features_maceoff] component order: {list(ASR_COMPONENTS)}")
    if failed:
        print(f"[cache_features_maceoff] {len(failed)} failures:")
        for rid, msg in failed[:10]:
            print(f"  - {rid}: {msg}")


if __name__ == "__main__":
    main()
