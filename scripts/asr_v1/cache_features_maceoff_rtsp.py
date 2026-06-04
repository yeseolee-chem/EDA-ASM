"""Precompute MACE-OFF per-atom features for (R, TS, P) — the 3-way pipeline.

Loads each dipolar reaction's R = r0+r1, TS = TS_imag_mode.xyz (DFT-converged
TS with imaginary frequency), P = p0. Calls the same MACE-OFF backbone used
in the 2-way cache, but stores three feature tensors per reaction instead of
two. Output schema matches ``training_rtsp.CachedFeatureBundleRTSP``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor
from eda_asm.asr_v1.data import (
    ASR_COMPONENTS,
    iter_reaction_triples,
    load_label_table,
)
from eda_asm.asr_v1.training_rtsp import CachedFeatureBundleRTSP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml",
                    help="reuses asr_v1.yaml-style cfg for label/data paths")
    ap.add_argument("--model-size", default="medium",
                    choices=["small", "medium", "large"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--default-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--num-layers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="default: outputs/asr_v1/features_dipolar_maceoff_<size>_rtsp.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path.cwd()
    labels_path = repo_root / cfg["labels_parquet"]
    dipolar_root = repo_root / cfg["dipolar_root"]
    default_out = f"outputs/asr_v1/features_dipolar_maceoff_{args.model_size}_rtsp.pt"
    out_path = Path(args.out) if args.out else repo_root / default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache_rtsp] labels   : {labels_path}")
    print(f"[cache_rtsp] dipolar  : {dipolar_root}")
    print(f"[cache_rtsp] output   : {out_path}")
    print(f"[cache_rtsp] model    : MACE-OFF23 {args.model_size}")
    print(f"[cache_rtsp] dtype    : {args.default_dtype}")
    print(f"[cache_rtsp] device   : {args.device or 'auto'}")

    df = load_label_table(labels_path, family=cfg["family"])
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"[cache_rtsp] loaded {len(df)} labeled reactions ({cfg['family']})")

    backbone = MACEOFFFeatureExtractor(
        model_size=args.model_size,
        device=args.device,
        default_dtype=args.default_dtype,
        num_layers=args.num_layers,
    )
    print(f"[cache_rtsp] feature_dim = {backbone.feature_dim}")

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
    failed: list[tuple[str, str]] = []

    t0 = time.time()
    for i, sample in enumerate(iter_reaction_triples(df, dipolar_root=dipolar_root,
                                                     skip_errors=False)):
        try:
            R_feat = backbone.extract(sample.R_atoms)
            TS_feat = backbone.extract(sample.TS_atoms)
            P_feat = backbone.extract(sample.P_atoms)
        except Exception as exc:                                   # noqa: BLE001
            failed.append((sample.reaction_id, repr(exc)))
            continue
        reaction_ids.append(sample.reaction_id)
        R_features.append(R_feat)
        TS_features.append(TS_feat)
        P_features.append(P_feat)
        labels.append(torch.from_numpy(sample.label))
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1:3d}/{len(df)}] {sample.reaction_id} "
                  f"R={tuple(R_feat.shape)} TS={tuple(TS_feat.shape)} "
                  f"P={tuple(P_feat.shape)}  ({dt:.1f}s)")

    bundle = CachedFeatureBundleRTSP(
        reaction_ids=reaction_ids,
        R_features=R_features,
        TS_features=TS_features,
        P_features=P_features,
        labels=torch.stack(labels, dim=0).float(),
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
        "input_mode": "rtsp",
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[cache_rtsp] wrote {len(bundle)} reactions → {out_path}")
    print(f"[cache_rtsp] components: {list(ASR_COMPONENTS)}")
    if failed:
        print(f"[cache_rtsp] {len(failed)} failures:")
        for rid, msg in failed[:10]:
            print(f"  - {rid}: {msg}")


if __name__ == "__main__":
    main()
