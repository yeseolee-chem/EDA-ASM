"""Build v2 Δ-learning bundles by swapping the `descriptors` tensor.

The existing training pipeline (src/eda_asm/asr_v1/training_delta.py) reads
the Δ-baseline features from `bundle.descriptors`. To compare baselines we
take the canonical v6 bundle:

    outputs/asr_v1/features_v6_maceoff_medium_delta.pt   (789, descriptors = geom6)

and emit three v2 variants under analysis/exp_6arm_redesign_v2/bundles/:

    features_v6_delta_geom6.pt       descriptors = geom6 (6-d)        [verbatim copy]
    features_v6_delta_xtb.pt         descriptors = xtb features (19-d)
    features_v6_delta_xtb_geom6.pt   descriptors = concat(geom6, xtb) (25-d)

No existing code is modified. The runner picks up the v2 bundle via the
`FEATURES_DELTA` env var that phase3_trackA_delta_runner.py already supports.

Each bundle also gets a sibling `<bundle>.families.json` (rid → family map)
copied from the existing canonical bundle so the loader works as-is.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC_BUNDLE = REPO / "outputs/asr_v1/features_v6_maceoff_medium_delta.pt"
SRC_FAM_JSON = REPO / "outputs/asr_v1/features_v6_maceoff_medium_delta.families.json"
XTB_CACHE = HERE / "xtb_cache/xtb_features.parquet"
OUT_DIR = HERE / "bundles"

# Must match XTB_FEATURE_NAMES in src/baselines.py.
XTB_FEATURE_NAMES = [
    "E_int_kcal", "E_complex_kcal", "E_fragA_kcal", "E_fragB_kcal",
    "dipole_complex_norm", "dipole_fragA_norm", "dipole_fragB_norm",
    "dipole_int",
    "HOMO_complex", "LUMO_complex", "gap_complex",
    "HOMO_fragA", "LUMO_fragA", "gap_fragA",
    "HOMO_fragB", "LUMO_fragB", "gap_fragB",
    "sum_q_A_frag_atoms", "n_atoms",
]


def main() -> None:
    if not SRC_BUNDLE.exists():
        sys.exit(f"[FATAL] canonical bundle missing: {SRC_BUNDLE}")
    if not XTB_CACHE.exists():
        sys.exit(f"[FATAL] xtb cache missing: {XTB_CACHE}; run build_xtb_cache.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bundle] loading canonical bundle: {SRC_BUNDLE}")
    bundle = torch.load(SRC_BUNDLE, map_location="cpu", weights_only=False)
    N = len(bundle["reaction_ids"])
    print(f"[bundle] N={N}, feature_dim={bundle['feature_dim']}")
    print(f"[bundle] existing descriptors shape: {bundle['descriptors'].shape}")

    rids = bundle["reaction_ids"]
    rid_to_idx = {r: i for i, r in enumerate(rids)}

    xtb_df = pd.read_parquet(XTB_CACHE).set_index("rid")
    missing = [r for r in rids if r not in xtb_df.index]
    if missing:
        print(f"[bundle] WARN: {len(missing)} bundle rids missing from xtb cache; "
              "those rows get NaN-filled features (will be column-mean-imputed at ridge time).",
              file=sys.stderr)

    # Build xtb feature matrix in bundle order.
    xtb_mat = np.full((N, len(XTB_FEATURE_NAMES)), np.nan, dtype=np.float32)
    for i, rid in enumerate(rids):
        if rid in xtb_df.index:
            row = xtb_df.loc[rid]
            xtb_mat[i] = np.array([row.get(k, np.nan) for k in XTB_FEATURE_NAMES], dtype=np.float32)
    # NaN-impute with column means (so torch tensor is finite for ridge / NN delta).
    col_mean = np.nanmean(xtb_mat, axis=0)
    nan_mask = np.isnan(xtb_mat)
    xtb_mat[nan_mask] = np.broadcast_to(col_mean, xtb_mat.shape)[nan_mask]
    # Standardise (zero-mean, unit-std) so xtb features are on a similar
    # numeric scale to the geom6 ones; the inner ridge re-standardises but
    # this protects the NN delta head's input std stats.
    xtb_std = xtb_mat - xtb_mat.mean(axis=0, keepdims=True)
    sigma = xtb_std.std(axis=0, keepdims=True) + 1e-6
    xtb_std = xtb_std / sigma

    geom6 = bundle["descriptors"].numpy().astype(np.float32)
    print(f"[bundle] geom6 shape={geom6.shape}, xtb shape={xtb_std.shape}")

    # Variants
    variants = {
        "geom6":     geom6,
        "xtb":       xtb_std,
        "xtb_geom6": np.concatenate([geom6, xtb_std], axis=1),
    }

    for name, desc in variants.items():
        out_pt = OUT_DIR / f"features_v6_delta_{name}.pt"
        out_obj = dict(bundle)
        out_obj["descriptors"] = torch.from_numpy(desc.astype(np.float32))
        torch.save(out_obj, out_pt)
        # Sidecar families.json
        out_fam = OUT_DIR / f"features_v6_delta_{name}.families.json"
        if SRC_FAM_JSON.exists() and not out_fam.exists():
            shutil.copy(SRC_FAM_JSON, out_fam)
        print(f"[bundle] wrote {out_pt}  (descriptors {desc.shape})  + families.json")

    print("[bundle] done.")


if __name__ == "__main__":
    main()
