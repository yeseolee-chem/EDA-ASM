"""Write minimal per-reaction .pt geometry files for the 143 replacement reactions
so the review app + ORCA input generator can access their TS coordinates.

Input:  outputs/frag_review/replacements_need_features.json
Output: /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt
         with dict-of-dict format: {"TS": {"z": tensor, "pos": tensor}}

NOTE: This does NOT compute real MACE-OFF23 embeddings. It only writes the TS
z + pos, which is all the review app and ORCA EDA input generator read. If m2
/m3 xTB descriptors are ever needed for these new IDs, run the real MACE
extractor (pipeline_rebuild/spec_v1/stage2_mace_features.py) on them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
REPLACE_JSON = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/frag_review/replacements_need_features.json")


def main():
    with open(REPLACE_JSON) as f:
        replacements = json.load(f)

    n_ok = n_skip = n_updated = 0
    for entry in replacements:
        rid = entry["reaction_id"]
        out = FEAT_DIR / f"{rid}.pt"
        # Which geometry is the "primary" one (varies by refinement basis)
        prim_geom = entry.get("geom_used", "TS")
        other_geom = entry.get("other_geom")
        z = np.asarray(entry["z"], dtype=int)
        pos = np.asarray(entry["pos"], dtype=float)

        payload = {prim_geom: {"z": z, "pos": pos}}
        if other_geom and "other_z" in entry:
            payload[other_geom] = {
                "z": np.asarray(entry["other_z"], dtype=int),
                "pos": np.asarray(entry["other_pos"], dtype=float),
            }

        if out.exists():
            existing = torch.load(str(out), map_location="cpu", weights_only=False)
            # Merge: keep any pre-existing geom entries; overwrite with new payload.
            merged = dict(existing)
            for k, v in payload.items():
                merged[k] = v
            torch.save(merged, str(out))
            n_updated += 1
        else:
            torch.save(payload, str(out))
            n_ok += 1

    print(f"wrote {n_ok} new .pt files, updated {n_updated} existing, skipped {n_skip}")


if __name__ == "__main__":
    main()
