"""Export TS geometry per reaction as TWO separate XYZ files (fragA + fragB).

Avogadro 2 assigns a distinct colour to each loaded molecule, so opening
both XYZ files in the same window makes the fragment split obvious.

Writes:
  outputs/frag_view_xyz/{family}/{rid}_fragA.xyz
  outputs/frag_view_xyz/{family}/{rid}_fragB.xyz
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
PART_JSON = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json")
LABELS_PQ = REPO / "labels/adf/adf_labels_v6_multifamily.parquet"
OUT_ROOT = REPO / "outputs/frag_view_xyz"


def load_ts(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    z = np.asarray(d["TS"]["z"], dtype=int)
    pos = np.asarray(d["TS"]["pos"], dtype=float)
    return z, pos


def write_xyz(path: Path, z, pos, indices, comment: str):
    idx = list(indices)
    lines = [str(len(idx)), comment]
    for i in idx:
        sym = chemical_symbols[int(z[i])]
        x, y, zc = pos[i]
        lines.append(f"{sym:<3s} {x:>15.8f} {y:>15.8f} {zc:>15.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    labels = pd.read_parquet(LABELS_PQ)
    with open(PART_JSON) as f:
        partitions = json.load(f)

    n_ok = n_skip = n_err = 0
    for row in labels.itertuples(index=False):
        rid, fam = row.reaction_id, row.family
        out_a = OUT_ROOT / fam / f"{rid}_fragA.xyz"
        out_b = OUT_ROOT / fam / f"{rid}_fragB.xyz"
        if out_a.exists() and out_b.exists():
            n_skip += 1
            continue
        try:
            part = partitions[rid]
            if "error" in part:
                raise RuntimeError(part["error"])
            z, pos = load_ts(rid)
            idx_a = part["frag_A_indices"]
            idx_b = part["frag_B_indices"]
            write_xyz(out_a, z, pos, idx_a,
                      f"{rid} fragA n={len(idx_a)} family={fam}")
            write_xyz(out_b, z, pos, idx_b,
                      f"{rid} fragB n={len(idx_b)} family={fam}")
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {rid}: {exc}", flush=True)

    print(f"done: ok={n_ok}  skipped={n_skip}  errors={n_err}")
    print(f"output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
