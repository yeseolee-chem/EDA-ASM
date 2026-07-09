"""Export TS geometry per reaction as a PDB with fragA/fragB split by chain.

Reads:
  - labels/adf/adf_labels_v6_multifamily.parquet  (reaction_id, family)
  - /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json
  - /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt

Writes:
  outputs/frag_view/{family}/{rid}.pdb
    - chain A  → fragA atoms  (residue FRA, resSeq 1)
    - chain B  → fragB atoms  (residue FRB, resSeq 2)

Open in Avogadro 2 and colour by chain/residue to see the partition.
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
OUT_ROOT = REPO / "outputs/frag_view"


def load_ts(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    z = np.asarray(d["TS"]["z"], dtype=int)
    pos = np.asarray(d["TS"]["pos"], dtype=float)
    return z, pos


def write_pdb(rid: str, family: str, z, pos, idx_a, idx_b, out_path: Path):
    idx_a_set = set(int(i) for i in idx_a)
    idx_b_set = set(int(i) for i in idx_b)

    lines = [
        f"HEADER    FRAGMENT VIEW  {rid}  family={family}",
        f"REMARK    fragA n={len(idx_a)}  fragB n={len(idx_b)}  total={len(z)}",
        f"REMARK    chain A = fragA (FRA)   chain B = fragB (FRB)",
    ]

    serial = 1
    for i in range(len(z)):
        if i in idx_a_set:
            chain, resname, resseq = "A", "FRA", 1
        elif i in idx_b_set:
            chain, resname, resseq = "B", "FRB", 2
        else:
            # Unassigned atom (shouldn't happen after partition, but be safe)
            chain, resname, resseq = "X", "UNK", 3
        sym = chemical_symbols[int(z[i])]
        # Right-justify element in 4-char atom name field, then space-align.
        name = f"{sym:>2s}{i:<2d}"[:4]
        x, y, zc = pos[i]
        lines.append(
            f"HETATM{serial:>5d} {name:<4s} {resname:<3s} {chain}{resseq:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{zc:>8.3f}  1.00  0.00          {sym:>2s}"
        )
        serial += 1
    lines.append("END")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
    labels = pd.read_parquet(LABELS_PQ)
    with open(PART_JSON) as f:
        partitions = json.load(f)

    n_ok = n_skip = n_err = 0
    for row in labels.itertuples(index=False):
        rid, fam = row.reaction_id, row.family
        out = OUT_ROOT / fam / f"{rid}.pdb"
        if out.exists():
            n_skip += 1
            continue
        try:
            part = partitions[rid]
            if "error" in part:
                raise RuntimeError(part["error"])
            z, pos = load_ts(rid)
            write_pdb(rid, fam, z, pos, part["frag_A_indices"], part["frag_B_indices"], out)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {rid}: {exc}", flush=True)

    print(f"done: ok={n_ok}  skipped={n_skip}  errors={n_err}")
    print(f"output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
