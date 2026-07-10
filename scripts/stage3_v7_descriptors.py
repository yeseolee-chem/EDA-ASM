"""Stage-3 for v7 cohort: compute descriptors_v7.parquet (d1..d24) for
all 776 reactions in orca_eda_labels_v7.parquet.

Uses:
  - Cohort:      labels/orca/orca_eda_labels_v7.parquet
  - Partitions:  outputs/final_776_v7/fragmentation/orca_inp_partitions.json
  - Charges:     labels/orca/orca_eda_charges_v7.parquet   (from extract_v7_charges.py)
  - Geometries:  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/<rid>.pt

Output:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v7.parquet

Idempotent per-shard: writes shard parquet, skips rxns already computed.
"""
from __future__ import annotations

# tblite before torch (libgomp GOMP_5.0)
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pipeline_rebuild" / "spec_v1"))

from eda_asm.asr_v1.baseline_physics import compute_descriptors
# Reuse compute_xtb_block + compute_geom6 from spec_v1 stage3 (identical logic)
from stage3_xtb_and_descriptors import compute_geom6, compute_xtb_block  # noqa: E402

LABELS_V7    = REPO / "labels/orca/orca_eda_labels_v7.parquet"
PART_V7      = REPO / "outputs/final_776_v7/fragmentation/orca_inp_partitions.json"
CHARGES_V7   = REPO / "labels/orca/orca_eda_charges_v7.parquet"
FEAT_DIR     = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
CHUNK_DIR    = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v7_chunks")
OUT_PARQUET  = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v7.parquet")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_COLS = [f"d{i}" for i in range(1, 25)]


def load_atoms(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    R  = ase.Atoms(numbers=d["R"]["z"],  positions=np.asarray(d["R"]["pos"]))
    TS = ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"]))
    P  = ase.Atoms(numbers=d["P"]["z"],  positions=np.asarray(d["P"]["pos"]))
    return R, TS, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    labels = pd.read_parquet(LABELS_V7)
    partitions = json.loads(PART_V7.read_text())
    charges = pd.read_parquet(CHARGES_V7).set_index("reaction_id")

    labels = labels.reset_index(drop=True)
    labels = labels[labels.index % args.nshards == args.shard].reset_index(drop=True)
    print(f"shard {args.shard}/{args.nshards} -> {len(labels)} rxns", flush=True)

    out_pq = CHUNK_DIR / f"shard_{args.shard:03d}_of_{args.nshards}.parquet"
    if out_pq.exists():
        prev = pd.read_parquet(out_pq)
        if "error" in prev.columns:
            prev = prev[prev["error"].isna()]
        have = set(prev["reaction_id"])
    else:
        prev = pd.DataFrame()
        have = set()

    rows = []
    t0 = time.time()
    n_ok = n_fail = 0
    for i, row in labels.iterrows():
        rid, fam = row.reaction_id, row.family
        if rid in have:
            n_ok += 1
            continue
        _t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] i={i} rid={rid}", flush=True)
        try:
            R_at, TS_at, P_at = load_atoms(rid)
            geom6 = compute_geom6(R_at, TS_at, P_at)

            part = partitions.get(rid)
            if part is None or "error" in part:
                raise RuntimeError(f"missing/error partition: {part}")
            frag_A = part["frag_A_indices"]
            frag_B = part["frag_B_indices"]
            if not frag_A or not frag_B:
                raise RuntimeError("empty fragment")
            if set(frag_A) & set(frag_B):
                raise RuntimeError("frag A/B overlap")

            if rid not in charges.index:
                raise RuntimeError("no charge row")
            ch = charges.loc[rid]
            xtb = compute_xtb_block(
                TS_at, frag_A, frag_B,
                total_charge=int(ch["total_charge"]),
                charge_a=int(ch["fragment_charge_a"]),
                charge_b=int(ch["fragment_charge_b"]),
                mult_a=int(ch["fragment_mult_a"]),
                mult_b=int(ch["fragment_mult_b"]),
            )
            row_out = {"reaction_id": rid, "family": fam}
            for k, v in enumerate(geom6):
                row_out[f"d{k+1}"] = float(v)
            row_out.update(xtb)
            rows.append(row_out)
            n_ok += 1
            print(f"  ok ({time.time()-_t:.1f}s)", flush=True)
        except Exception as e:
            rows.append({"reaction_id": rid, "family": fam,
                         "error": f"{type(e).__name__}: {e}"})
            n_fail += 1
            print(f"  FAIL {type(e).__name__}: {e} ({time.time()-_t:.1f}s)", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] progress {i+1}/{len(labels)} "
                  f"ok={n_ok} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)

    df_new = pd.DataFrame(rows)
    df_out = pd.concat([prev, df_new], ignore_index=True) if not prev.empty else df_new
    df_out.to_parquet(out_pq)
    print(f"wrote {out_pq} ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
