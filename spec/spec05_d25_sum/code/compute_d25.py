"""SPEC_05 T1 - compute d25 (fragment strain @ reactant-ref).

d25 = ((E_A_TS - E_A_R) + (E_B_TS - E_B_R)) * kappa   [kcal/mol]

- E_frag@TS: already available in d9, d10 of descriptors_v9.parquet (kcal/mol)
- E_frag@R:  NEW GFN2-xTB single-point on fragment atoms with R geometry.

Reuses _run_xtb from pipeline_rebuild/spec_v1/stage3_xtb_and_descriptors.py.
Reads:
  - v9 labels for cohort (783 rxns)
  - v9 orca_inp_partitions.json for frag_A/B indices
  - v9 orca_eda_charges_v9.parquet for charge/mult
  - MACE .pt for R/TS positions + atomic numbers

Output: spec/spec05_d25_sum/data/descriptors_d25_refR.parquet
(idempotent per shard: --shard/--nshards)
"""
from __future__ import annotations
import os

# tblite import must precede torch (libgomp GOMP_5.0 issue)
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import argparse, json, sys, time
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "pipeline_rebuild" / "spec_v1"))
from stage3_xtb_and_descriptors import _run_xtb  # noqa: E402

LABELS_V7   = REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
PART_V7     = REPO / "outputs/v8_review/manual_partitions.json"
CHARGES_V7  = REPO / "labels/orca/orca_eda_charges_v9.parquet"
DESC_V7     = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v9.parquet")
FEAT_DIR    = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")
CHUNK_DIR   = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d25_refR_v9_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KCAL = 627.5094740631


def load_atoms(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), weights_only=False, map_location="cpu")
    R  = ase.Atoms(numbers=d["R"]["z"],  positions=np.asarray(d["R"]["pos"]))
    TS = ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"]))
    return R, TS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    labels = pd.read_parquet(LABELS_V7)
    partitions = json.loads(PART_V7.read_text())
    charges = pd.read_parquet(CHARGES_V7).set_index("reaction_id")
    desc_v9 = pd.read_parquet(DESC_V7).set_index("reaction_id")

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
        prev = pd.DataFrame(); have = set()

    rows = []; t0 = time.time(); n_ok = n_fail = 0
    for i, row in labels.iterrows():
        rid, fam = row.reaction_id, row.family
        if rid in have:
            n_ok += 1; continue
        _t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] i={i} rid={rid}", flush=True)
        try:
            part = partitions.get(rid)
            if part is None or "error" in part:
                raise RuntimeError("no partition")
            frag_A = np.array(part["frag_A_indices"], dtype=int)
            frag_B = np.array(part["frag_B_indices"], dtype=int)
            if rid not in charges.index:
                raise RuntimeError("no charge row")
            ch = charges.loc[rid]
            q_A, m_A = int(ch["fragment_charge_a"]), int(ch["fragment_mult_a"])
            q_B, m_B = int(ch["fragment_charge_b"]), int(ch["fragment_mult_b"])
            R_at, TS_at = load_atoms(rid)
            Z_R = np.array(R_at.get_atomic_numbers()); Z_TS = np.array(TS_at.get_atomic_numbers())
            if not np.array_equal(Z_R, Z_TS):
                # qmrxn20 P sometimes differs, but R vs TS should match (per CLAUDE.md).
                # If mismatch, still try; will likely fail with size mismatch.
                pass
            pos_R = R_at.get_positions(); pos_TS = TS_at.get_positions()

            # E(fragA @ TS) already in desc_v9.d9 (kcal/mol) but we want Hartree; recompute for consistency
            rA_TS = _run_xtb(Z_TS[frag_A], pos_TS[frag_A], charge=q_A, mult=m_A)
            rA_R  = _run_xtb(Z_R[frag_A],  pos_R[frag_A],  charge=q_A, mult=m_A)
            rB_TS = _run_xtb(Z_TS[frag_B], pos_TS[frag_B], charge=q_B, mult=m_B)
            rB_R  = _run_xtb(Z_R[frag_B],  pos_R[frag_B],  charge=q_B, mult=m_B)
            E_A_TS = rA_TS["E_h"]; E_A_R = rA_R["E_h"]
            E_B_TS = rB_TS["E_h"]; E_B_R = rB_R["E_h"]
            d25 = ((E_A_TS - E_A_R) + (E_B_TS - E_B_R)) * HARTREE_TO_KCAL
            rows.append({
                "reaction_id": rid, "family": fam,
                "d25": float(d25),
                "E_A_TS_Eh": float(E_A_TS), "E_A_R_Eh": float(E_A_R),
                "E_B_TS_Eh": float(E_B_TS), "E_B_R_Eh": float(E_B_R),
                "scf_ok": True,
            })
            n_ok += 1
            print(f"  ok d25={d25:+.2f} kcal/mol ({time.time()-_t:.1f}s)", flush=True)
        except Exception as e:
            rows.append({"reaction_id": rid, "family": fam,
                         "error": f"{type(e).__name__}: {e}",
                         "d25": float("nan"), "scf_ok": False})
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
