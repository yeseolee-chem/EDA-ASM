"""Stage 3 (spec v1) — build the full descriptor tensor d1..d24 per reaction.

d1..d6  : geom6 (Kabsch RMSD × 2, Pauli surrogate, elst surrogate, disp, size)
          — computed exactly as in the PDF spec.
d7..d21 : GFN2-xTB single-points on {complex, fragA, fragB} at TS geom.
d22..d24: derived from complex xTB output (Parr ω, Σq², inter-frag Σ|WBO|).

Requires:
  - /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt
     (already exists — provides TS positions + atomic numbers per reaction)
  - /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json
     (from fragment_partition.py)
  - labels/orca/orca_eda_labels.parquet
     (for fragment_charge_a/b, fragment_mult_a/b, total_charge)

Outputs:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1.parquet
     (one row per reaction, 24 float columns + reaction_id + family)

Idempotent; recomputes only reactions missing from the output.
"""
from __future__ import annotations

# IMPORTANT: tblite must be imported BEFORE torch because torch ships a
# libgomp that lacks the GOMP_5.0 symbol required by tblite. Loading torch
# first hides the system libgomp behind torch's copy and breaks tblite.
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import json
import sys
import time
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
import torch
from ase.data import vdw_radii

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
from eda_asm.asr_v1.baseline_physics import compute_descriptors

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
PARTITIONS_JSON = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json")
OUT_PARQUET = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1.parquet")
CHUNK_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KCAL = 627.509474

DESCRIPTOR_COLS = [f"d{i}" for i in range(1, 25)]


def _run_xtb(numbers: np.ndarray, positions: np.ndarray,
             charge: int, mult: int, want_bond_orders: bool = False) -> dict:
    Calculator = _TbliteCalculator
    n_unpaired = mult - 1
    calc = Calculator("GFN2-xTB", numbers, positions / 0.5291772108,  # Å → Bohr
                      charge=float(charge), uhf=n_unpaired)
    calc.set("verbosity", 0)
    calc.set("max-iter", 500)
    calc.set("mixer-damping", 0.2)
    calc.set("accuracy", 0.1)
    # Note: tblite's `add()` only supports interactions (electric-field,
    # solvation, ...). Bond orders and dipole are already returned by
    # singlepoint() by default, so no extra setup needed.
    res = calc.singlepoint()
    E_h = float(res.get("energy"))
    orb_E = np.asarray(res.get("orbital-energies")).ravel()
    orb_occ = np.asarray(res.get("orbital-occupations")).ravel()
    occ_thr = 1.0
    is_occ = orb_occ > occ_thr
    homo_idx = int(np.max(np.where(is_occ)[0])) if is_occ.any() else 0
    lumo_idx = int(homo_idx + 1) if (homo_idx + 1) < len(orb_E) else homo_idx
    HOMO = float(orb_E[homo_idx])
    LUMO = float(orb_E[lumo_idx])
    dipole = np.asarray(res.get("dipole")).ravel()
    charges = np.asarray(res.get("charges")).ravel()
    out = dict(
        E_h=E_h, HOMO_h=HOMO, LUMO_h=LUMO,
        dipole_norm=float(np.linalg.norm(dipole)),
        charges=charges,
    )
    if want_bond_orders:
        bo = np.asarray(res.get("bond-orders"))
        if bo.ndim == 3:
            bo = bo.sum(axis=-1)
        out["bond_orders"] = bo
    return out


def compute_geom6(R_atoms, TS_atoms, P_atoms) -> np.ndarray:
    """Compute d1..d6 with qmrxn20 P-fallback for d2 already handled at load."""
    if len(P_atoms) != len(TS_atoms):
        # Use TS as P surrogate — spec-compliant fallback documented in
        # pipeline_rebuild spec v1 README (qmrxn20 e2/sn2 lose LG atoms in P).
        P_atoms = TS_atoms
    return compute_descriptors(R_atoms, TS_atoms, P_atoms)  # (6,) float32


def compute_xtb_block(TS_atoms, frag_A_idx, frag_B_idx,
                      total_charge, charge_a, charge_b, mult_a, mult_b) -> dict:
    """Returns d7..d24 plus intermediate rc/ra/rb outputs for diagnostics."""
    numbers = np.array(TS_atoms.get_atomic_numbers())
    positions = TS_atoms.get_positions()

    # 1) Complex — need bond orders for d24
    rc = _run_xtb(numbers, positions, charge=total_charge, mult=1,
                  want_bond_orders=True)
    # 2) fragA at TS positions
    idx_a = np.array(frag_A_idx, dtype=int)
    ra = _run_xtb(numbers[idx_a], positions[idx_a],
                  charge=charge_a, mult=mult_a)
    # 3) fragB at TS positions
    idx_b = np.array(frag_B_idx, dtype=int)
    rb = _run_xtb(numbers[idx_b], positions[idx_b],
                  charge=charge_b, mult=mult_b)

    E_C = rc["E_h"]; E_A = ra["E_h"]; E_B = rb["E_h"]
    d7 = (E_C - E_A - E_B) * HARTREE_TO_KCAL
    d8 = E_C * HARTREE_TO_KCAL
    d9 = E_A * HARTREE_TO_KCAL
    d10 = E_B * HARTREE_TO_KCAL
    d11 = rc["dipole_norm"]
    d12 = ra["dipole_norm"]
    d13 = rb["dipole_norm"]
    d14 = rc["dipole_norm"] - ra["dipole_norm"] - rb["dipole_norm"]
    d15, d16 = rc["HOMO_h"], rc["LUMO_h"]
    d17, d18 = ra["HOMO_h"], ra["LUMO_h"]
    d19, d20 = rb["HOMO_h"], rb["LUMO_h"]
    q_C = rc["charges"]
    d21 = float(q_C[idx_a].sum())

    # d22 = Parr electrophilicity ω = μ² / (2η), μ = (HOMO+LUMO)/2, η = gap/2
    mu = 0.5 * (rc["HOMO_h"] + rc["LUMO_h"])
    eta = 0.5 * max(rc["LUMO_h"] - rc["HOMO_h"], 1e-6)
    d22 = float((mu ** 2) / (2 * eta))
    d23 = float(np.sum(q_C ** 2))

    # d24 = Σ_{a∈A, b∈B} |WBO_ab|
    bo = rc["bond_orders"]  # (n, n)
    d24 = float(np.abs(bo[np.ix_(idx_a, idx_b)]).sum())

    return dict(
        d7=d7, d8=d8, d9=d9, d10=d10,
        d11=d11, d12=d12, d13=d13, d14=d14,
        d15=d15, d16=d16, d17=d17, d18=d18, d19=d19, d20=d20,
        d21=d21, d22=d22, d23=d23, d24=d24,
    )


def load_ts_atoms(rid: str):
    """Load TS Atoms from the cached MACE feature file (contains numbers+positions)."""
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    return (
        ase.Atoms(numbers=d["R"]["z"], positions=np.asarray(d["R"]["pos"])),
        ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"])),
        ase.Atoms(numbers=d["P"]["z"], positions=np.asarray(d["P"]["pos"])),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    labels = pd.read_parquet(REPO / "labels/adf/adf_labels_v6_multifamily.parquet")
    orca = pd.read_parquet(REPO / "labels/orca/orca_eda_labels.parquet")
    orca = orca.set_index("reaction_id")
    with open(PARTITIONS_JSON) as f:
        partitions = json.load(f)

    # Shard by index modulo (round-robin so family balance is preserved per shard)
    labels = labels.reset_index(drop=True)
    labels = labels[labels.index % args.nshards == args.shard].reset_index(drop=True)
    print(f"shard {args.shard}/{args.nshards} → {len(labels)} reactions", flush=True)

    out_pq = CHUNK_DIR / f"shard_{args.shard:03d}_of_{args.nshards}.parquet"
    if out_pq.exists():
        prev = pd.read_parquet(out_pq)
        # Keep only successful entries — retry failures with new settings.
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
        _t_rxn = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] i={i} rid={rid}", flush=True)
        try:
            R_at, TS_at, P_at = load_ts_atoms(rid)
            geom6 = compute_geom6(R_at, TS_at, P_at)  # (6,)
            part = partitions[rid]
            if "error" in part:
                raise RuntimeError(f"partition error: {part['error']}")
            frag_A = part["frag_A_indices"]
            frag_B = part["frag_B_indices"]
            # Coverage check
            if set(frag_A) & set(frag_B):
                raise RuntimeError("frag A/B overlap")
            if not frag_A or not frag_B:
                raise RuntimeError("empty fragment")

            orca_row = orca.loc[rid]
            xtb = compute_xtb_block(
                TS_at, frag_A, frag_B,
                total_charge=int(orca_row["total_charge"]),
                charge_a=int(orca_row["fragment_charge_a"]),
                charge_b=int(orca_row["fragment_charge_b"]),
                mult_a=int(orca_row["fragment_mult_a"]),
                mult_b=int(orca_row["fragment_mult_b"]),
            )
            row_out = {"reaction_id": rid, "family": fam}
            for k, v in enumerate(geom6):
                row_out[f"d{k + 1}"] = float(v)
            row_out.update(xtb)
            rows.append(row_out)
            n_ok += 1
            print(f"  ok  ({time.time()-_t_rxn:.1f}s)", flush=True)
        except Exception as e:
            rows.append({"reaction_id": rid, "family": fam,
                         "error": f"{type(e).__name__}: {e}"})
            n_fail += 1
            print(f"  FAIL {type(e).__name__}: {e}  ({time.time()-_t_rxn:.1f}s)", flush=True)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(labels)} "
                  f"ok={n_ok} fail={n_fail} elapsed={elapsed:.0f}s", flush=True)

    df_new = pd.DataFrame(rows)
    if not prev.empty:
        df_out = pd.concat([prev, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(out_pq)
    print(f"wrote {out_pq}   ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
