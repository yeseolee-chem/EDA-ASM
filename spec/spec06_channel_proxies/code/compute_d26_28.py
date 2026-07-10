"""SPEC_06 T1 - compute channel-matched proxies d26/d27/d28 (v7 776 cohort).

d26 (elst)  = kappa * Sum_{a in A, b in B} q_a * q_b / r_ab
              q from separated fragment single-point Mulliken charges;
              r_ab in Bohr from complex TS geometry (matches EDA elst frozen density).
d27 (Pauli) = Sum_{a in A, b in B} exp((r_vdW_ab - r_ab_A) / 0.3)
              only inter-fragment pairs; r in Angstrom; r_vdW from ase.
d28 (oi)    = |Sum_{a in A, b in B} WBO_ab| * (1/(eLUMO_B - eHOMO_A) + 1/(eLUMO_A - eHOMO_B))
              WBO from complex xTB (bond_orders); orbital energies from
              separated fragment single-points.

Reuses _run_xtb from stage3. Input geometry = MACE .pt TS positions.
Charges/mults from labels/orca/orca_eda_charges_v7.parquet.
Partitions from outputs/final_776_v7/fragmentation/orca_inp_partitions.json.

Output: /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d26_28_chunks/shard_*.parquet
"""
from __future__ import annotations

# tblite before torch (libgomp GOMP_5.0)
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import argparse, json, sys, time
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch
from ase.data import vdw_radii
from scipy.spatial.distance import cdist

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "pipeline_rebuild" / "spec_v1"))
from stage3_xtb_and_descriptors import _run_xtb  # noqa: E402

LABELS_V7  = REPO / "labels/orca/orca_eda_labels_v7.parquet"
PART_V7    = REPO / "outputs/final_776_v7/fragmentation/orca_inp_partitions.json"
CHARGES_V7 = REPO / "labels/orca/orca_eda_charges_v7.parquet"
FEAT_DIR   = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
CHUNK_DIR  = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d26_28_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KCAL = 627.5094740631
BOHR_TO_ANG = 0.5291772108


def load_ts_atoms(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), weights_only=False, map_location="cpu")
    return ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"]))


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
            idx_A = np.array(part["frag_A_indices"], dtype=int)
            idx_B = np.array(part["frag_B_indices"], dtype=int)
            if not len(idx_A) or not len(idx_B):
                raise RuntimeError("empty fragment")

            if rid not in charges.index:
                raise RuntimeError("no charge row")
            ch = charges.loc[rid]
            q_tot = int(ch["total_charge"])
            q_A_c, m_A = int(ch["fragment_charge_a"]), int(ch["fragment_mult_a"])
            q_B_c, m_B = int(ch["fragment_charge_b"]), int(ch["fragment_mult_b"])

            TS_at = load_ts_atoms(rid)
            Z = np.array(TS_at.get_atomic_numbers())
            pos_ang = TS_at.get_positions()  # Angstrom (matches ase Atoms)
            pos_bohr = pos_ang / BOHR_TO_ANG

            # 1) Complex SP with bond-orders (for WBO matrix)
            rc = _run_xtb(Z, pos_ang, charge=q_tot, mult=1, want_bond_orders=True)
            WBO = rc["bond_orders"]  # (N, N)

            # 2) Separated fragment SPs (frozen density surrogate: EDA elst uses
            #    fragment-A and fragment-B independent electron densities).
            rA = _run_xtb(Z[idx_A], pos_ang[idx_A], charge=q_A_c, mult=m_A)
            rB = _run_xtb(Z[idx_B], pos_ang[idx_B], charge=q_B_c, mult=m_B)
            qa = rA["charges"]  # (|A|,)
            qb = rB["charges"]  # (|B|,)
            eLUMO_A = rA["LUMO_h"]; eHOMO_A = rA["HOMO_h"]
            eLUMO_B = rB["LUMO_h"]; eHOMO_B = rB["HOMO_h"]

            # d26 - electrostatic Coulomb sum (charges * charges / r) in Bohr -> kcal/mol
            r_bohr = cdist(pos_bohr[idx_A], pos_bohr[idx_B])  # (|A|, |B|)
            d26 = float(np.sum(np.outer(qa, qb) / (r_bohr + 1e-12)) * HARTREE_TO_KCAL)

            # d27 - inter-fragment Pauli exp overlap (Angstrom scale, matches d3)
            r_ang = cdist(pos_ang[idx_A], pos_ang[idx_B])  # (|A|, |B|)
            rvdw = np.array([vdw_radii[int(z)] for z in Z])
            rvdw_A = rvdw[idx_A]; rvdw_B = rvdw[idx_B]
            r_vdw_sum = rvdw_A[:, None] + rvdw_B[None, :]
            d27 = float(np.exp((r_vdw_sum - r_ang) / 0.3).sum())

            # d28 - oi FMO proxy
            gap_AB = eLUMO_B - eHOMO_A
            gap_BA = eLUMO_A - eHOMO_B
            wbo_inter = float(np.abs(WBO[np.ix_(idx_A, idx_B)]).sum())
            inv_gap = 1.0 / (gap_AB + 1e-12) + 1.0 / (gap_BA + 1e-12)
            d28 = float(wbo_inter * inv_gap)

            rows.append({
                "reaction_id": rid, "family": fam,
                "d26": d26, "d27": d27, "d28": d28,
                "wbo_inter": wbo_inter,
                "gap_AB_h": float(gap_AB), "gap_BA_h": float(gap_BA),
                "scf_ok": True,
            })
            n_ok += 1
            print(f"  ok  d26={d26:+.2f}  d27={d27:.2e}  d28={d28:+.2e}  "
                  f"({time.time()-_t:.1f}s)", flush=True)
        except Exception as e:
            rows.append({"reaction_id": rid, "family": fam,
                         "error": f"{type(e).__name__}: {e}",
                         "d26": float("nan"), "d27": float("nan"),
                         "d28": float("nan"), "scf_ok": False})
            n_fail += 1
            print(f"  FAIL {type(e).__name__}: {e}  ({time.time()-_t:.1f}s)", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] progress {i+1}/{len(labels)}  "
                  f"ok={n_ok} fail={n_fail}  elapsed={time.time()-t0:.0f}s", flush=True)

    df_new = pd.DataFrame(rows)
    df_out = pd.concat([prev, df_new], ignore_index=True) if not prev.empty else df_new
    df_out.to_parquet(out_pq)
    print(f"wrote {out_pq}  ok={n_ok}  fail={n_fail}")


if __name__ == "__main__":
    main()
