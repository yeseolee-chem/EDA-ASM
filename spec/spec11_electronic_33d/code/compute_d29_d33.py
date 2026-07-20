"""SPEC_11 Stage 1 - compute d29..d33 for the v9 783-reaction cohort.

Two xTB passes per reaction (different molecules, cannot be merged):

  Pass alpha (isolated fragments, frozen at TS geometry):
    - fragA SP  -> dipole_A, gradient_A
    - fragB SP  -> dipole_B, gradient_B
    Yields: d29 (anisotropic elst), d33 (residual restoring force).
    d30 needs no xTB (arithmetic on d21 + charges table).

  Pass beta  (complex A U B at TS geometry):
    - single-point with overlap-matrix + hamiltonian-matrix + charges
    Yields: d31 (AO overlap^2 across A|B), d32 (|H0| across A|B).

Output: /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d29_33_v9_chunks/shard_*.parquet
Merged by merge_d29_d33.py -> spec/spec11_electronic_33d/data/descriptors_d29_d33.parquet
"""
from __future__ import annotations
import os
# tblite BEFORE torch (libgomp).
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import argparse, json, sys, time
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "spec/spec11_electronic_33d/code"))
from xtb_extract import run_xtb_extended  # noqa: E402

LABELS_V9  = REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
PART_V9    = REPO / "outputs/v8_review/manual_partitions.json"
CHARGES_V9 = REPO / "labels/orca/orca_eda_charges_v9.parquet"
FEAT_DIR   = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")
CHUNK_DIR  = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d29_33_v9_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KCAL = 627.5094740631
BOHR_TO_ANG     = 0.5291772108


def load_ts_atoms(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), weights_only=False, map_location="cpu")
    return ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"]))


def compute_d29(numbers, pos_bohr, idx_A, idx_B, dipole_A_raw, dipole_B_raw,
                Q_A_formal: int, Q_B_formal: int):
    """Anisotropic (monopole-dipole + dipole-dipole) elst increment [kcal/mol].

    Both isolated-fragment dipoles are recentred to each fragment's
    nuclear-charge centre C_X so the value is well-defined for charged
    fragments (spec 1.1). Monopole-monopole is intentionally omitted.
    """
    Z = np.asarray(numbers, dtype=np.float64)
    ZA = Z[idx_A]; ZB = Z[idx_B]
    rA = pos_bohr[idx_A]; rB = pos_bohr[idx_B]

    C_A = (ZA[:, None] * rA).sum(0) / ZA.sum()
    C_B = (ZB[:, None] * rB).sum(0) / ZB.sum()

    # Recentre dipoles: p_recent = p_raw_about_origin - Q * C (both frames
    # centred on the same fragment's nuclear-charge centre).
    p_A = dipole_A_raw - Q_A_formal * C_A
    p_B = dipole_B_raw - Q_B_formal * C_B

    R = C_B - C_A
    R_len = float(np.linalg.norm(R))
    if R_len < 1e-6:
        return 0.0
    Rhat = R / R_len

    T_qd = (Q_B_formal * float(p_A @ Rhat) - Q_A_formal * float(p_B @ Rhat)) / (R_len ** 2)
    T_dd = (float(p_A @ p_B) - 3.0 * float(p_A @ Rhat) * float(p_B @ Rhat)) / (R_len ** 3)
    return float((T_qd + T_dd) * HARTREE_TO_KCAL)


def compute_d31_d32(overlap, hamiltonian, ao_labels, idx_A, idx_B):
    """AO-blocked overlap^2 sum (d31) and |H0| sum (d32) across A<->B."""
    setA = set(int(a) for a in idx_A)
    setB = set(int(a) for a in idx_B)
    mask_A = np.array([int(a) in setA for a in ao_labels], dtype=bool)
    mask_B = np.array([int(a) in setB for a in ao_labels], dtype=bool)
    if mask_A.sum() == 0 or mask_B.sum() == 0:
        return 0.0, 0.0
    S_AB = overlap[np.ix_(mask_A, mask_B)]
    H_AB = hamiltonian[np.ix_(mask_A, mask_B)]
    d31 = float(np.sum(S_AB ** 2))
    d32 = float(np.sum(np.abs(H_AB)) * HARTREE_TO_KCAL)
    return d31, d32


def compute_d33(gA_hartree_bohr, gB_hartree_bohr):
    """Sum of Frobenius norms of nuclear gradient on isolated fragment
    frozen at TS geometry. [Hartree/Bohr]."""
    nA = float(np.sqrt(np.sum(np.asarray(gA_hartree_bohr) ** 2)))
    nB = float(np.sqrt(np.sum(np.asarray(gB_hartree_bohr) ** 2)))
    return nA + nB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    labels = pd.read_parquet(LABELS_V9)
    partitions = json.loads(PART_V9.read_text())
    charges = pd.read_parquet(CHARGES_V9).set_index("reaction_id")

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
        row_out = {"reaction_id": rid, "family": fam,
                   "d29": float("nan"), "d30": float("nan"),
                   "d31": float("nan"), "d32": float("nan"),
                   "d33": float("nan"),
                   "scf_ok_alpha": False, "scf_ok_beta": False,
                   "scf_ok": False, "error": None}
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
            pos_ang = TS_at.get_positions()
            pos_bohr = pos_ang / BOHR_TO_ANG

            # -------- pass alpha: isolated fragments --------
            rA = run_xtb_extended(Z[idx_A], pos_ang[idx_A],
                                  charge=q_A_c, mult=m_A, want_gradient=True)
            rB = run_xtb_extended(Z[idx_B], pos_ang[idx_B],
                                  charge=q_B_c, mult=m_B, want_gradient=True)
            row_out["scf_ok_alpha"] = True

            # d29 - anisotropic elst
            d29 = compute_d29(Z, pos_bohr, idx_A, idx_B,
                              rA["dipole"], rB["dipole"], q_A_c, q_B_c)

            # d33 - residual restoring force
            d33 = compute_d33(rA["gradient"], rB["gradient"])

            # -------- pass beta: complex --------
            rc = run_xtb_extended(Z, pos_ang, charge=q_tot, mult=1,
                                  want_matrices=True)
            row_out["scf_ok_beta"] = True

            # d30 - inter-fragment charge transfer (arithmetic on charges + d21)
            q_C = rc["charges"]
            d21_local = float(q_C[idx_A].sum())
            d30 = d21_local - float(q_A_c)

            # d31/d32 - AO overlap^2 and |H0| across A|B. orbital_map comes
            # from calc.get('shell-map') composed with calc.get('orbital-map').
            ao_labels = rc["orbital_map"]
            if ao_labels is None or len(ao_labels) != rc["overlap"].shape[0]:
                raise RuntimeError(
                    f"AO map length mismatch: {None if ao_labels is None else len(ao_labels)} vs "
                    f"{rc['overlap'].shape[0]} (norb)")
            d31, d32 = compute_d31_d32(rc["overlap"], rc["hamiltonian"],
                                       ao_labels, idx_A, idx_B)

            row_out.update(d29=d29, d30=d30, d31=d31, d32=d32, d33=d33,
                           scf_ok=True, error=None)
            n_ok += 1
            print(f"  ok  d29={d29:+.3e}  d30={d30:+.3f}  d31={d31:.3e}  "
                  f"d32={d32:+.3e}  d33={d33:.3e}  ({time.time()-_t:.1f}s)",
                  flush=True)
        except Exception as e:
            row_out["error"] = f"{type(e).__name__}: {e}"
            n_fail += 1
            print(f"  FAIL {type(e).__name__}: {e}  ({time.time()-_t:.1f}s)",
                  flush=True)
        rows.append(row_out)
        if (i + 1) % 25 == 0:
            df_partial = pd.DataFrame(rows)
            df_out = pd.concat([prev, df_partial], ignore_index=True) if not prev.empty else df_partial
            df_out.to_parquet(out_pq)
            print(f"[{time.strftime('%H:%M:%S')}] progress {i+1}/{len(labels)}  "
                  f"ok={n_ok} fail={n_fail}  elapsed={time.time()-t0:.0f}s "
                  f"(checkpointed)", flush=True)

    df_new = pd.DataFrame(rows)
    df_out = pd.concat([prev, df_new], ignore_index=True) if not prev.empty else df_new
    df_out.to_parquet(out_pq)
    print(f"wrote {out_pq}  ok={n_ok}  fail={n_fail}")


if __name__ == "__main__":
    main()
