"""Stage 3 (v8) — build the full descriptor tensor d1..d24 per reaction.

Cohort: 799 reactions in outputs/v8_review/labels/labels_v8_5channel.parquet.

Inputs:
  - Geometries: outputs/v8_review/raw_geoms/{rid}/{R,TS,P}.xyz
  - Partitions: outputs/v8_review/manual_partitions.json
      * frag_A_indices / frag_B_indices  -> TS partition (used everywhere here)
      * frag_A_indices_R / frag_B_indices_R -> R partition (unused here; strain
        SPs already exist at outputs/v8_review/strain_sp/{rid}/{fragA_R,fragB_R}.out)

Descriptors:
  d1..d6   geom6 (Kabsch RMSD x2, Pauli/elst/disp surrogates at TS, n_atoms).
           Reuses eda_asm.asr_v1.baseline_physics.compute_descriptors.
  d7..d21  GFN2-xTB single-points on {complex, fragA, fragB} at TS geom.
  d22..d24 derived from complex xTB output (Parr omega, sum q^2, inter-frag |WBO|).

Charge / multiplicity convention:
  - Default (dipolar, rgd1):
      complex charge=0 mult=1, fragA charge=0 mult=1, fragB charge=0 mult=1
  - qmrxn20_e2 and qmrxn20_sn2 (nucleophile/base is anionic on fragB):
      complex charge=-1, fragA charge=0, fragB charge=-1
  - Special cases dipolar_004594 and dipolar_005435 (fragA anionic):
      complex charge=-1, fragA charge=-1, fragB charge=0
  - All multiplicities are 1 (closed-shell singlet). If we hit an unpaired
    system in the future the same override table can carry mult_a/mult_b.

Output:
  Per-shard parquet under  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8/chunks/shard{S}.parquet
  Progress log             /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8/_progress.jsonl
  Merged parquet (final)   /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8.parquet

Sharding: --shard S --nshards N processes rows with row_index % N == S.
Idempotent: reactions already present in the shard's chunk parquet are skipped
on resubmit; only failures are retried.

CRITICAL: tblite must be imported BEFORE torch (torch's bundled libgomp lacks
GOMP_5.0). Also: never call calc.add("bond-orders") / calc.add("dipole") --
those are auto-returned by singlepoint(); calling add() silently disables dipole.
"""
from __future__ import annotations

# tblite MUST be imported before torch (see CLAUDE.md).
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
import torch  # noqa: F401  (kept for parity with pipeline; not strictly needed here)

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
from eda_asm.asr_v1.baseline_physics import compute_descriptors

LABELS_PARQUET = REPO / "outputs/v8_review/labels/labels_v8_5channel.parquet"
GEOM_ROOT = REPO / "outputs/v8_review/raw_geoms"
PARTITIONS_JSON = REPO / "outputs/v8_review/manual_partitions.json"

OUT_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8")
CHUNK_DIR = OUT_ROOT / "chunks"
PROGRESS = OUT_ROOT / "_progress.jsonl"
MERGED_PARQUET = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8.parquet")

HARTREE_TO_KCAL = 627.509474
BOHR_PER_ANGSTROM = 1.0 / 0.5291772108

DESCRIPTOR_COLS = [f"d{i}" for i in range(1, 25)]

# Reactions where fragA is the anion (rest of qmrxn20 puts the anion on fragB).
SPECIAL_FRAGA_ANION = {"dipolar_004594", "dipolar_005435"}


# -----------------------------------------------------------------------------
# Family / charge assignment
# -----------------------------------------------------------------------------

def infer_family(rid: str) -> str:
    """Return one of: dipolar, qmrxn20_e2, qmrxn20_sn2, rgd1. Falls back to
    the first underscore-separated token if the rid does not match the known
    prefixes (should not happen for the frozen v8 cohort)."""
    if rid.startswith("qmrxn20_e2_"):
        return "qmrxn20_e2"
    if rid.startswith("qmrxn20_sn2_"):
        return "qmrxn20_sn2"
    if rid.startswith("dipolar_"):
        return "dipolar"
    if rid.startswith("rgd1_"):
        return "rgd1"
    return rid.split("_", 1)[0]


def assign_charges(rid: str) -> tuple[int, int, int, int, int, int]:
    """Return (total_charge, charge_a, charge_b, mult_c, mult_a, mult_b).

    Defaults: neutral singlet everywhere.
    Overrides:
      qmrxn20_e2 / qmrxn20_sn2 : complex -1, fragB -1 (base/nucleophile on B).
      dipolar_004594 / dipolar_005435 : complex -1, fragA -1 (special).
    Multiplicities are always 1 for the v8 cohort.
    """
    fam = infer_family(rid)
    total, ca, cb = 0, 0, 0
    if fam in ("qmrxn20_e2", "qmrxn20_sn2"):
        total, ca, cb = -1, 0, -1
    if rid in SPECIAL_FRAGA_ANION:
        total, ca, cb = -1, -1, 0
    return total, ca, cb, 1, 1, 1


# -----------------------------------------------------------------------------
# xTB single-point wrapper
# -----------------------------------------------------------------------------

def _run_xtb(numbers: np.ndarray, positions: np.ndarray,
             charge: int, mult: int, want_bond_orders: bool = False) -> dict:
    """One GFN2-xTB single-point. Positions in Angstrom."""
    n_unpaired = mult - 1
    calc = _TbliteCalculator(
        "GFN2-xTB", numbers, positions * BOHR_PER_ANGSTROM,
        charge=float(charge), uhf=n_unpaired,
    )
    calc.set("verbosity", 0)
    calc.set("max-iter", 500)
    calc.set("mixer-damping", 0.2)
    calc.set("accuracy", 0.1)
    # NOTE: do NOT calc.add("bond-orders" / "dipole"). Those are auto-returned
    # by singlepoint(); calling add() breaks dipole (see CLAUDE.md gotcha).
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


# -----------------------------------------------------------------------------
# Descriptor blocks
# -----------------------------------------------------------------------------

def compute_geom6(R_atoms, TS_atoms, P_atoms) -> np.ndarray:
    """d1..d6 with QMrxn20 P-fallback (P may lose LG/proton atoms)."""
    if len(P_atoms) != len(TS_atoms):
        # Fallback: substitute TS for P -> d2 becomes 0. Same convention as
        # spec v1 (documented gotcha for qmrxn20_e2 / qmrxn20_sn2).
        P_atoms = TS_atoms
    return compute_descriptors(R_atoms, TS_atoms, P_atoms)  # (6,) float32


def compute_xtb_block(TS_atoms, frag_A_idx, frag_B_idx,
                      total_charge, charge_a, charge_b,
                      mult_c, mult_a, mult_b) -> dict:
    """Returns d7..d24 as a dict."""
    numbers = np.array(TS_atoms.get_atomic_numbers())
    positions = TS_atoms.get_positions()

    # 1) Complex — needs bond orders for d24
    rc = _run_xtb(numbers, positions, charge=total_charge, mult=mult_c,
                  want_bond_orders=True)
    # 2) fragA at TS positions
    idx_a = np.array(frag_A_idx, dtype=int)
    ra = _run_xtb(numbers[idx_a], positions[idx_a],
                  charge=charge_a, mult=mult_a)
    # 3) fragB at TS positions
    idx_b = np.array(frag_B_idx, dtype=int)
    rb = _run_xtb(numbers[idx_b], positions[idx_b],
                  charge=charge_b, mult=mult_b)

    E_C, E_A, E_B = rc["E_h"], ra["E_h"], rb["E_h"]
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

    # d22 = Parr electrophilicity omega = mu^2 / (2 eta),
    #       mu = (HOMO+LUMO)/2, eta = gap/2  (all for complex).
    mu = 0.5 * (rc["HOMO_h"] + rc["LUMO_h"])
    eta = 0.5 * max(rc["LUMO_h"] - rc["HOMO_h"], 1e-6)
    d22 = float((mu ** 2) / (2 * eta))
    d23 = float(np.sum(q_C ** 2))

    # d24 = sum_{a in A, b in B} |WBO_ab| from xTB bond orders in complex
    bo = rc["bond_orders"]  # (n, n)
    d24 = float(np.abs(bo[np.ix_(idx_a, idx_b)]).sum())

    return dict(
        d7=d7, d8=d8, d9=d9, d10=d10,
        d11=d11, d12=d12, d13=d13, d14=d14,
        d15=d15, d16=d16, d17=d17, d18=d18, d19=d19, d20=d20,
        d21=d21, d22=d22, d23=d23, d24=d24,
    )


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_triple(rid: str):
    """Load ase.Atoms for R, TS, P from outputs/v8_review/raw_geoms/{rid}/."""
    d = GEOM_ROOT / rid
    R = ase.io.read(str(d / "R.xyz"))
    TS = ase.io.read(str(d / "TS.xyz"))
    P = ase.io.read(str(d / "P.xyz"))
    return R, TS, P


def _log(pf, obj: dict) -> None:
    """Append one JSON line to the shared progress file."""
    pf.write(json.dumps(obj) + "\n")
    pf.flush()


# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------

def merge_chunks() -> Path:
    """Concatenate all shard parquets into MERGED_PARQUET. Returns the path."""
    shard_files = sorted(CHUNK_DIR.glob("shard*.parquet"))
    if not shard_files:
        raise RuntimeError(f"No shard parquets found in {CHUNK_DIR}")
    frames = [pd.read_parquet(p) for p in shard_files]
    df = pd.concat(frames, ignore_index=True)
    # De-dup on reaction_id (last write wins — retries are already filtered
    # in the shard writer, but be defensive across shards).
    df = df.drop_duplicates(subset=["reaction_id"], keep="last")
    MERGED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tmp = MERGED_PARQUET.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.rename(MERGED_PARQUET)
    return MERGED_PARQUET


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip compute; just merge all shard parquets into "
                         "descriptors_v8.parquet.")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        p = merge_chunks()
        print(f"[merge] wrote {p}", flush=True)
        return

    if args.nshards < 1:
        raise ValueError("--nshards must be >= 1")
    if not (0 <= args.shard < args.nshards):
        raise ValueError(f"--shard {args.shard} out of range for --nshards {args.nshards}")

    labels = pd.read_parquet(LABELS_PARQUET)
    with open(PARTITIONS_JSON) as f:
        partitions = json.load(f)

    labels = labels.reset_index(drop=True)
    mask = (np.arange(len(labels)) % args.nshards) == args.shard
    subset = labels[mask].reset_index(drop=True)
    print(f"shard {args.shard}/{args.nshards} -> {len(subset)} reactions", flush=True)

    shard_pq = CHUNK_DIR / f"shard{args.shard}.parquet"
    if shard_pq.exists():
        prev = pd.read_parquet(shard_pq)
        # Keep successful entries; retry failures on this run.
        if "error" in prev.columns:
            prev = prev[prev["error"].isna()]
        have = set(prev["reaction_id"])
        print(f"resume: {len(have)} reactions already in {shard_pq.name}", flush=True)
    else:
        prev = pd.DataFrame()
        have = set()

    rows = []
    t0 = time.time()
    n_ok = n_fail = n_skip = 0
    with open(PROGRESS, "a") as pf:
        for i, row in subset.iterrows():
            rid = row.reaction_id
            fam = infer_family(rid)
            if rid in have:
                n_skip += 1
                continue
            _t_rxn = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] i={i} rid={rid}", flush=True)
            try:
                R_at, TS_at, P_at = load_triple(rid)
                geom6 = compute_geom6(R_at, TS_at, P_at)  # (6,)

                if rid not in partitions:
                    raise RuntimeError("rid missing from manual_partitions.json")
                part = partitions[rid]
                if "error" in part:
                    raise RuntimeError(f"partition error: {part['error']}")
                frag_A = part.get("frag_A_indices")
                frag_B = part.get("frag_B_indices")
                if frag_A is None or frag_B is None:
                    raise RuntimeError("frag_A_indices / frag_B_indices missing")
                if set(frag_A) & set(frag_B):
                    raise RuntimeError("frag A/B overlap")
                if not frag_A or not frag_B:
                    raise RuntimeError("empty fragment")
                n_ts = len(TS_at)
                if max(max(frag_A), max(frag_B)) >= n_ts:
                    raise RuntimeError(
                        f"partition index out of range (n_ts={n_ts}, "
                        f"maxA={max(frag_A)}, maxB={max(frag_B)})"
                    )

                tc, ca, cb, mc, ma, mb = assign_charges(rid)
                xtb = compute_xtb_block(
                    TS_at, frag_A, frag_B,
                    total_charge=tc, charge_a=ca, charge_b=cb,
                    mult_c=mc, mult_a=ma, mult_b=mb,
                )

                row_out = {"reaction_id": rid, "family": fam,
                           "total_charge": tc,
                           "fragment_charge_a": ca, "fragment_charge_b": cb,
                           "fragment_mult_a": ma, "fragment_mult_b": mb,
                           "error": None}
                for k, v in enumerate(geom6):
                    row_out[f"d{k + 1}"] = float(v)
                row_out.update(xtb)
                rows.append(row_out)
                n_ok += 1
                _log(pf, {"rid": rid, "shard": args.shard, "status": "ok",
                          "dt_s": round(time.time() - _t_rxn, 2)})
                print(f"  ok  ({time.time() - _t_rxn:.1f}s)", flush=True)
            except Exception as e:
                rows.append({"reaction_id": rid, "family": fam,
                             "error": f"{type(e).__name__}: {e}"})
                n_fail += 1
                _log(pf, {"rid": rid, "shard": args.shard,
                          "err": f"{type(e).__name__}: {e}",
                          "dt_s": round(time.time() - _t_rxn, 2)})
                print(f"  FAIL {type(e).__name__}: {e}  "
                      f"({time.time() - _t_rxn:.1f}s)", flush=True)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{i + 1}/{len(subset)} ok={n_ok} skip={n_skip} "
                      f"fail={n_fail} elapsed={elapsed:.0f}s", flush=True)

    df_new = pd.DataFrame(rows)
    df_out = pd.concat([prev, df_new], ignore_index=True) if not prev.empty else df_new
    # De-dup within the shard (retries) — keep the freshest attempt.
    if "reaction_id" in df_out.columns:
        df_out = df_out.drop_duplicates(subset=["reaction_id"], keep="last")
    tmp = shard_pq.with_suffix(".parquet.tmp")
    df_out.to_parquet(tmp)
    tmp.rename(shard_pq)
    print(f"wrote {shard_pq}   ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
