"""Recover d1..d24 for the 98 reactions that failed GFN2-xTB, using ORCA.

For each failed rxn:
- geom6 (d1..d6): recomputed from the R/TS/P atoms (no QM needed).
- xTB block (d7..d24): replaced by ORCA PBEh-3c single-points on
  complex@TS, fragA@TS, fragB@TS. If ORCA prints separate exchange /
  Coulomb terms, they collapse into Pauli when needed (per user note),
  but for descriptor construction we only use E_total, dipole vector,
  orbital energies (HOMO / LUMO), Mulliken atomic charges, and Mayer
  bond orders — the same scalars we used from tblite.

Sharded: --shard i / --nshards 8. Writes
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1_orca_chunks/
      shard_{i}_of_{n}.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
from eda_asm.asr_v1.baseline_physics import compute_descriptors

ORCA = "/home1/yeseo1ee/orca_6_1_1_avx2/orca"
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
PARTITIONS_JSON = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json")
DESC_PARQUET = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1.parquet")
OUT_CHUNK = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v1_orca_chunks")
OUT_CHUNK.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KCAL = 627.509474

ELEMENT_SYMBOLS = {
    1:  "H", 6:  "C", 7:  "N", 8:  "O", 9:  "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
    5: "B", 14: "Si",
}


def load_ts_atoms(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu",
                   weights_only=False)
    R = ase.Atoms(numbers=d["R"]["z"], positions=np.asarray(d["R"]["pos"]))
    TS = ase.Atoms(numbers=d["TS"]["z"], positions=np.asarray(d["TS"]["pos"]))
    P = ase.Atoms(numbers=d["P"]["z"], positions=np.asarray(d["P"]["pos"]))
    return R, TS, P


def compute_geom6(R, TS, P) -> np.ndarray:
    if len(P) != len(TS):
        P = TS
    return compute_descriptors(R, TS, P)


def write_orca_input(name, numbers, positions, charge, mult, workdir: Path,
                     ncpu: int = 4) -> Path:
    inp = workdir / f"{name}.inp"
    # PBEh-3c is a fast composite hybrid; robust for CHNOF/Cl/Br molecules.
    # Additional keywords:
    #   TightSCF, KDIIS + SlowConv for hard cases,
    #   Grid5 for stability, output Mulliken + dipole + Mayer bond orders.
    # Serial ORCA (no %pal) — this cluster lacks openmpi in PATH by default
    # and the wall isn't the bottleneck (PBEh-3c is fast for < 30 atoms).
    # PBEh-3c is a composite hybrid; SlowConv + KDIIS help problem systems.
    # ORCA 6.1 prints Mulliken charges by default; Mayer bond orders via
    # %output block.
    header = [
        "! PBEh-3c TightSCF SlowConv KDIIS",
        "%maxcore 4000",
        "%scf",
        "  MaxIter 400",
        "end",
        "%output",
        "  Print[P_Mayer] 1",
        "end",
        f"* xyz {int(charge)} {int(mult)}",
    ]
    coords = []
    for z, xyz in zip(numbers, positions):
        sym = ELEMENT_SYMBOLS.get(int(z), "X")
        coords.append(f"{sym:>3s}  {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")
    coords.append("*")
    inp.write_text("\n".join(header + coords) + "\n")
    return inp


def run_orca(inp: Path, timeout_s: int = 3600) -> str:
    """ORCA MUST be invoked from its own working dir with a basename input."""
    out_txt = inp.with_suffix(".out")
    with open(out_txt, "w") as f:
        try:
            subprocess.run([ORCA, inp.name], stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout_s, cwd=str(inp.parent), check=False)
        except subprocess.TimeoutExpired:
            f.write("\n!!TIMEOUT!!\n")
    return out_txt.read_text()


def parse_orca(txt: str) -> dict:
    """Parse energy, dipole, orbital energies (HOMO/LUMO), Mulliken charges,
    Mayer bond orders. Returns floats/arrays or raises ValueError."""
    if "!!TIMEOUT!!" in txt:
        raise ValueError("ORCA timeout")
    if "ORCA TERMINATED NORMALLY" not in txt:
        raise ValueError("ORCA did not terminate normally")

    # Energy — last "FINAL SINGLE POINT ENERGY"
    m = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    if not m:
        raise ValueError("no FINAL SINGLE POINT ENERGY found")
    E_h = float(m[-1])

    # Dipole (Debye→a.u. we'll use au directly; ORCA prints "Total Dipole Moment")
    m = re.search(r"Total Dipole Moment\s*:\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)",
                  txt)
    dipole = np.array([float(m.group(i)) for i in (1, 2, 3)]) if m else np.zeros(3)

    # Orbital energies + occupations. Block header "ORBITAL ENERGIES".
    orb_block = re.search(
        r"ORBITAL ENERGIES\s*\n[-\s]+\n[^\n]*\n[^\n]*\n((?:\s*\d+\s+[\d\.]+\s+[-\d\.]+\s+[-\d\.]+\s*\n)+)",
        txt)
    HOMO_h = LUMO_h = float("nan")
    if orb_block:
        rows = orb_block.group(1).strip().split("\n")
        idx = []; occ = []; e_h = []
        for r in rows:
            parts = r.split()
            if len(parts) >= 4:
                idx.append(int(parts[0]))
                occ.append(float(parts[1]))
                e_h.append(float(parts[2]))
        occ = np.array(occ); e_h = np.array(e_h)
        homo = np.where(occ > 1.0)[0]
        if len(homo):
            hi = int(homo[-1])
            HOMO_h = float(e_h[hi])
            LUMO_h = float(e_h[hi + 1]) if hi + 1 < len(e_h) else HOMO_h

    # Mulliken charges — block "MULLIKEN ATOMIC CHARGES"
    charges_block = re.search(
        r"MULLIKEN ATOMIC CHARGES\s*\n[-]+\n((?:\s*\d+\s+\w+\s*:\s*[-\d\.]+\s*\n)+)",
        txt)
    charges = []
    if charges_block:
        for r in charges_block.group(1).strip().split("\n"):
            m = re.match(r"\s*\d+\s+\w+\s*:\s*([-\d\.]+)", r)
            if m:
                charges.append(float(m.group(1)))
    charges = np.array(charges) if charges else np.array([])

    # Mayer bond orders — pair list "B(  i-A ,  j-B ) : bo"
    mayer = {}
    for m in re.finditer(r"B\(\s*(\d+)-\w+\s*,\s*(\d+)-\w+\s*\)\s*:\s*([\d\.]+)", txt):
        i, j = int(m.group(1)), int(m.group(2))
        mayer[(i, j)] = float(m.group(3))
        mayer[(j, i)] = float(m.group(3))

    return dict(
        E_h=E_h, HOMO_h=HOMO_h, LUMO_h=LUMO_h,
        dipole_norm=float(np.linalg.norm(dipole)),
        charges=charges, mayer=mayer,
    )


def _parity_ok(numbers: np.ndarray, charge: int, mult: int) -> bool:
    n_electrons = int(sum(numbers)) - int(charge)
    n_unpaired = mult - 1
    return (n_electrons % 2) == (n_unpaired % 2)


def _fix_mult(numbers: np.ndarray, charge: int, mult: int) -> int:
    """Return a mult with parity matching the electron count. Prefer the
    caller's mult, else the closest admissible value (1 or 2)."""
    if _parity_ok(numbers, charge, mult):
        return mult
    # Flip parity by ±1; prefer smaller mult
    for cand in (mult - 1, mult + 1, 1, 2, 3):
        if cand >= 1 and _parity_ok(numbers, charge, cand):
            return cand
    return 1


def compute_orca_block(TS, frag_A_idx, frag_B_idx,
                       total_charge, charge_a, charge_b,
                       mult_a, mult_b, workdir: Path, ncpu: int = 4) -> dict:
    numbers = np.array(TS.get_atomic_numbers())
    positions = TS.get_positions()

    # Auto-adjust multiplicities so ORCA accepts them (parity must match
    # electron count). The ADF label multiplicities can be inconsistent with
    # our reconstructed fragment partitions.
    mult_complex = _fix_mult(numbers, total_charge, 1)
    mult_a = _fix_mult(numbers[np.array(frag_A_idx, dtype=int)], charge_a, mult_a)
    mult_b = _fix_mult(numbers[np.array(frag_B_idx, dtype=int)], charge_b, mult_b)

    # 1) complex
    inp = write_orca_input("complex", numbers, positions,
                           charge=total_charge, mult=mult_complex,
                           workdir=workdir, ncpu=ncpu)
    rc = parse_orca(run_orca(inp))
    # 2) fragA at TS positions
    idx_a = np.array(frag_A_idx, dtype=int)
    inp = write_orca_input("fragA", numbers[idx_a], positions[idx_a],
                           charge=charge_a, mult=mult_a, workdir=workdir, ncpu=ncpu)
    ra = parse_orca(run_orca(inp))
    # 3) fragB
    idx_b = np.array(frag_B_idx, dtype=int)
    inp = write_orca_input("fragB", numbers[idx_b], positions[idx_b],
                           charge=charge_b, mult=mult_b, workdir=workdir, ncpu=ncpu)
    rb = parse_orca(run_orca(inp))

    E_C, E_A, E_B = rc["E_h"], ra["E_h"], rb["E_h"]
    d7 = (E_C - E_A - E_B) * HARTREE_TO_KCAL
    d8 = E_C * HARTREE_TO_KCAL
    d9 = E_A * HARTREE_TO_KCAL
    d10 = E_B * HARTREE_TO_KCAL
    d11 = rc["dipole_norm"]; d12 = ra["dipole_norm"]; d13 = rb["dipole_norm"]
    d14 = d11 - d12 - d13
    d15, d16 = rc["HOMO_h"], rc["LUMO_h"]
    d17, d18 = ra["HOMO_h"], ra["LUMO_h"]
    d19, d20 = rb["HOMO_h"], rb["LUMO_h"]
    q_C = rc["charges"]
    d21 = float(q_C[idx_a].sum()) if len(q_C) else 0.0
    mu = 0.5 * (rc["HOMO_h"] + rc["LUMO_h"])
    eta = 0.5 * max(rc["LUMO_h"] - rc["HOMO_h"], 1e-6)
    d22 = float((mu ** 2) / (2 * eta))
    d23 = float(np.sum(q_C ** 2)) if len(q_C) else 0.0
    # d24: interfragment Mayer bond orders
    d24 = 0.0
    for a in idx_a:
        for b in idx_b:
            d24 += abs(rc["mayer"].get((int(a) + 1, int(b) + 1), 0.0))
    d24 = float(d24)

    return dict(
        d7=d7, d8=d8, d9=d9, d10=d10, d11=d11, d12=d12, d13=d13, d14=d14,
        d15=d15, d16=d16, d17=d17, d18=d18, d19=d19, d20=d20,
        d21=d21, d22=d22, d23=d23, d24=d24,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--ncpu", type=int, default=4)
    args = ap.parse_args()

    # Load failed rxns
    desc = pd.read_parquet(DESC_PARQUET)
    failed = desc[desc.error.notna()][["reaction_id", "family"]].reset_index(drop=True)
    print(f"{len(failed)} rxns to recover", flush=True)

    labels = pd.read_parquet(REPO / "labels/orca/orca_eda_labels.parquet")
    labels = labels.set_index("reaction_id")
    partitions = json.load(open(PARTITIONS_JSON))

    # Round-robin shard
    failed = failed[failed.index % args.nshards == args.shard].reset_index(drop=True)
    print(f"shard {args.shard}/{args.nshards} → {len(failed)} rxns", flush=True)

    out_pq = OUT_CHUNK / f"shard_{args.shard:03d}_of_{args.nshards}.parquet"
    have = set()
    if out_pq.exists():
        prev = pd.read_parquet(out_pq)
        if "error" in prev.columns:
            prev = prev[prev["error"].isna()]
        have = set(prev.reaction_id)
    else:
        prev = pd.DataFrame()

    rows = []
    for i, row in failed.iterrows():
        rid, fam = row.reaction_id, row.family
        if rid in have:
            continue
        _t_rxn = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] i={i} rid={rid}", flush=True)
        workdir = Path(tempfile.mkdtemp(prefix=f"orca_{rid}_",
                                        dir="/gpfs/tmp_cpu2/yeseo1ee"))
        try:
            R, TS, P = load_ts_atoms(rid)
            geom6 = compute_geom6(R, TS, P)
            part = partitions[rid]
            if "error" in part:
                raise RuntimeError(f"partition: {part['error']}")
            frag_A = part["frag_A_indices"]; frag_B = part["frag_B_indices"]
            if not frag_A or not frag_B or (set(frag_A) & set(frag_B)):
                raise RuntimeError("bad partition")
            orca_row = labels.loc[rid]
            xtb = compute_orca_block(
                TS, frag_A, frag_B,
                total_charge=int(orca_row["total_charge"]),
                charge_a=int(orca_row["fragment_charge_a"]),
                charge_b=int(orca_row["fragment_charge_b"]),
                mult_a=int(orca_row["fragment_mult_a"]),
                mult_b=int(orca_row["fragment_mult_b"]),
                workdir=workdir, ncpu=args.ncpu,
            )
            row_out = {"reaction_id": rid, "family": fam}
            for k, v in enumerate(geom6):
                row_out[f"d{k + 1}"] = float(v)
            row_out.update(xtb)
            rows.append(row_out)
            print(f"  ok  ({time.time() - _t_rxn:.1f}s)", flush=True)
        except Exception as e:
            rows.append({"reaction_id": rid, "family": fam,
                         "error": f"{type(e).__name__}: {e}"})
            print(f"  FAIL {type(e).__name__}: {e}  ({time.time() - _t_rxn:.1f}s)",
                  flush=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    df_new = pd.DataFrame(rows)
    df_out = pd.concat([prev, df_new], ignore_index=True) if not prev.empty else df_new
    df_out.to_parquet(out_pq)
    print(f"wrote {out_pq}   n={len(df_out)}", flush=True)


if __name__ == "__main__":
    main()
