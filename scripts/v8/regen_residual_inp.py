"""Regenerate eda.inp for residual failures:
 - 4 dipolar failures (SCF issues, odd-electron issues)
 - e2 shard-8 leftovers (that were cancelled mid-array)

For dipolar:
  - dipolar_002978, dipolar_005323: SCF convergence failure -> add SlowConv + SOSCF
  - dipolar_004594, dipolar_005435: odd total electrons (sumZ odd) -> try charge=-1

For e2 shard-8 leftovers: manifest_e2_rerun.txt entries not yet completed.

Writes manifest_residual.txt for downstream runner.
"""
from __future__ import annotations
import json
from pathlib import Path
import ase.io
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
MANUAL_JSON = V8 / "manual_partitions.json"
MANIFEST_RESIDUAL = ORCA_ROOT / "manifest_residual.txt"
MANIFEST_E2_RERUN = ORCA_ROOT / "manifest_e2_rerun.txt"


def write_dipolar_inp(rid, A, B, *, total_charge=0, slowconv=False):
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"{rid}: unassigned")

    fA_c = 0
    fB_c = total_charge  # put charge on B (arbitrary but consistent)
    header = "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF"
    if slowconv:
        header += " SlowConv SOSCF"

    lines = [
        header,
        "%maxcore 3500",
        "",
        "%scf",
        "  MaxIter 500",
        "end",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        f"  FRAG1_C {fA_c}",
        "  FRAG1_M 1",
        f"  FRAG2_C {fB_c}",
        "  FRAG2_M 1",
        "end",
        "",
        f"* xyz {total_charge} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")

    out = ORCA_ROOT / rid / "eda.inp"
    out.write_text("\n".join(lines))
    (ORCA_ROOT / rid / "eda.out").unlink(missing_ok=True)
    (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)
    return out


def main():
    manual = json.loads(MANUAL_JSON.read_text())
    residual = []

    # --- 4 dipolar failures ---
    scf_rids = ["dipolar_002978", "dipolar_005323"]
    anion_rids = ["dipolar_004594", "dipolar_005435"]
    fixed_rid = ["dipolar_000658"]  # already fixed by user via app -> its inp already regenerated

    for rid in scf_rids:
        p = manual.get(rid, {})
        A = p.get("frag_A_indices", []); B = p.get("frag_B_indices", [])
        write_dipolar_inp(rid, A, B, total_charge=0, slowconv=True)
        print(f"SlowConv+SOSCF: {rid}")
        residual.append(rid)

    for rid in anion_rids:
        p = manual.get(rid, {})
        A = p.get("frag_A_indices", []); B = p.get("frag_B_indices", [])
        write_dipolar_inp(rid, A, B, total_charge=-1, slowconv=False)
        print(f"charge=-1:      {rid}")
        residual.append(rid)

    residual.extend(fixed_rid)
    print(f"fixed by user:  {fixed_rid[0]}")

    # --- e2 shard-8 leftovers ---
    e2_all = MANIFEST_E2_RERUN.read_text().splitlines()
    e2_todo = []
    for rid in e2_all:
        rid = rid.strip()
        if not rid:
            continue
        out = ORCA_ROOT / rid / "eda.out"
        if out.exists() and "ORCA TERMINATED NORMALLY" in out.read_text(errors="ignore"):
            continue
        e2_todo.append(rid)
    print(f"e2 leftovers (not yet done): {len(e2_todo)}")
    residual.extend(e2_todo)

    MANIFEST_RESIDUAL.write_text("\n".join(residual) + "\n")
    print(f"residual manifest ({len(residual)} rids) -> {MANIFEST_RESIDUAL}")


if __name__ == "__main__":
    main()
