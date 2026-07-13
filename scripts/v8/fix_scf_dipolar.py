"""Regenerate eda.inp for dipolar rxns that failed SCF convergence.
Adds SlowConv + SOSCF + %scf MaxIter 500 to help convergence.
Preserves the existing TS partition. Deletes eda.out so runner retries.
Writes manifest_scf_fix.txt for the ORCA runner.
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
MP = V8 / "manual_partitions.json"
MANIFEST = ORCA_ROOT / "manifest_scf_fix.txt"

TARGET_RIDS = ["dipolar_002784", "dipolar_003315", "dipolar_003648"]


def write_inp(rid, A, B, fA_c, fB_c, tc):
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"{rid}: unassigned atoms")
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF SlowConv SOSCF",
        "%maxcore 3500",
        "",
        "%scf",
        "  MaxIter 500",
        "end",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym TightSCF SlowConv SOSCF"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym TightSCF SlowConv SOSCF"',
        f"  FRAG1_C {fA_c}",
        "  FRAG1_M 1",
        f"  FRAG2_C {fB_c}",
        "  FRAG2_M 1",
        "end",
        "",
        f"* xyz {tc} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    inp = ORCA_ROOT / rid / "eda.inp"
    inp.write_text("\n".join(lines))
    (ORCA_ROOT / rid / "eda.out").unlink(missing_ok=True)
    (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)
    return inp


def main():
    m = json.loads(MP.read_text())
    written = []
    for rid in TARGET_RIDS:
        e = m.get(rid, {})
        A = e.get("frag_A_indices", [])
        B = e.get("frag_B_indices", [])
        if not A or not B:
            print(f"SKIP {rid}: no partition")
            continue
        # dipolar defaults: neutral, mult=1
        write_inp(rid, A, B, fA_c=0, fB_c=0, tc=0)
        print(f"regen (SlowConv+SOSCF+MaxIter 500): {rid}  |A|={len(A)}  |B|={len(B)}")
        written.append(rid)
    MANIFEST.write_text("\n".join(written) + "\n")
    print(f"manifest: {MANIFEST} ({len(written)} rids)")


if __name__ == "__main__":
    main()
