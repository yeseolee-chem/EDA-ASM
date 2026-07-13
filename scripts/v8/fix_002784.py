"""Aggressive retry for dipolar_002784 (EDA-NOCV module failure post-SCF).

Strategy: tighter SCF + fine integration grid + no TRAH + explicit reset freq.
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
RID = "dipolar_002784"


def main():
    e = json.loads(MP.read_text()).get(RID, {})
    A = e.get("frag_A_indices", [])
    B = e.get("frag_B_indices", [])
    ts_at = ase.io.read(str(RAW / RID / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA VeryTightSCF SlowConv defgrid3 NoTRAH",
        "%maxcore 3500",
        "",
        "%scf",
        "  MaxIter 500",
        "  DirectResetFreq 1",
        "end",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv defgrid3"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv defgrid3"',
        "  FRAG1_C 0",
        "  FRAG1_M 1",
        "  FRAG2_C 0",
        "  FRAG2_M 1",
        "end",
        "",
        "* xyz 0 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    inp = ORCA_ROOT / RID / "eda.inp"
    inp.write_text("\n".join(lines))
    (ORCA_ROOT / RID / "eda.out").unlink(missing_ok=True)
    (ORCA_ROOT / RID / "eda.err").unlink(missing_ok=True)
    print(f"wrote {inp}  |A|={len(A)}  |B|={len(B)}")


if __name__ == "__main__":
    main()
