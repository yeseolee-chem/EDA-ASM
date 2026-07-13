"""Regenerate inputs for:
  - 2 dipolar rxns with odd total electrons (charge=-1 with fragA carrying the -1)
    -> eda.inp AND fragA_R.inp updated
  - fragB_R.inp stays as neutral (already even)
  - 8 not-run rxns keep existing inp (just need submission)

Writes manifest_final.txt containing 8+2 EDA rerun rids and 2 SP rerun (fragA only) pairs.
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
SP_ROOT = V8 / "strain_sp"
MP = V8 / "manual_partitions.json"
MANIFEST_EDA = ORCA_ROOT / "manifest_final_eda.txt"
MANIFEST_SP = SP_ROOT / "manifest_final_sp.txt"

# Rxns needing charge=-1 (odd sumZ)
CHARGE_MINUS_ONE = ["dipolar_004594", "dipolar_005435"]

# Rxns that never ran (need re-execution with existing inp)
NOT_RUN = [
    "dipolar_001911", "dipolar_002456", "dipolar_002808", "dipolar_002814",
    "dipolar_002819", "dipolar_002856", "dipolar_002864", "dipolar_002882",
]


def write_eda_inp(rid, A, B, fA_c, fB_c, tc):
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"{rid}: unassigned")
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF",
        "%maxcore 3500",
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
        f"* xyz {tc} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    (ORCA_ROOT / rid / "eda.inp").write_text("\n".join(lines))
    (ORCA_ROOT / rid / "eda.out").unlink(missing_ok=True)
    (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)


def write_sp(rid, frag_idx, charge, out_path):
    """Write SP for one fragment at R geometry, from R.xyz."""
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z = r_at.get_atomic_numbers()
    pos = r_at.get_positions()
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym TightSCF",
        "%maxcore 3500",
        "",
        f"* xyz {charge} 1",
    ]
    for i in frag_idx:
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    m = json.loads(MP.read_text())
    eda_manifest = list(NOT_RUN)  # already-existing inp, just need run

    # Charge -1 fix (fragA carries -1)
    for rid in CHARGE_MINUS_ONE:
        e = m.get(rid, {})
        A = e.get("frag_A_indices", [])
        B = e.get("frag_B_indices", [])
        A_R = e.get("frag_A_indices_R", [])
        # EDA: fragA_c = -1, fragB_c = 0, total = -1
        write_eda_inp(rid, A, B, fA_c=-1, fB_c=0, tc=-1)
        # strain SP fragA: charge = -1
        sp_dir = SP_ROOT / rid
        sp_dir.mkdir(exist_ok=True)
        write_sp(rid, A_R, charge=-1, out_path=sp_dir / "fragA_R.inp")
        (sp_dir / "fragA_R.out").unlink(missing_ok=True)
        (sp_dir / "fragA_R.err").unlink(missing_ok=True)
        eda_manifest.append(rid)
        print(f"charge=-1 fix: {rid}   |A|={len(A)}(charged) |B|={len(B)}")
    MANIFEST_EDA.write_text("\n".join(eda_manifest) + "\n")
    MANIFEST_SP.write_text("\n".join(CHARGE_MINUS_ONE) + "\n")
    print(f"\nEDA manifest: {MANIFEST_EDA} ({len(eda_manifest)} rids)")
    print(f"SP manifest:  {MANIFEST_SP} ({len(CHARGE_MINUS_ONE)} rids fragA only)")


if __name__ == "__main__":
    main()
