"""Generate ORCA SP inputs for isolated fragments at R geometry.

For each of the 800 rxns:
  - fragA_R.inp: SP of frag_A atoms at their R positions
  - fragB_R.inp: SP of frag_B atoms at their R positions

These are the E_A(R) and E_B(R) terms for the ASM strain channel:
  ΔE_strain = [E_A(TS) - E_A(R)] + [E_B(TS) - E_B(R)]

E_A(TS), E_B(TS) come from the existing EDA outputs (eda_frag1.out / eda_frag2.out).
"""
from __future__ import annotations
import json
from pathlib import Path
import ase.io
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
MANUAL_JSON = V8 / "manual_partitions.json"
SP_ROOT = V8 / "strain_sp"
SP_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = SP_ROOT / "manifest.txt"


def family(rid: str) -> str:
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def write_sp(rid: str, frag_idx: list[int], frag_charge: int, out_path: Path):
    """Write ORCA SP for the specified fragment atoms at R geometry."""
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z = r_at.get_atomic_numbers()
    pos = r_at.get_positions()
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym TightSCF",
        "%maxcore 3500",
        "",
        f"* xyz {frag_charge} 1",
    ]
    for i in frag_idx:
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    manual = json.loads(MANUAL_JSON.read_text())
    n_ok = 0; n_skip = 0
    manifest = []
    for rid, entry in manual.items():
        # STRAIN SP uses R partition (frag_A_indices_R), NOT TS partition
        A = entry.get("frag_A_indices_R")
        B = entry.get("frag_B_indices_R")
        if A is None or B is None or not A or not B:
            print(f"SKIP {rid}: no R partition")
            n_skip += 1
            continue
        fam = family(rid)
        # Charges match EDA convention
        fA_c = 0
        if fam in ("qmrxn20_sn2", "qmrxn20_e2"):
            fB_c = -1
        else:
            fB_c = 0

        rid_dir = SP_ROOT / rid
        rid_dir.mkdir(exist_ok=True)
        write_sp(rid, A, fA_c, rid_dir / "fragA_R.inp")
        write_sp(rid, B, fB_c, rid_dir / "fragB_R.inp")
        manifest.append(rid)
        n_ok += 1

    MANIFEST.write_text("\n".join(manifest) + "\n")
    print(f"wrote SP inputs for {n_ok} rxns  ({n_skip} skipped)")
    print(f"manifest: {MANIFEST}  ({len(manifest)} rids)")


if __name__ == "__main__":
    main()
