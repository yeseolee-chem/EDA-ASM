"""Regenerate eda.inp for the 542 recovered rxns (TS=R, no ambiguity)
using the TS partition. Only regenerates if content differs from disk.
Writes manifest_recovered.txt for downstream runner.
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
MANIFEST = ORCA_ROOT / "manifest_recovered.txt"


def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def build_inp_text(rid, A, B):
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"{rid}: unassigned")
    fam = _fam(rid)
    total_c = 0; fA_c = 0; fB_c = 0
    if fam in ("qmrxn20_sn2", "qmrxn20_e2"):
        fB_c = -1; total_c = -1
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
        f"* xyz {total_c} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    return "\n".join(lines)


def main():
    m = json.loads(MP.read_text())
    manifest = []
    n_rewritten = 0; n_kept = 0; n_deleted_out = 0
    for rid, entry in m.items():
        if entry.get("needs_TS_review", True):
            continue
        A = entry.get("frag_A_indices", [])
        B = entry.get("frag_B_indices", [])
        if not A or not B:
            continue
        try:
            content = build_inp_text(rid, A, B)
        except Exception as e:
            print(f"ERR {rid}: {e}")
            continue
        inp_path = ORCA_ROOT / rid / "eda.inp"
        inp_path.parent.mkdir(parents=True, exist_ok=True)
        current = inp_path.read_text() if inp_path.exists() else ""
        if current != content:
            inp_path.write_text(content)
            n_rewritten += 1
            # only delete eda.out if inp actually changed (partition changed)
            out_path = ORCA_ROOT / rid / "eda.out"
            if out_path.exists():
                out_path.unlink()
                (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)
                n_deleted_out += 1
        else:
            n_kept += 1
        manifest.append(rid)
    MANIFEST.write_text("\n".join(manifest) + "\n")
    print(f"recovered rxns:      {len(manifest)}")
    print(f"  inp rewritten:     {n_rewritten}")
    print(f"  inp unchanged:     {n_kept}")
    print(f"  eda.out deleted:   {n_deleted_out} (partition changed)")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
