"""Regenerate eda.inp for all failed qmrxn20_e2 reactions with correct
charge specification (base carries -1, total charge -1). Also delete the
broken eda.out so the runner picks these up on next submit.

Writes a targeted manifest at outputs/v8_review/orca_inputs/manifest_e2_rerun.txt.
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
MANIFEST_OUT = ORCA_ROOT / "manifest_e2_rerun.txt"


def write_e2_inp(rid: str, frag_A: list[int], frag_B: list[int]) -> Path:
    """qmrxn20_e2: fragment B = anionic base -> FRAG2_C=-1, total=-1, mult=1."""
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in frag_A:
        frag_of[i] = 1
    for i in frag_B:
        frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"{rid}: unassigned atoms {[i for i, f in enumerate(frag_of) if f is None]}")

    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF",
        "%maxcore 3500",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        "  FRAG1_C 0",
        "  FRAG1_M 1",
        "  FRAG2_C -1",
        "  FRAG2_M 1",
        "end",
        "",
        "* xyz -1 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*")
    lines.append("")
    inp_path = ORCA_ROOT / rid / "eda.inp"
    inp_path.write_text("\n".join(lines))
    return inp_path


def find_failed_e2():
    out = []
    for d in sorted(ORCA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith("qmrxn20_e2_"):
            continue
        eda_out = d / "eda.out"
        if not eda_out.exists():
            continue
        try:
            txt = eda_out.read_text(errors="ignore")
        except Exception:
            continue
        if "ORCA TERMINATED NORMALLY" in txt:
            continue
        out.append(d.name)
    return out


def main():
    manual = json.loads(MANUAL_JSON.read_text())
    failed = find_failed_e2()
    print(f"failed qmrxn20_e2 rxns: {len(failed)}")
    n_ok = 0
    n_missing = 0
    written = []
    for rid in failed:
        p = manual.get(rid, {})
        A = p.get("frag_A_indices")
        B = p.get("frag_B_indices")
        if A is None or B is None:
            print(f"MISS {rid}: no partition in manual_partitions.json")
            n_missing += 1
            continue
        try:
            write_e2_inp(rid, A, B)
        except Exception as e:
            print(f"ERR  {rid}: {e}")
            n_missing += 1
            continue
        # remove broken eda.out so runner re-tries
        (ORCA_ROOT / rid / "eda.out").unlink(missing_ok=True)
        (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)
        n_ok += 1
        written.append(rid)
    MANIFEST_OUT.write_text("\n".join(written) + "\n")
    print(f"regenerated {n_ok} inp,  skipped {n_missing},  manifest -> {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
