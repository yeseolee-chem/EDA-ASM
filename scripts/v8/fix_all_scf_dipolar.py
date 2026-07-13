"""Generic fix: regenerate eda.inp for any currently-failed dipolar rxn
with SlowConv + SOSCF + MaxIter 500.

Auto-detects failures (eda.out exists but no TERMINATED NORMALLY).
Writes manifest_scf_fix.txt for the runner.
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


def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def write_inp_slowconv(rid, A, B):
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
    tc = 0; fA_c = 0; fB_c = 0
    if fam in ("qmrxn20_sn2", "qmrxn20_e2"):
        fB_c = -1; tc = -1
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


EXCLUDE = {"dipolar_002784"}  # handled by rerun_002784 (aggressive VeryTightSCF+defgrid3)


def find_scf_failures():
    """Return list of rids whose eda.out shows SCF-related failure."""
    fails = []
    for d in sorted(ORCA_ROOT.iterdir()):
        if not d.is_dir() or d.name in EXCLUDE:
            continue
        out = d / "eda.out"
        if not out.exists():
            continue
        try:
            txt = out.read_text(errors="ignore")
        except Exception:
            continue
        if "ORCA TERMINATED NORMALLY" in txt:
            continue
        # SCF-related
        if ("DIIS Error" in txt or
            "MatrixLife" in txt or
            "This wavefunction IS NOT CONVERGED" in txt or
            "failed in the EDA-NOCV" in txt):
            fails.append(d.name)
    return fails


def main():
    m = json.loads(MP.read_text())
    fails = find_scf_failures()
    print(f"currently-failed SCF-related rxns: {len(fails)}")
    written = []
    for rid in fails:
        e = m.get(rid, {})
        A = e.get("frag_A_indices", [])
        B = e.get("frag_B_indices", [])
        if not A or not B:
            print(f"SKIP {rid}: no partition"); continue
        write_inp_slowconv(rid, A, B)
        print(f"regen SlowConv+SOSCF+MaxIter500: {rid}  |A|={len(A)}  |B|={len(B)}")
        written.append(rid)
    MANIFEST.write_text("\n".join(written) + "\n")
    print(f"manifest: {MANIFEST} ({len(written)} rids)")


if __name__ == "__main__":
    main()
