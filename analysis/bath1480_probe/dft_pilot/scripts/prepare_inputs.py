#!/usr/bin/env python3
"""Prepare DFT TS reopt + EDA-NOCV SPE inputs for pilot reactions.

For each rxn_id:
- Read AM1-based eda.inp (has fragment labels C(1) etc.)
- Strip labels → write dft_optts.inp (B3LYP-D3BJ/def2-SVP SMD water, OptTS+NumFreq)
- Store fragment label mapping in labels.json for later EDA SPE input build
"""
import json
import re
import sys
from pathlib import Path

PILOT_BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot")
SRC_DATA = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")

# 5 pilot rxns spanning 16..55 atoms
RXN_IDS = [3, 7, 56, 489, 1217]

# --- ORCA input templates ---
OPTTS_HEADER = """! OptTS NumFreq B3LYP D3BJ def2-SVP CPCM(water) NoSym TightSCF
%maxcore 3500
%pal nprocs 4 end
%cpcm
  smd true
  smdsolvent "water"
end
%geom
  Calc_Hess true
  Recalc_Hess 10
end
* xyz 0 1
"""

EDA_HEADER = """! B3LYP D3BJ def2-TZVP CPCM(water) NoSym EDA TightSCF
%maxcore 3500
%pal nprocs 4 end
%cpcm
  smd true
  smdsolvent "water"
end
%eda
  FRAG1 "B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF"
  FRAG2 "B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF"
  FRAG1_C 0
  FRAG1_M 1
  FRAG2_C 0
  FRAG2_M 1
end
* xyz 0 1
"""

ATOM_RE = re.compile(r"^\s*([A-Za-z]+)\(([12])\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_eda_inp(path: Path):
    """Return list of {elem, frag(1|2), x, y, z}."""
    atoms = []
    for line in path.read_text().splitlines():
        m = ATOM_RE.match(line)
        if m:
            elem, frag, x, y, z = m.groups()
            atoms.append({"elem": elem, "frag": int(frag),
                          "x": float(x), "y": float(y), "z": float(z)})
    return atoms


def write_optts_input(atoms, out_path):
    lines = [OPTTS_HEADER]
    for a in atoms:
        lines.append(f"  {a['elem']:<3}  {a['x']:16.8f}  {a['y']:16.8f}  {a['z']:16.8f}")
    lines.append("*")
    lines.append("")
    out_path.write_text("\n".join(lines))


def write_labels_json(atoms, out_path):
    """Store fragment assignment (order matches atom index)."""
    out_path.write_text(json.dumps({
        "elements": [a["elem"] for a in atoms],
        "frags":    [a["frag"] for a in atoms],
    }, indent=2))


def main():
    for rxn in RXN_IDS:
        src = SRC_DATA / f"rxn_{rxn:04d}" / "eda.inp"
        atoms = parse_eda_inp(src)
        wdir = PILOT_BASE / "work" / f"rxn_{rxn:04d}"
        wdir.mkdir(parents=True, exist_ok=True)
        write_optts_input(atoms, wdir / "dft_optts.inp")
        write_labels_json(atoms, wdir / "labels.json")
        print(f"rxn_{rxn:04d}: {len(atoms)} atoms  → {wdir}/dft_optts.inp")


if __name__ == "__main__":
    main()
