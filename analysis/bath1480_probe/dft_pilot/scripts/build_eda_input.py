#!/usr/bin/env python3
"""After OptTS converges, build EDA-NOCV SPE input at def2-TZVP.

Reads:
- <workdir>/dft_optts.xyz  (ORCA writes final optimized geometry here)
- <workdir>/labels.json    (fragment assignment for original atom order)

Writes:
- <workdir>/eda_tz.inp
"""
import json
import sys
from pathlib import Path

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


def read_xyz(xyz_path):
    """Return list of (elem, x, y, z)."""
    lines = xyz_path.read_text().splitlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def main():
    if len(sys.argv) != 2:
        print("usage: build_eda_input.py <workdir>", file=sys.stderr)
        sys.exit(1)
    wdir = Path(sys.argv[1])

    xyz = wdir / "dft_optts.xyz"
    labels = json.loads((wdir / "labels.json").read_text())

    atoms = read_xyz(xyz)
    if len(atoms) != len(labels["frags"]):
        print(f"ERROR: atom count mismatch ({len(atoms)} vs {len(labels['frags'])})",
              file=sys.stderr)
        sys.exit(2)

    # Sanity: elements must match (ORCA preserves atom order in opt output)
    for i, (a, e_expected) in enumerate(zip(atoms, labels["elements"])):
        if a[0] != e_expected:
            print(f"ERROR: atom {i} element {a[0]} != expected {e_expected}",
                  file=sys.stderr)
            sys.exit(2)

    lines = [EDA_HEADER]
    for (elem, x, y, z), frag in zip(atoms, labels["frags"]):
        lines.append(f"  {elem}({frag})  {x:16.8f}  {y:16.8f}  {z:16.8f}")
    lines.append("*")
    lines.append("")
    (wdir / "eda_tz.inp").write_text("\n".join(lines))
    print(f"wrote {wdir}/eda_tz.inp  ({len(atoms)} atoms)")


if __name__ == "__main__":
    main()
