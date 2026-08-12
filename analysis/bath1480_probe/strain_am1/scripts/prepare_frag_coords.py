#!/usr/bin/env python3
"""Extract frag1/frag2 AM1 TS coordinates from every kisti_results eda.inp.

Writes:
  strain_am1/data/rxn_XXXX/frag1_am1.xyz  (fragment 1 atoms, AM1 TS coords)
  strain_am1/data/rxn_XXXX/frag2_am1.xyz  (fragment 2 atoms, AM1 TS coords)

XYZ files are standard 3-line-header + coord format ready for ORCA `xyzfile`.
"""
import re
from pathlib import Path

SRC_DATA = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")
OUT_DATA = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/data")

ATOM_RE = re.compile(
    r"^\s*([A-Za-z]+)\(([12])\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


def parse_eda_inp(path: Path):
    frag1, frag2 = [], []
    for ln in path.read_text().splitlines():
        m = ATOM_RE.match(ln)
        if not m:
            continue
        elem, frag, x, y, z = m.groups()
        (frag1 if frag == "1" else frag2).append(
            (elem, float(x), float(y), float(z))
        )
    return frag1, frag2


def write_xyz(atoms, path: Path, comment: str):
    lines = [str(len(atoms)), comment]
    for e, x, y, z in atoms:
        lines.append(f"{e:<3}  {x:16.8f}  {y:16.8f}  {z:16.8f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    rxn_dirs = sorted(SRC_DATA.glob("rxn_[0-9][0-9][0-9][0-9]"))
    print(f"found {len(rxn_dirs)} rxn dirs in source")

    ok = 0
    skipped = 0
    fails = []
    for src in rxn_dirs:
        inp = src / "eda.inp"
        if not inp.exists():
            fails.append((src.name, "eda.inp missing"))
            continue
        try:
            f1, f2 = parse_eda_inp(inp)
        except Exception as e:
            fails.append((src.name, str(e)))
            continue
        if not f1 or not f2:
            fails.append((src.name, f"frag counts {len(f1)}/{len(f2)}"))
            continue

        out_dir = OUT_DATA / src.name
        out_dir.mkdir(parents=True, exist_ok=True)
        f1_path = out_dir / "frag1_am1.xyz"
        f2_path = out_dir / "frag2_am1.xyz"

        if f1_path.exists() and f2_path.exists():
            skipped += 1
            ok += 1
            continue

        write_xyz(f1, f1_path,
                  f"{src.name} frag1 (AM1 TS geom, {len(f1)} atoms)")
        write_xyz(f2, f2_path,
                  f"{src.name} frag2 (AM1 TS geom, {len(f2)} atoms)")
        ok += 1

    print(f"ok: {ok}  skipped (already existed): {skipped}  failed: {len(fails)}")
    if fails:
        for name, why in fails[:10]:
            print(f"  FAIL {name}: {why}")


if __name__ == "__main__":
    main()
