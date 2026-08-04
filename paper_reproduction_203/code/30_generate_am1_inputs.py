#!/usr/bin/env python3
"""
Generate Gaussian AM1 input files (.gjf) from geometries/ for all rxns.

File naming convention (matches Grayson f_extract.py parsing expectations):
  gaussian_inputs/
    ts/       ts_<rn>.gjf             → file.split('_')[1] = <rn>
    gs/       gs_<rn>_reactant_1.gjf, gs_<rn>_reactant_2.gjf
    dist_gs/  dist_<rn>_reactant_1.gjf, dist_<rn>_reactant_2.gjf

Route line: `#P AM1 Freq NoSymm`
  - AM1: Dewar 1985 semi-empirical HF
  - Freq: needed for APT (Atomic Polar Tensor) charges — cclib parses these
  - NoSymm: match our ORCA convention (no symmetry usage)
  - #P: verbose output (Mulliken analysis + APT tensors + full SCF details)

Total files: N_rxn × 5 (1 TS + 2 gs + 2 dist_gs). For 203 rxns → 1015 files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROUTE = "#P AM1 Freq NoSymm"
MEM = "2GB"
NPROC = 2


def read_xyz(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    out = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        out.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return out


def write_gjf(dest: Path, chk_stem: str, title: str, charge: int, mult: int,
              atoms: list[tuple[str, float, float, float]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"%chk={chk_stem}.chk",
        f"%mem={MEM}",
        f"%nprocshared={NPROC}",
        ROUTE,
        "",
        title,
        "",
        f"{charge} {mult}",
    ]
    for el, x, y, z in atoms:
        lines.append(f" {el:<2s}  {x:14.7f}  {y:14.7f}  {z:14.7f}")
    lines += ["", ""]   # Gaussian requires blank line + trailing blank
    dest.write_text("\n".join(lines))


def process_rxn(rid: str, rn: int, geom_dir: Path, out_dirs: dict) -> None:
    meta_path = geom_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing")
    meta = json.loads(meta_path.read_text())

    # All 400 rxns are 0/1 for total & both fragments (verified earlier)
    charge_total = meta["charge_total"]
    mult_total = meta["mult_total"]
    # For fragments, assume 0/1 (verified) — could be extended by meta if needed
    charge_frag = 0
    mult_frag = 1

    # TS
    ts_atoms = read_xyz(geom_dir / "TS.xyz")
    write_gjf(
        out_dirs["ts"] / f"ts_{rn}.gjf",
        f"ts_{rn}",
        f"{rid} AM1 SPE on TS complex (n={len(ts_atoms)})",
        charge_total, mult_total, ts_atoms,
    )

    # Relaxed reactants
    for slot_name, xyz_name in [("reactant_1", "reactant_1.xyz"), ("reactant_2", "reactant_2.xyz")]:
        atoms = read_xyz(geom_dir / xyz_name)
        write_gjf(
            out_dirs["gs"] / f"gs_{rn}_{slot_name}.gjf",
            f"gs_{rn}_{slot_name}",
            f"{rid} AM1 SPE on relaxed {slot_name} (n={len(atoms)})",
            charge_frag, mult_frag, atoms,
        )

    # Distorted reactants
    for slot_name, xyz_name in [("reactant_1", "reactant_1_distorted.xyz"),
                                 ("reactant_2", "reactant_2_distorted.xyz")]:
        atoms = read_xyz(geom_dir / xyz_name)
        write_gjf(
            out_dirs["dist_gs"] / f"dist_{rn}_{slot_name}.gjf",
            f"dist_{rn}_{slot_name}",
            f"{rid} AM1 SPE on distorted {slot_name} @ TS geom (n={len(atoms)})",
            charge_frag, mult_frag, atoms,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent.parent / "MANIFEST.txt")
    ap.add_argument("--geom-root", type=Path,
                    default=Path(__file__).parent.parent / "geometries")
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).parent.parent / "gaussian_inputs")
    args = ap.parse_args()

    out_dirs = {
        "ts":      args.out_root / "ts",
        "gs":      args.out_root / "gs",
        "dist_gs": args.out_root / "dist_gs",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Parse manifest: line = "rn\treaction_id"
    rxns: list[tuple[int, str]] = []
    for line in args.manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rxns.append((int(parts[0]), parts[1]))
    print(f"generating .gjf for {len(rxns)} rxns")

    fails = []
    for rn, rid in rxns:
        try:
            process_rxn(rid, rn, args.geom_root / rid, out_dirs)
        except Exception as e:
            fails.append((rid, str(e)))
            print(f"  FAIL {rid}: {e}")

    if fails:
        print(f"\n{len(fails)} FAIL(s)")
        return 1

    # count files
    for name, d in out_dirs.items():
        n = len(list(d.glob("*.gjf")))
        expected = len(rxns) if name == "ts" else len(rxns) * 2
        print(f"  {name}: {n} .gjf (expect {expected})")

    total = sum(len(list(d.glob("*.gjf"))) for d in out_dirs.values())
    print(f"  TOTAL: {total} .gjf (expect {len(rxns) * 5})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
