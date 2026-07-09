"""Generate ORCA optimization inputs for the strain pass.

For each of the 776 completed EDA reactions, extract fragA and fragB atom
coordinates from the eda.inp and produce two ORCA opt inputs:

  outputs/orca_strain/inputs/<rid>__fA/opt.inp
  outputs/orca_strain/inputs/<rid>__fB/opt.inp

Each opt.inp requests geometry optimization from the TS-frozen fragment
coordinates. The final SP energy (`FINAL SINGLE POINT ENERGY`) is the
relaxed-fragment energy used to compute strain:

  E_strain_A = E(fragA at TS geom, from EDA supermolecule) - E(fragA relaxed)
"""
from __future__ import annotations
import re
from pathlib import Path

from ase.data import chemical_symbols  # unused but re-exports symbol table

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
IN_ROOT = REPO / "outputs/orca_eda/inputs"
OUT_ROOT = REPO / "outputs/orca_strain/inputs"

ATOM_RE = re.compile(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")


def _parse_eda_inp(path: Path):
    """Return (total_charge, fragA_charge, fragA_mult, fragB_charge, fragB_mult,
    A_atoms, B_atoms) where A_atoms/B_atoms are (symbol, x, y, z) tuples."""
    text = path.read_text()
    total = 0; qa = 0; ma = 1; qb = 0; mb = 1
    in_xyz = False
    A, B = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("FRAG1_C"):
            qa = int(s.split()[-1])
        elif s.startswith("FRAG1_M"):
            ma = int(s.split()[-1])
        elif s.startswith("FRAG2_C"):
            qb = int(s.split()[-1])
        elif s.startswith("FRAG2_M"):
            mb = int(s.split()[-1])
        elif s.startswith("* xyz"):
            parts = s.split()
            total = int(parts[2])
            in_xyz = True
            continue
        elif in_xyz and s == "*":
            break
        elif in_xyz:
            m = ATOM_RE.match(line)
            if m:
                sym, fid, x, y, z = m.groups()
                atom = (sym, float(x), float(y), float(z))
                if fid == "1": A.append(atom)
                else: B.append(atom)
    return total, qa, ma, qb, mb, A, B


def _render_opt_input(atoms, charge, mult, maxcore=3500):
    """OPT + TightSCF, serial ORCA (no %pal, no MPI on this cluster).
    Loose_Opt keeps runtime bounded even for large flexible fragments."""
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym Opt TightSCF SlowConv KDIIS",
        f"%maxcore {maxcore}",
        "%scf",
        "  MaxIter 300",
        "end",
        "%geom",
        "  MaxIter 100",
        "end",
        f"* xyz {int(charge)} {int(mult)}",
    ]
    for sym, x, y, z in atoms:
        lines.append(f"{sym:<3s}  {x:15.8f} {y:15.8f} {z:15.8f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    for rid_dir in sorted(IN_ROOT.iterdir()):
        if not rid_dir.is_dir() or "broken" in rid_dir.name:
            continue
        eda_inp = rid_dir / "eda.inp"
        if not eda_inp.exists():
            continue
        try:
            total, qa, ma, qb, mb, A, B = _parse_eda_inp(eda_inp)
            if not A or not B:
                n_err += 1; continue
            outA = OUT_ROOT / f"{rid_dir.name}__fA"
            outB = OUT_ROOT / f"{rid_dir.name}__fB"
            outA.mkdir(parents=True, exist_ok=True)
            outB.mkdir(parents=True, exist_ok=True)
            (outA / "opt.inp").write_text(_render_opt_input(A, qa, ma))
            (outB / "opt.inp").write_text(_render_opt_input(B, qb, mb))
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {rid_dir.name}: {exc}", flush=True)

    print(f"done: {n_ok} reactions × 2 fragments = {n_ok*2} opt inputs (+ {n_err} errors)")
    print(f"output: {OUT_ROOT}")

    # Also write the manifest for the runner
    manifest = OUT_ROOT.parent / "manifest.txt"
    entries = sorted(d.name for d in OUT_ROOT.iterdir() if d.is_dir())
    manifest.write_text("\n".join(entries) + "\n")
    print(f"manifest: {len(entries)} entries → {manifest}")


if __name__ == "__main__":
    main()
