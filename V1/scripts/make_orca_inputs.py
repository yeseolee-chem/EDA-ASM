"""Phase 1 — emit ORCA inputs for a substrate, per V1 Claisen ASR/EDA spec §6.

Generates three ORCA inputs under runs/<id>/orca/:
  reactant.inp  — wB97X-3c Opt + Freq (geometry minimization + harmonic freq)
  ts_scan.inp   — wB97X-3c relaxed scan along C1-C6 from 3.0 to 1.9 Å (12 pts)
  ts.inp        — wB97X-3c OptTS + Freq (TS optimization with Hess recalc)

atom indices for the scan (C1, C6) come from runs/<id>/build/atom_map.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"

REACTANT_TMPL = """! wB97X-3c Opt Freq TightSCF
%pal nprocs {NCPU} end
%maxcore {MAXCORE}
* xyzfile 0 1 ../build/mol.xyz
"""

SCAN_TMPL = """! wB97X-3c Opt TightSCF
%pal nprocs {NCPU} end
%maxcore {MAXCORE}
%geom
  Scan B {C1} {C6} = 3.0, 1.9, 12 end
end
* xyzfile 0 1 reactant.xyz
"""

TS_TMPL = """! wB97X-3c OptTS Freq TightSCF
%pal nprocs {NCPU} end
%maxcore {MAXCORE}
%geom
  Calc_Hess true
  Recalc_Hess 5
end
* xyzfile 0 1 ts_guess.xyz
"""


def make_inputs(rxn_id: str, ncpu: int, maxcore: int) -> Path:
    am_path = RUNS / rxn_id / "build" / "atom_map.json"
    if not am_path.exists():
        raise FileNotFoundError(f"missing atom_map.json — run Phase 0 first: {am_path}")
    am = json.loads(am_path.read_text())
    C1 = am["core_indices"]["C1"]
    C6 = am["core_indices"]["C6"]

    orca_dir = RUNS / rxn_id / "orca"
    orca_dir.mkdir(parents=True, exist_ok=True)

    (orca_dir / "reactant.inp").write_text(
        REACTANT_TMPL.format(NCPU=ncpu, MAXCORE=maxcore)
    )
    (orca_dir / "ts_scan.inp").write_text(
        SCAN_TMPL.format(NCPU=ncpu, MAXCORE=maxcore, C1=C1, C6=C6)
    )
    (orca_dir / "ts.inp").write_text(
        TS_TMPL.format(NCPU=ncpu, MAXCORE=maxcore)
    )
    return orca_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--ncpu", type=int, default=8)
    p.add_argument("--maxcore", type=int, default=4000)
    args = p.parse_args()
    out = make_inputs(args.id, args.ncpu, args.maxcore)
    print(f"inputs written under {out}/")
    for name in ("reactant.inp", "ts_scan.inp", "ts.inp"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
