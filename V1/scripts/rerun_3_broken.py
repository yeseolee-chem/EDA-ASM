"""Re-emit ORCA inputs for nme2, oh, no2 with hardened settings (spec §9 retry).

Differences from default Phase 1 inputs:
  ts_scan.inp: range 4.0 -> 1.8 Å (23 pts) instead of 3.0 -> 1.9 Å (12 pts);
               MaxIter 200 for inner geom opt (default ~50)
  ts.inp:      Calc_Hess true, Recalc_Hess 3, TS_Mode {B C1 C6} end
               so OptTS Hessian-follows the C1-C6 stretch (the [3,3] coord)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"

SCAN_TMPL_V2 = """! wB97X-3c Opt TightSCF
%pal nprocs 8 end
%maxcore 4000
%geom
  MaxIter 200
  Scan B {C1} {C6} = 4.0, 1.8, 23 end
end
* xyzfile 0 1 reactant.xyz
"""

TS_TMPL_V2 = """! wB97X-3c OptTS Freq TightSCF
%pal nprocs 8 end
%maxcore 4000
%geom
  Calc_Hess true
  Recalc_Hess 3
  TS_Mode {{B {C1} {C6}}} end
  MaxIter 200
end
* xyzfile 0 1 ts_guess.xyz
"""


def rewrite_inputs(rxn_id: str) -> None:
    am = json.loads((RUNS / rxn_id / "build" / "atom_map.json").read_text())
    C1 = am["core_indices"]["C1"]
    C6 = am["core_indices"]["C6"]
    orca = RUNS / rxn_id / "orca"

    # Archive old results before overwriting
    archive = orca / "v1_attempt1"
    archive.mkdir(exist_ok=True)
    for name in ("ts_scan.out", "ts_scan.err", "ts.out", "ts.err",
                 "ts_guess.xyz", "ts.xyz", "ts_scan.inp", "ts.inp"):
        src = orca / name
        if src.exists():
            shutil.move(str(src), str(archive / name))
    # Move per-step scan xyz files too (each ~1 KB)
    for p in orca.glob("ts_scan.0*.xyz"):
        shutil.move(str(p), str(archive / p.name))
    # Clear stage sentinels (keep .done_reactant — reactant is fine)
    for n in (".done_scan", ".done_ts", ".done_orca"):
        s = orca / n
        if s.exists():
            s.unlink()

    (orca / "ts_scan.inp").write_text(SCAN_TMPL_V2.format(C1=C1, C6=C6))
    (orca / "ts.inp").write_text(TS_TMPL_V2.format(C1=C1, C6=C6))
    print(f"  {rxn_id}: archived to v1_attempt1/, new ts_scan + ts inputs written")


if __name__ == "__main__":
    for rxn in ("nme2", "oh", "no2"):
        rewrite_inputs(rxn)
    print("done.")
