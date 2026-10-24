#!/usr/bin/env python3
"""D1-G3 interim export: every finished Phase O unit ($G3/O/units/<uid>/unit.json, T-5 test units excluded) in one
JSON file, with the final geometry of each unit inlined (sbatch / srun on a compute node only; reads $G3, writes
only analysis/d1_g3/results/<out>).

  python done_json.py [--cutoff 'YYYY-MM-DD HH:MM:SS'] [--out O_done.json]

--cutoff keeps only units whose unit.json was written at or before that time (a fixed snapshot).
Output: {"written", "cutoff", "n_units", "counts": {"<kind>.<method>": {status: n}}, "units": {uid: unit.json +
"unit_dir" + "final_xyz_path" + "final_xyz"}}. Geometries are in label atom order (opt.py docstring).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
G3 = Path(os.environ.get("D1G3_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_g3"))
UNITS = G3 / "O" / "units"


def final_xyz(ud, r):
    """the unit's final geometry file: ts / ref -> <final_dir>/<ts|ref>.xyz, B -> optts/ts.xyz."""
    kind = r.get("kind") or ud.name.split(".")[0]
    if kind == "B":
        p = ud / "optts" / "ts.xyz"
    else:
        p = ud / (r.get("final_dir") or "a1") / ("ts.xyz" if kind == "ts" else "ref.xyz")
    return p if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--out", default="O_done.json")
    a = ap.parse_args()
    cut = dt.datetime.strptime(a.cutoff, "%Y-%m-%d %H:%M:%S").timestamp() if a.cutoff else None
    units, counts = {}, {}
    for uj in sorted(UNITS.glob("*/unit.json")):
        ud = uj.parent
        if ud.name.startswith(("T5.", "L.")):                      # T-5 tests and G3L pilot units (U-7-5)
            continue
        if cut is not None and uj.stat().st_mtime > cut:
            continue
        r = json.loads(uj.read_text())
        p = final_xyz(ud, r)
        r.update(unit_dir=str(ud), final_xyz_path=str(p) if p else None, final_xyz=p.read_text() if p else None)
        units[ud.name] = r
        key = ud.name.split(".")[0] if ud.name.startswith("B.") else ".".join(ud.name.split(".")[:2])
        st = counts.setdefault(key, {})
        st[r.get("status")] = st.get(r.get("status"), 0) + 1
    rep = dict(spec="D1G3.md Phase O interim export (user request 2026-10-25)",
               written=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cutoff=a.cutoff, units_root=str(UNITS),
               n_units=len(units), counts=counts, units=units)
    out = HERE / "results" / Path(a.out).name
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out)
    print(json.dumps({k: rep[k] for k in ("n_units", "counts")}, indent=1), "->", out, flush=True)


if __name__ == "__main__":
    main()
