#!/usr/bin/env python3
"""val_grad.py — V2a (VALIDATION_SPEC §4, §6.3): ORCA gradient at Coley's TS, levels L0 / L1 / L2.

  python val_grad.py --rxn 20 --task V2a_0020
EnGrad variant of each level (orca_direct.level_header(engrad=True)): `OptTS Freq` -> `EnGrad`, no %geom
blocks. Reports max|g| and RMS g (Eh/bohr) from <level>.engrad and the §6.3 output checks (VWN string,
RIJCOSX state, D3(BJ) parameters, SMD CDS and eps) per level.
Writes <scratch>/v2a/<rxn>/result.json. Idempotent per level.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402
import orca_direct as od  # noqa: E402

LEVELS = ("L0", "L1", "L2")


def coley_ts(cfg, rid):
    pdir = Path(cfg["d0_profiles"]) / str(rid)
    f = sorted(p for p in pdir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz")[0]
    return f, pc.read_xyz(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn", type=int, required=True); ap.add_argument("--task", required=True)
    a = ap.parse_args()
    cfg = pc.load_config()
    wd = Path(cfg["scratch"]) / "v2a" / f"{a.rxn:04d}"; wd.mkdir(parents=True, exist_ok=True)
    if (wd / "result.json").is_file():
        print(f"{a.task}: done"); return
    f, (syms, xyz) = coley_ts(cfg, a.rxn)
    res = dict(task=a.task, rxn_id=a.rxn, coley_ts=f.name, n_atoms=len(syms), levels={})
    ok = True
    for lv in LEVELS:
        inp = wd / f"{lv}.inp"
        if not inp.is_file():
            od.write_inp(inp, od.level_header(cfg, lv, n_cores=pc.n_cores(), engrad=True), syms, xyz, 0, 1)
        text = od.run(cfg, inp).read_text(errors="replace")
        r = dict(header=od.input_header(inp), checks=od.level_checks(text), hours=od.run_hours(text),
                 terminated=od.terminated(text), energy_eh=od.last_fspe(text))
        r["level_ok"] = od.level_ok(lv, r["checks"])
        if r["terminated"] and (wd / f"{lv}.engrad").is_file():
            _, g = od.parse_engrad(wd / f"{lv}.engrad")
            r.update(max_abs_grad=float(np.abs(g).max()), rms_grad=float(np.sqrt((g ** 2).mean())))
        else:
            ok = False
        res["levels"][lv] = r
    d3 = [tuple(sorted(res["levels"][lv]["checks"]["d3"].items())) for lv in LEVELS]
    res["d3_identical_across_levels"] = len(set(d3)) == 1
    (wd / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({lv: {k: res["levels"][lv].get(k) for k in ("max_abs_grad", "rms_grad")} for lv in LEVELS}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
