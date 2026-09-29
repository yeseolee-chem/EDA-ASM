#!/usr/bin/env python3
"""val_grad.py — V2a (VALIDATION_SPEC §4, §6.3): ORCA gradient at Coley's TS, levels L0 / L1 / L2.

  python val_grad.py --rxn 20 --task V2a_0020
EnGrad variant of each level (orca_direct.level_header(engrad=True)): `OptTS Freq` -> `EnGrad`, no %geom
blocks. Reports max|g| and RMS g (Eh/bohr) from <level>.engrad and the §6.3 output checks (VWN string,
RIJCOSX state, D3(BJ) parameters, SMD CDS and eps) per level.
FIX F-B: L0/L1 without RIJCOSX in the output, or any level without a VWN string, is a parser defect:
result_error.json + queue STOP. FIX F-C: result.json is written only on success (it marks the task done);
errors go to result_error.json, which a retry overwrites. Idempotent per level.
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

    def error(msg):          # FIX F-C: result.json only on success; result_error.json is overwritten on a retry
        res["error"] = msg
        (wd / "result_error.json").write_text(json.dumps(res, indent=1))
        print(msg, file=sys.stderr)

    for lv in LEVELS:
        inp = wd / f"{lv}.inp"
        if not inp.is_file():
            od.write_inp(inp, od.level_header(cfg, lv, n_cores=pc.n_cores(), engrad=True), syms, xyz, 0, 1)
        text = od.run(cfg, inp).read_text(errors="replace")
        r = dict(header=od.input_header(inp), checks=od.level_checks(text), hours=od.run_hours(text),
                 terminated=od.terminated(text), energy_eh=od.last_fspe(text))
        r["level_ok"] = lo = od.level_ok(lv, r["checks"])
        res["levels"][lv] = r
        if not (r["terminated"] and (wd / f"{lv}.engrad").is_file()):
            error(f"{lv}: ORCA did not terminate normally or wrote no .engrad"); sys.exit(1)
        _, g = od.parse_engrad(wd / f"{lv}.engrad")
        r.update(max_abs_grad=float(np.abs(g).max()), rms_grad=float(np.sqrt((g ** 2).mean())))
        # FIX F-B: L0/L1 must show RIJCOSX and every level a VWN string, else the level parser is broken
        defect = ([f"RIJCOSX not seen ({r['checks'].get('rijcosx')!r})"] if lv in ("L0", "L1") and lo["rijcosx_on"] is False else []) \
            + (["VWN string not found"] if lo["vwn_ok"] is None else [])
        if defect:
            error(f"{lv}: level parser defect: {'; '.join(defect)}")
            od.stop(f"V2a rxn {a.rxn} {lv}: level parser defect ({'; '.join(defect)}) — FIX F-B positive control")
    d3 = [tuple(sorted(res["levels"][lv]["checks"]["d3"].items())) for lv in LEVELS]
    res["d3_identical_across_levels"] = len(set(d3)) == 1
    (wd / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({lv: {k: res["levels"][lv].get(k) for k in ("max_abs_grad", "rms_grad")} for lv in LEVELS}, indent=1))


if __name__ == "__main__":
    main()
