#!/usr/bin/env python3
"""val_determinism.py — V1a / V1b (VALIDATION_SPEC §4, A6 a/b): same ORCA input, same result?

  python val_determinism.py --mode v1a --src J08 --task V1a_J08
    the pilot job's 5 SP inputs, byte-identical, re-run in a new folder (5 serial ORCA processes in
    parallel, as run_sp); the 7 outputs' FINAL SINGLE POINT ENERGY and the 9 method-① labels are
    compared with the pilot's.
  python val_determinism.py --mode v1b --src J08 --task V1b_J08
    eda.inp only, with %pal nprocs 4 (pilot: 1); eda / eda_frag1 / eda_frag2 energies compared.
The pilot outputs are read from a copy (<scratch>/v1/<task>/pilot_sp), never in place.
Writes <scratch>/v1/<task>/result.json. Idempotent (ORCA skipped for outputs that terminated normally).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402

SP = ("eda", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")
OUTS = ("eda", "eda_frag1", "eda_frag2", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")
TARGETS = ("barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft")


def orca_ok(p: Path) -> bool:
    return p.is_file() and "ORCA TERMINATED NORMALLY" in p.read_text(errors="replace")


def run_parallel(wd: Path, names):
    todo = [n for n in names if not orca_ok(wd / f"{n}.out")]
    procs = []
    for n in todo:
        for f in wd.glob(f"{n}*"):
            if f.suffix != ".inp":
                f.unlink()
        fo, fe = open(wd / f"{n}.out", "w"), open(wd / f"{n}.err", "w")
        procs.append(subprocess.Popen([os.environ["ORCA_BIN"], f"{n}.inp"], cwd=wd, stdout=fo, stderr=fe))
    for p in procs:
        p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("v1a", "v1b"), required=True)
    ap.add_argument("--src", required=True); ap.add_argument("--task", required=True)
    a = ap.parse_args()
    cfg = pc.load_config()
    base = Path(cfg["scratch"]) / "v1" / a.task
    if (base / "result.json").is_file():
        print(f"{a.task}: done"); return
    _, s3 = pc.import_repo_modules(cfg)
    src = Path(cfg["pilot_scratch"]) / "jobs" / a.src / "sp"
    ref = base / "pilot_sp"                        # read-only pilot outputs -> copy
    if not ref.is_dir():
        ref.mkdir(parents=True)
        for f in list(src.glob("*.inp")) + list(src.glob("*.out")) + [src / "input_meta.json"]:
            shutil.copy(f, ref / f.name)
    wd = base / "run"; wd.mkdir(parents=True, exist_ok=True)
    res = dict(task=a.task, mode=a.mode, src_job=a.src)
    if a.mode == "v1a":
        for n in SP:
            if not (wd / f"{n}.inp").is_file():
                shutil.copy(ref / f"{n}.inp", wd / f"{n}.inp")
        res["inputs_identical"] = all((wd / f"{n}.inp").read_bytes() == (ref / f"{n}.inp").read_bytes() for n in SP)
        run_parallel(wd, SP)
        names = OUTS
    else:
        if not (wd / "eda.inp").is_file():
            t = re.sub(r"%pal nprocs \d+ end", "%pal nprocs 4 end", (ref / "eda.inp").read_text())
            (wd / "eda.inp").write_text(t)
        res["eda_pal"] = re.search(r"%pal nprocs \d+ end", (wd / "eda.inp").read_text()).group(0)
        run_parallel(wd, ("eda",))
        names = ("eda", "eda_frag1", "eda_frag2")
    bad = [n for n in names if not orca_ok(wd / f"{n}.out")]
    if bad:
        res["error"] = f"not terminated normally: {bad}"
        (base / "result.json").write_text(json.dumps(res, indent=1)); sys.exit(1)
    dE = {n: s3.read_fspe(wd / f"{n}.out", 1) - s3.read_fspe(ref / f"{n}.out", 1) for n in names}
    res.update(dE_eh=dE, max_abs_dE_eh=max(abs(x) for x in dE.values()))
    if a.mode == "v1a":
        meta = json.loads((ref / "input_meta.json").read_text())
        new, old = s3.parse_one_rxn(wd, meta), s3.parse_one_rxn(ref, meta)
        res["status"] = (new.get("status"), old.get("status"))
        dl = {t: new[t] - old[t] for t in TARGETS if t in new and t in old}
        res.update(dlabel_kcal=dl, max_abs_dlabel_kcal=max((abs(x) for x in dl.values()), default=None),
                   eda_table_identical=new.get("eda_raw_eh") == old.get("eda_raw_eh"))
    (base / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
