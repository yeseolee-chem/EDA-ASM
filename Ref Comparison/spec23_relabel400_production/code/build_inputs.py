"""spec23 build_inputs — generate 1200 ORCA input directories for the full cohort.

Reads:
  spec19/results/manifest.pkl        — charge, mult, ts_idx_A per reaction
  spec19/structures/rxn_XXXX/*.xyz   — geometries (unchanged since spec19)

Writes:
  {WORKDIR}/{rid}/{eda,fragA_opt,fragB_opt}/
    {jobtype}.inp        — ORCA input
    meta.json            — sha256 of each input geometry consumed,
                          route line, ORCA version tag, nprocs, maxcore
  results/job_manifest.csv     — (rid, sub_source, reaction_number, jobtype,
                                 workdir, input, out) per row
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec23_relabel400_production"
S19 = REPO / "Ref Comparison/spec19_espley_s2_structures"
MANIFEST = S19 / "results/manifest.pkl"
STRUCTS = S19 / "structures"

WORKDIR = Path("/gpfs/tmp_cpu2/yeseo1ee/spec23_wb97x3c_workdir")
OUT_JOB_MANIFEST = STAGE / "results/job_manifest.csv"
BUILD_LOG = STAGE / "logs/build.log"

# Route lines — SAME for every reaction (G23-C).
EDA_ROUTE = "! wB97X-3c EDA NoSym TightSCF"
EDA_FRAG_ROUTE = "wB97X-3c NoSym TightSCF"
OPT_ROUTE = "! wB97X-3c Opt TightSCF NoSym"

NPROCS = 4
MAXCORE_MB = 3500
ORCA_VERSION = "6.1.1"


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    elems, coords = [], []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        elems.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elems, np.array(coords, dtype=np.float64)


def eda_input(ts_path: Path, ts_idx_A: set, ca: int, ma: int, cb: int, mb: int, ct: int) -> str:
    elems, coords = read_xyz(ts_path)
    lines = [
        EDA_ROUTE,
        f"%maxcore {MAXCORE_MB}",
        f"%pal nprocs {NPROCS} end",
        "%scf",
        "  MaxIter 500",
        "end",
        "%eda",
        f'  FRAG1 "{EDA_FRAG_ROUTE}"',
        f'  FRAG2 "{EDA_FRAG_ROUTE}"',
        f"  FRAG1_C {ca}",
        f"  FRAG1_M {ma}",
        f"  FRAG2_C {cb}",
        f"  FRAG2_M {mb}",
        "end",
        f"* xyz {ct} 1",
    ]
    for i, (e, xyz) in enumerate(zip(elems, coords)):
        tag = "(1)" if i in ts_idx_A else "(2)"
        lines.append(f"  {e:2s}{tag}  {xyz[0]:>16.8f}  {xyz[1]:>16.8f}  {xyz[2]:>16.8f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def opt_input(xyz_path: Path, charge: int, mult: int) -> str:
    elems, coords = read_xyz(xyz_path)
    lines = [
        OPT_ROUTE,
        f"%maxcore {MAXCORE_MB}",
        f"%pal nprocs {NPROCS} end",
        "%scf",
        "  MaxIter 500",
        "end",
        "%geom",
        "  MaxIter 200",
        "end",
        f"* xyz {charge} {mult}",
    ]
    for e, xyz in zip(elems, coords):
        lines.append(f"  {e:2s}  {xyz[0]:>16.8f}  {xyz[1]:>16.8f}  {xyz[2]:>16.8f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def write_meta(out_dir: Path, jobtype: str, geom_paths: list, route: str, charge: int, mult: int):
    meta = {
        "jobtype": jobtype,
        "orca_version": ORCA_VERSION,
        "route_line": route,
        "nprocs": NPROCS,
        "maxcore_mb": MAXCORE_MB,
        "charge": charge,
        "multiplicity": mult,
        "geometry_sha256": {str(p.name): _sha(p) for p in geom_paths},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    STAGE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    STAGE.joinpath("results").mkdir(exist_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    with open(BUILD_LOG, "w") as fh:
        _log(fh, "=== spec23 build_inputs ===")
        _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__}")

        mf = pd.read_pickle(MANIFEST)
        _log(fh, f"[load] manifest n={len(mf)}")

        rows = []
        for _, row in mf.iterrows():
            rid = row["reaction_id"]
            rn = int(row["reaction_number"])
            sub = row["sub_source"]
            src = STRUCTS / f"rxn_{rn:04d}"

            ts_idx_A = set(int(i) for i in row["ts_idx_A"])
            ct = int(row["charge"]["total"])
            ca = int(row["charge"]["A"])
            cb = int(row["charge"]["B"])
            ma = int(row["mult"]["A"])
            mb = int(row["mult"]["B"])

            # EDA
            d = WORKDIR / rid / "eda"; d.mkdir(parents=True, exist_ok=True)
            (d / "eda.inp").write_text(eda_input(src / "ts.xyz", ts_idx_A, ca, ma, cb, mb, ct))
            write_meta(d, "eda", [src / "ts.xyz"], EDA_ROUTE, ct, 1)
            rows.append({"reaction_id": rid, "sub_source": sub, "reaction_number": rn,
                          "jobtype": "eda", "workdir": str(d),
                          "input": str(d / "eda.inp"), "out": str(d / "eda.out")})

            # fragA_opt
            d = WORKDIR / rid / "fragA_opt"; d.mkdir(parents=True, exist_ok=True)
            (d / "fragA_opt.inp").write_text(opt_input(src / "r_A.xyz", ca, ma))
            write_meta(d, "fragA_opt", [src / "r_A.xyz"], OPT_ROUTE, ca, ma)
            rows.append({"reaction_id": rid, "sub_source": sub, "reaction_number": rn,
                          "jobtype": "fragA_opt", "workdir": str(d),
                          "input": str(d / "fragA_opt.inp"), "out": str(d / "fragA_opt.out")})

            # fragB_opt
            d = WORKDIR / rid / "fragB_opt"; d.mkdir(parents=True, exist_ok=True)
            (d / "fragB_opt.inp").write_text(opt_input(src / "r_B.xyz", cb, mb))
            write_meta(d, "fragB_opt", [src / "r_B.xyz"], OPT_ROUTE, cb, mb)
            rows.append({"reaction_id": rid, "sub_source": sub, "reaction_number": rn,
                          "jobtype": "fragB_opt", "workdir": str(d),
                          "input": str(d / "fragB_opt.inp"), "out": str(d / "fragB_opt.out")})

        pd.DataFrame(rows).to_csv(OUT_JOB_MANIFEST, index=False)
        _log(fh, f"[write] {OUT_JOB_MANIFEST}  n_jobs={len(rows)}")
        _log(fh, "=== build_inputs OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
