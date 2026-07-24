"""spec22 G22-0 — pilot input generator.

Picks 5 reactions spanning the fragment-size range (seed-fixed,
stratified across both halves) and generates the three ORCA inputs
per reaction:
  eda.inp        — EDA-NOCV on TS complex (CP)
  fragA_opt.inp  — isolated fragment A optimisation
  fragB_opt.inp  — isolated fragment B optimisation

Writes:
  {WORKDIR}/{rid}/{eda,fragA_opt,fragB_opt}/inp
  results/pilot_manifest.csv     — (rid, jobtype, workdir) rows for the array script

Charge / multiplicity read from spec19/results/manifest.pkl.
Fragment atom-index assignment (user-mandated) read from mapping.pkl.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec22_relabel400_wb97x3c"
S19 = REPO / "Ref Comparison/spec19_espley_s2_structures"
MANIFEST = S19 / "results/manifest.pkl"
MAPPING = S19 / "results/mapping.pkl"
STRUCTS = S19 / "structures"

WORKDIR = Path("/gpfs/tmp_cpu2/yeseo1ee/spec22_wb97x3c_workdir")
OUT_PILOT_MANIFEST = STAGE / "results/pilot_manifest.csv"
BUILD_LOG = STAGE / "logs/build.log"

# ORCA route lines
EDA_ROUTE = "! wB97X-3c EDA NoSym TightSCF"
EDA_FRAG_ROUTE = "wB97X-3c NoSym TightSCF"
OPT_ROUTE = "! wB97X-3c Opt TightSCF NoSym"

MAXCORE_MB = 3500  # per proc
PILOT_N = 5
PILOT_SEED = 42


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    elems, coords = [], []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        elems.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elems, np.array(coords, dtype=np.float64)


def pick_pilot(mf: pd.DataFrame) -> list:
    """Pick 5 reactions spanning TS atom-count range, stratified across halves."""
    rng = np.random.default_rng(PILOT_SEED)
    ts_n = mf["natoms"].apply(lambda d: d["ts"]).values
    order = np.argsort(ts_n)
    # smallest, largest, and 3 quantile points at 25/50/75
    picks = []
    picks.append(int(order[0]))                                 # smallest
    picks.append(int(order[-1]))                                # largest
    for q in (0.25, 0.5, 0.75):
        idx = int(order[int(round(q * (len(order) - 1)))])
        picks.append(idx)
    # dedup + stratify: ensure at least one from each sub_source
    picks = list(dict.fromkeys(picks))
    subs = set(mf.iloc[picks]["sub_source"].tolist())
    if subs != {"locked_778", "spec16"}:
        # swap in one from the missing half at median
        missing = ({"locked_778", "spec16"} - subs).pop()
        cand = mf[mf["sub_source"] == missing]
        med = int(cand["natoms"].apply(lambda d: d["ts"]).median())
        best = cand.iloc[(cand["natoms"].apply(lambda d: d["ts"]) - med).abs().argsort()].index[0]
        picks[-1] = int(best)
    return picks


def write_eda_input(rid: str, mapping_row: dict, charge_info: dict, ts_path: Path, out: Path):
    """Emit ORCA EDA-NOCV input with (1)/(2) atom-fragment labels."""
    elems, coords = read_xyz(ts_path)
    ts_idx_A = set(int(i) for i in mapping_row["ts_idx_A"])
    ct = int(charge_info["total_charge"])
    ca = int(charge_info["fragment_charge_a"])
    cb = int(charge_info["fragment_charge_b"])
    ma = int(charge_info["fragment_mult_a"])
    mb = int(charge_info["fragment_mult_b"])

    lines = [
        EDA_ROUTE,
        f"%maxcore {MAXCORE_MB}",
        # nprocs=1 for the pilot to isolate ORCA-level issues from MPI issues.
        # Production will re-enable %pal once MPI is validated.
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
    out.write_text("\n".join(lines) + "\n")


def write_opt_input(xyz_path: Path, charge: int, mult: int, out: Path):
    """Emit isolated-fragment optimisation input (no ghost basis)."""
    elems, coords = read_xyz(xyz_path)
    lines = [
        OPT_ROUTE,
        f"%maxcore {MAXCORE_MB}",
        # nprocs=1 for the pilot to isolate ORCA-level issues from MPI issues.
        # Production will re-enable %pal once MPI is validated.
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
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    STAGE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    STAGE.joinpath("results").mkdir(exist_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    with open(BUILD_LOG, "w") as fh:
        _log(fh, "=== spec22 G22-0 pilot input generation ===")
        _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__}")

        mf = pd.read_pickle(MANIFEST)
        mapping = pd.read_pickle(MAPPING)
        _log(fh, f"[load] manifest n={len(mf)}, mapping n={len(mapping)}")

        picks = pick_pilot(mf)
        pilot = mf.iloc[picks].copy().reset_index(drop=True)
        _log(fh, f"[pilot] chose {len(pilot)} reactions:")
        for _, row in pilot.iterrows():
            _log(fh, f"    {row['reaction_id']}  sub={row['sub_source']}  "
                     f"n_ts={row['natoms']['ts']}  n_A={row['natoms']['r_A']}  n_B={row['natoms']['r_B']}")

        rows = []
        for _, row in pilot.iterrows():
            rid = row["reaction_id"]
            rn = int(row["reaction_number"])
            src_dir = STRUCTS / f"rxn_{rn:04d}"

            mp = mapping[int(row["reaction_number"])]
            ci = {
                "total_charge":      row["charge"]["total"],
                "fragment_charge_a": row["charge"]["A"],
                "fragment_charge_b": row["charge"]["B"],
                "fragment_mult_a":   row["mult"]["A"],
                "fragment_mult_b":   row["mult"]["B"],
            }

            for jobtype, mkfn in [
                ("eda", lambda out: write_eda_input(
                    rid, mp, ci, src_dir / "ts.xyz", out)),
                ("fragA_opt", lambda out: write_opt_input(
                    src_dir / "r_A.xyz", int(ci["fragment_charge_a"]), int(ci["fragment_mult_a"]), out)),
                ("fragB_opt", lambda out: write_opt_input(
                    src_dir / "r_B.xyz", int(ci["fragment_charge_b"]), int(ci["fragment_mult_b"]), out)),
            ]:
                jobdir = WORKDIR / rid / jobtype
                jobdir.mkdir(parents=True, exist_ok=True)
                inp_path = jobdir / f"{jobtype}.inp"
                mkfn(inp_path)
                rows.append({
                    "reaction_id": rid,
                    "sub_source":  row["sub_source"],
                    "reaction_number": rn,
                    "jobtype": jobtype,
                    "workdir": str(jobdir),
                    "input":   str(inp_path),
                    "out":     str(jobdir / f"{jobtype}.out"),
                })

        pd.DataFrame(rows).to_csv(OUT_PILOT_MANIFEST, index=False)
        _log(fh, f"[write] {OUT_PILOT_MANIFEST}  n_jobs={len(rows)}")
        _log(fh, "=== pilot inputs OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
