#!/usr/bin/env python3
"""SPEC18 Step 2 — xTB reaction-path search (GFN2-xTB `--path`).

For each rxn: feed R and P (both in TS atom order, from
`reactant_complex/rxn_XXXX.npz`) → xtb `--path` → parse the emitted
`xtbpath_ts.xyz` as the xTB TS guess. Save as `ts_xtb.npy`.

Sharding: pass `--shard K --nshard N` to process `ids[K::N]`. Each
shard writes its own per-shard CSV; `02b_merge_shards.py` combines
them into `xtb_path_{MODE}.csv`.

Idempotent: existing `ts_xtb.npy` short-circuits the run.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"

XTB = shutil.which("xtb") or "xtb"
GFN = int(os.environ.get("XTB_GFN", "2"))
THREADS = int(os.environ.get("XTB_THREADS", "4"))
TIMEOUT = int(os.environ.get("XTB_TIMEOUT_S", "600"))

# xTB path-validity thresholds (SPEC18 audit 2026-09-13).
# The path finder reports MANY trial runs and picks one. If the taken run
# never actually reaches the product (large product-end RMSD) or lands a
# non-physical barrier, the emitted xtbpath_ts.xyz is garbage. Reject
# these BEFORE saving `ts_xtb.npy` so Step 4 correctly falls back to
# (R+P)/2 instead of poisoning the training data.
#
# Coley 5,269 rxns DFT G_act: min 0.51, median 19.81, p95 38.28, max 75.80
# xTB error vs DFT is typically ±15 kcal/mol; BARRIER_MAX = 100 gives
# ~25 kcal/mol margin above the empirical DFT ceiling.
PROD_RMSD_MAX = float(os.environ.get("PROD_RMSD_MAX", "0.5"))   # Å
BARRIER_MIN   = float(os.environ.get("BARRIER_MIN", "0.0"))     # kcal/mol
BARRIER_MAX   = float(os.environ.get("BARRIER_MAX", "100.0"))   # kcal/mol

# Locked path.inp — SPEC §3 rule: identical parameters across all rxns.
PATH_INP = """$path
   nrun=1
   npoint=25
   anopt=10
   kpush=0.003
   kpull=-0.015
   ppull=0.05
   alp=1.2
$end
"""


def write_xyz(path: Path, syms, xyz, comment: str = "") -> None:
    lines = [str(len(syms)), comment]
    for s, (x, y, z) in zip(syms, xyz):
        lines.append(f"{s:<3}{x:16.8f}{y:16.8f}{z:16.8f}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz(path: Path):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def parse_log(text: str) -> dict:
    """Extract barrier/gradient/TS-write info from xtb `--path` stdout.

    Also parses the per-trial `path trials` table so callers can verify
    the run xtb actually took reached the product side.
    """
    out = dict(
        fwd_barrier=np.nan, bwd_barrier=np.nan, dE=np.nan,
        grad_norm=np.nan, ts_point=-1, ts_written=False,
        terminated_normally=False,
        # Per-trial path validity (audit-added)
        n_trials=0, n_reached=0, taken_run=-1,
        taken_barrier=np.nan, taken_dE=np.nan, taken_prod_rmsd=np.nan,
    )
    m = re.search(r"forward\s+barrier\s+\(kcal\)\s*:\s*(-?[\d.]+)", text)
    if m:
        out["fwd_barrier"] = float(m.group(1))
    m = re.search(r"backward\s+barrier\s+\(kcal\)\s*:\s*(-?[\d.]+)", text)
    if m:
        out["bwd_barrier"] = float(m.group(1))
    m = re.search(r"reaction\s+energy\s+\(kcal\)\s*:\s*(-?[\d.]+)", text)
    if m:
        out["dE"] = float(m.group(1))
    m = re.search(r"norm\(g\) at est\. TS, point:\s*([\d.]+)\s+(\d+)", text)
    if m:
        out["grad_norm"] = float(m.group(1))
        out["ts_point"] = int(m.group(2))
    out["ts_written"] = "estimated TS on file xtbpath_ts.xyz" in text
    out["terminated_normally"] = (
        "normal termination of xtb" in text or out["ts_written"]
    )

    # Per-trial table: `run N  barrier: B  dE: D  product-end path RMSD: R`
    trials = re.findall(
        r"run\s*(\d+)\s+barrier:\s*(-?[\d.]+)\s+dE:\s*(-?[\d.]+)\s+"
        r"product-end path RMSD:\s*([\d.]+)",
        text,
    )
    out["n_trials"] = len(trials)
    tr = {int(r): (float(b), float(d), float(p)) for r, b, d, p in trials}
    # Count runs that actually reached the product side (product-end RMSD < 0.5 Å)
    out["n_reached"] = sum(1 for _, _, p in tr.values() if p < PROD_RMSD_MAX)

    # `path K taken with M points` — this identifies which run xtb committed to
    m = re.search(r"path\s+(\d+)\s+taken with\s+(\d+)\s+points", text)
    if m:
        k = int(m.group(1))
        out["taken_run"] = k
        if k in tr:
            b, d, p = tr[k]
            out["taken_barrier"] = b
            out["taken_dE"] = d
            out["taken_prod_rmsd"] = p
    return out


def run_one(rid: int, work: Path) -> dict:
    """Run xtb --path for a single rxn. Returns row dict."""
    npz_path = PREV / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz"
    if not npz_path.exists():
        return dict(rxn_id=rid, status="npz_missing")
    npz = np.load(npz_path, allow_pickle=True)
    syms = [str(s) for s in npz["syms"]]
    R, P = npz["R"], npz["P"]
    work.mkdir(parents=True, exist_ok=True)
    write_xyz(work / "R.xyz", syms, R, f"rxn {rid} reactant (TS order)")
    write_xyz(work / "P.xyz", syms, P, f"rxn {rid} product (TS order)")
    (work / "path.inp").write_text(PATH_INP)

    env = dict(os.environ,
               OMP_NUM_THREADS=str(THREADS),
               MKL_NUM_THREADS=str(THREADS),
               OMP_STACKSIZE="1G")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [XTB, "R.xyz", "--path", "P.xyz", "--input", "path.inp",
             "--gfn", str(GFN), "--chrg", "0", "--uhf", "0"],
            cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        (work / "xtb.log").write_text(log)
        info = parse_log(log)
        info["wall_s"] = time.time() - t0
        info["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        (work / "xtb.log").write_text("TIMEOUT")
        return dict(rxn_id=rid, status="timeout", wall_s=TIMEOUT)

    ts_file = work / "xtbpath_ts.xyz"
    if not info["ts_written"] or not ts_file.exists():
        return dict(rxn_id=rid, status="no_ts", **info)

    try:
        ts_syms, ts_xyz = read_xyz(ts_file)
    except Exception as e:  # noqa: BLE001
        return dict(rxn_id=rid, status=f"parse_error:{type(e).__name__}", **info)

    if ts_syms != syms or len(ts_xyz) != len(R):
        return dict(rxn_id=rid, status="order_mismatch", **info)
    if not np.isfinite(ts_xyz).all():
        return dict(rxn_id=rid, status="nan_coords", **info)

    # Path-validity gates (audit 2026-09-13): the emitted xtbpath_ts.xyz
    # is only trustworthy if xtb's chosen trial actually connected R to P
    # AND the barrier is physically reasonable. Otherwise the "TS" sits on
    # an unrelated barrier and would poison the guess for OTFM.
    taken_prod = info.get("taken_prod_rmsd", float("nan"))
    if np.isfinite(taken_prod) and taken_prod > PROD_RMSD_MAX:
        return dict(rxn_id=rid, status="path_not_connected", **info)
    barrier = info.get("fwd_barrier", float("nan"))
    if np.isfinite(barrier) and not (BARRIER_MIN <= barrier <= BARRIER_MAX):
        return dict(rxn_id=rid, status="barrier_unphysical", **info)
    taken_dE = info.get("taken_dE", float("nan"))
    if np.isfinite(taken_dE) and taken_dE > 5.0:
        # Coley reactions are exothermic ([3+2] cycloaddition); dE > +5 kcal
        # on the chosen path means the trial ended in a non-product basin.
        return dict(rxn_id=rid, status="dE_sign_wrong", **info)

    np.save(work / "ts_xtb.npy", ts_xyz)
    return dict(rxn_id=rid, status="ok", **info)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=os.environ.get("XTB_MODE", "pilot"),
                    choices=["pilot", "full"])
    ap.add_argument("--shard", type=int, default=0,
                    help="0-based shard index")
    ap.add_argument("--nshard", type=int, default=1,
                    help="total number of shards")
    args = ap.parse_args()

    if args.mode == "pilot":
        ids = pd.read_csv(BASE / "artifacts" / "pilot_sample.csv").rxn_id.tolist()
    else:
        # Use coley_all.pkl rxn_id list (== 5,261 rxns that pass all gates).
        import pickle
        with open(PREV / "data" / "coley_all.pkl", "rb") as f:
            pkl = pickle.load(f)
        ids = list(pkl["rxn_id"])

    ids = sorted(ids)
    my_ids = ids[args.shard::args.nshard]
    print(f"[{args.mode}] shard {args.shard}/{args.nshard}: "
          f"{len(my_ids)} rxns  (total {len(ids)})", flush=True)

    rows = []
    for i, rid in enumerate(my_ids):
        work = BASE / "xtb_runs" / f"rxn_{rid:04d}"
        done = work / "ts_xtb.npy"
        if done.exists():
            rows.append(dict(rxn_id=rid, status="ok", reused=True))
            continue
        rows.append(run_one(rid, work))
        if (i + 1) % 25 == 0:
            n_ok = sum(1 for r in rows if r.get("status") == "ok")
            print(f"  {args.shard}: {i+1}/{len(my_ids)}  ok={n_ok}", flush=True)

    D = pd.DataFrame(rows)
    out = BASE / "artifacts" / f"xtb_path_{args.mode}_shard{args.shard:02d}.csv"
    D.to_csv(out, index=False)
    n_ok = int((D.status == "ok").sum())
    counts = D.status.value_counts().to_dict()
    print(f"shard {args.shard}: ok={n_ok}/{len(D)}  ->  {out}")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
