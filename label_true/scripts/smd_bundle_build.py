#!/usr/bin/env python3
"""Bundle top-3500 rxns (rxn_5973 descending) into 5 folders of 700 each,
each bundle also gets its own copy of the ORCA installation.

Layout produced:
    /gpfs/tmp_cpu2/yeseo1ee/eda_smd_bundles/
        bundle_1/  (rxn_5973 ... rxn_XXXX — 700 rxns)
            inputs/rxn_XXXX/eda.inp
            orca_6_1_1_avx2/  (17 GB ORCA copy)
        bundle_2/  ...
        ...
        bundle_5/
"""
import shutil
import sys
import time
from pathlib import Path

SCRATCH_INPUTS = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
BUNDLE_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_bundles")
ORCA_SRC = Path("/home1/yeseo1ee/orca_6_1_1_avx2")

CHUNK_SIZE = 700
N_CHUNKS = 5
TOTAL = CHUNK_SIZE * N_CHUNKS   # 3500


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # Descending numeric sort of all existing rxn dirs
    all_rxns = sorted(
        (int(d.name.split("_")[1]) for d in SCRATCH_INPUTS.glob("rxn_*")),
        reverse=True,
    )
    if len(all_rxns) < TOTAL:
        sys.exit(f"Not enough rxns: {len(all_rxns)} < {TOTAL}")
    top = all_rxns[:TOTAL]
    log(f"Top 3500 rxns: rxn_{top[0]:04d} → rxn_{top[-1]:04d}")

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    for k in range(N_CHUNKS):
        bundle = BUNDLE_ROOT / f"bundle_{k+1}"
        inputs_dir = bundle / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        chunk = top[k*CHUNK_SIZE:(k+1)*CHUNK_SIZE]
        log(f"bundle_{k+1}: rxn_{chunk[0]:04d} ... rxn_{chunk[-1]:04d} ({len(chunk)} rxns)")

        # Copy each rxn dir (just the eda.inp file — that's all scratch has for pending rxns)
        for rid in chunk:
            src = SCRATCH_INPUTS / f"rxn_{rid:04d}"
            dst = inputs_dir / f"rxn_{rid:04d}"
            if dst.exists():
                continue
            shutil.copytree(src, dst)
        log(f"  {len(chunk)} rxn folders copied to {inputs_dir}")

        # Copy ORCA installation into the bundle (17 GB)
        orca_dst = bundle / "orca_6_1_1_avx2"
        if orca_dst.exists():
            log(f"  ORCA already exists at {orca_dst}, skipping")
        else:
            log(f"  copying ORCA (17 GB) to {orca_dst} ...")
            t0 = time.time()
            shutil.copytree(ORCA_SRC, orca_dst, symlinks=True)
            log(f"  ORCA copied in {time.time()-t0:.1f}s")

    log("done. summary:")
    for k in range(N_CHUNKS):
        bundle = BUNDLE_ROOT / f"bundle_{k+1}"
        n_rxn = sum(1 for _ in (bundle / "inputs").glob("rxn_*"))
        sz = subprocess_check_du(bundle)
        log(f"  bundle_{k+1}: {n_rxn} rxns, size {sz}")


def subprocess_check_du(path):
    import subprocess
    try:
        return subprocess.check_output(["du", "-sh", str(path)]).split()[0].decode()
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
