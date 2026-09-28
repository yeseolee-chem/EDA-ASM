#!/usr/bin/env python3
"""Migrate 5 bundles from extracted-ORCA (17 GB each) to tar.xz form (448 MB each).

For each bundle_{1..5} under eda_smd_relabel/inputs/:
    1. Remove extracted orca_6_1_1_avx2/  (17 GB freed)
    2. Copy the ORCA tar.xz into the bundle root  (448 MB)
    3. Rewrite launch.sh with an auto-extract preamble so `bash launch.sh`
       still remains the single command the target user runs.
"""
import os
import shutil
import time
from pathlib import Path

INPUTS = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
TARXZ_SRC = Path("/home1/yeseo1ee/projects/eda-asm-prediction/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz")
TARXZ_NAME = TARXZ_SRC.name    # 원본 그대로 유지 (풀면 이름 그대로 폴더 생성됨)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


LAUNCH_SH = r"""#!/bin/bash
# One-command launcher: extract ORCA (first time only) + submit 10-way SLURM array.
# Usage:  bash launch.sh
# Requirements on target: SLURM (sbatch), tar, xz, ~17 GB free after extraction.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------- 1) First-time ORCA extraction (idempotent) ------------
TARXZ="orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz"
EXTRACTED_DIR="orca_6_1_1_linux_x86-64_shared_openmpi418_avx2"
ORCA_DIR="orca_6_1_1_avx2"

if [ ! -x "$ORCA_DIR/orca" ]; then
    if [ ! -f "$TARXZ" ]; then
        echo "ERROR: Neither $ORCA_DIR/orca nor $TARXZ exists here." >&2
        exit 2
    fi
    if ! command -v tar >/dev/null || ! command -v xz >/dev/null; then
        echo "ERROR: 'tar' and 'xz' required for first-time setup." >&2
        exit 3
    fi
    echo "First-time setup: extracting ORCA (about 5 min, 17 GB) ..."
    t0=$(date +%s)
    tar -xJf "$TARXZ"
    if [ -d "$EXTRACTED_DIR" ] && [ ! -d "$ORCA_DIR" ]; then
        mv "$EXTRACTED_DIR" "$ORCA_DIR"
    fi
    if [ ! -x "$ORCA_DIR/orca" ]; then
        echo "ERROR: ORCA binary not found at $ORCA_DIR/orca after extraction" >&2
        exit 4
    fi
    echo "  ORCA extracted in $(( $(date +%s) - t0 ))s."
fi

# ---------- 2) Sanity checks ---------------------------------------
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' not found. Target must be a SLURM cluster." >&2; exit 1
fi
[ -d ./inputs ] && [ "$(ls -1 ./inputs | grep -c '^rxn_')" -gt 0 ] || { echo "ERROR: no rxn dirs in ./inputs/" >&2; exit 5; }
[ -f ./worker.sh ] || { echo "ERROR: worker.sh missing" >&2; exit 6; }
mkdir -p logs state/done state/claim state/timing

N_RXN=$(ls -1 ./inputs | grep -c '^rxn_')
N_DONE=$(ls -1 ./state/done 2>/dev/null | wc -l)
N_PENDING=$((N_RXN - N_DONE))
echo "Bundle: $(pwd)"
echo "  rxns: $N_RXN  done: $N_DONE  pending: $N_PENDING"
if [ $N_PENDING -le 0 ]; then
    echo "All reactions already done. Nothing to submit."
    exit 0
fi

# ---------- 3) Verify SBATCH block, then submit --------------------
if ! sbatch --test-only ./worker.sh 2>/tmp/sbatch_err_$$; then
    echo "ERROR: sbatch rejected worker.sh:" >&2
    cat /tmp/sbatch_err_$$ >&2
    echo "" >&2
    echo "Fix hint: edit --partition= or --account= inside worker.sh header." >&2
    rm -f /tmp/sbatch_err_$$; exit 7
fi
rm -f /tmp/sbatch_err_$$

JID=$(sbatch --parsable ./worker.sh)
echo ""
echo "Submitted array job: $JID (10 workers × 48h walltime)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls state/done | wc -l          # progress out of $N_RXN"
echo "  tail -f logs/wrk_${JID}_0.out  # worker 0 log"
"""


README_TAIL = """
## First-time execution (auto-install)

`bash launch.sh` on a fresh target cluster automatically:

1. Extracts `orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz`
   into `orca_6_1_1_avx2/` (~5 min, 17 GB).
2. Verifies SBATCH block, submits 10-element SLURM array (48h walltime each).

On subsequent runs: extraction is skipped (idempotent). Just resume with
another `bash launch.sh` if some workers were wall-clipped.

Transferred bundle size: **~500 MB** (was 17 GB when extracted).
"""


def main():
    bundles = sorted(INPUTS.glob("bundle_*"))
    if len(bundles) != 5:
        raise SystemExit(f"Expected 5 bundles, found {len(bundles)}")

    for b in bundles:
        log(f"=== {b.name} ===")
        # 1) delete extracted orca dir
        odir = b / "orca_6_1_1_avx2"
        if odir.exists():
            log(f"  rm -rf {odir.name}/  (17 GB)")
            t0 = time.time()
            shutil.rmtree(odir)
            log(f"  removed in {time.time()-t0:.1f}s")
        # 2) copy tar.xz into bundle
        tarxz_dst = b / TARXZ_NAME
        if tarxz_dst.exists():
            log(f"  tar.xz already present, skipping copy")
        else:
            log(f"  cp tar.xz ({TARXZ_SRC.stat().st_size / 1e6:.0f} MB) → {b.name}/")
            t0 = time.time()
            shutil.copy2(TARXZ_SRC, tarxz_dst)
            log(f"  copied in {time.time()-t0:.1f}s")
        # 3) rewrite launch.sh
        (b / "launch.sh").write_text(LAUNCH_SH)
        os.chmod(b / "launch.sh", 0o755)
        log(f"  updated launch.sh (auto-extract)")
        # 3b) fix worker.sh: use SLURM_SUBMIT_DIR instead of BASH_SOURCE
        # (SLURM copies scripts to spool dir; dirname of BASH_SOURCE points there, not to bundle)
        ws = b / "worker.sh"
        if ws.exists():
            txt = ws.read_text()
            old = 'BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)'
            new = 'BUNDLE="${SLURM_SUBMIT_DIR:-$(pwd)}"'
            if old in txt and new not in txt:
                ws.write_text(txt.replace(old, new))
                log(f"  patched worker.sh (SLURM_SUBMIT_DIR fix)")
            elif new in txt:
                log(f"  worker.sh already patched")
            else:
                log(f"  WARN: worker.sh BUNDLE line not found — manual check needed")
        # 4) append to README
        rd = b / "README.md"
        if rd.exists() and "First-time execution" not in rd.read_text():
            rd.write_text(rd.read_text() + README_TAIL)
            log(f"  appended install note to README.md")

    log("\ndone. bundle sizes:")
    for b in bundles:
        sz = shutil._ntuple_diskusage.__module__  # dummy to keep import clean
        import subprocess
        s = subprocess.check_output(["du", "-sh", str(b)]).split()[0].decode()
        n = sum(1 for d in (b / "inputs").glob("rxn_*") if d.is_dir())
        log(f"  {b.name}: {n} rxns, {s}")


if __name__ == "__main__":
    main()
