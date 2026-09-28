#!/usr/bin/env python3
"""Move top-3500 rxns into 5 in-place bundles under eda_smd_relabel/inputs/.

Each bundle also gets:
  - a full ORCA installation copy (17 GB)
  - launch.sh — one-command launcher (bash launch.sh) for target cluster
  - worker.sh — SLURM array (0-9%10), --time=48:00:00, 8 cores/32 GB per element
  - README.md — usage notes

The current chain workers automatically skip rxn dirs no longer present at the
top level of inputs/ (via `[ -d $RUN_DIR ] || continue`), so no cleanup of
outer state is needed.

Layout after:
    /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/
        rxn_0000/  ... rxn_1955/   (remaining ~1765 for local chain)
        bundle_1/
            inputs/rxn_XXXX/eda.inp     (700 rxns, moved here)
            orca_6_1_1_avx2/            (17 GB ORCA copy)
            worker.sh
            launch.sh
            README.md
        bundle_2/  ... bundle_5/
"""
import os
import shutil
import stat
import sys
import time
from pathlib import Path

INPUTS = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
ORCA_SRC = Path("/home1/yeseo1ee/orca_6_1_1_avx2")

CHUNK_SIZE = 700
N_CHUNKS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


WORKER_SH = r"""#!/bin/bash
# SMD-relabel bundle worker (SLURM array, 48h walltime, 10-way).
# One element = one worker. All 10 workers share the 700-rxn pool via
# atomic mkdir-based claim and idempotent "already terminated" skip.
#
# ---- edit the resource block for your target cluster if needed ----
#SBATCH --job-name=smd_bundle
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9%10
#SBATCH --output=logs/wrk_%A_%a.out
# #SBATCH --partition=cpu           # uncomment + set to your partition
# #SBATCH --account=your_account    # uncomment + set to your account
# ------------------------------------------------------------------

set -uo pipefail

BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$BUNDLE"
mkdir -p logs state/done state/claim state/timing

# ---- ORCA runtime ------------------------------------------------
export ORCA_BIN="$BUNDLE/orca_6_1_1_avx2/orca"
if [ ! -x "$ORCA_BIN" ]; then
    echo "FATAL: ORCA binary not found or not executable at $ORCA_BIN" >&2
    exit 2
fi

# Auto-detect OpenMPI 4.1.x. Edit if your cluster keeps it elsewhere.
MPI_ROOT="${MPI_ROOT:-}"
if [ -z "$MPI_ROOT" ]; then
    for cand in \
        /usr/mpi/gcc/openmpi-4.1.8 \
        /usr/mpi/gcc/openmpi-4.1.7a1 \
        /usr/mpi/gcc/openmpi-4.1.7 \
        /opt/openmpi/4.1.8 /opt/openmpi/4.1.7 /opt/openmpi/4.1 \
        /usr/local/openmpi /usr/lib64/openmpi \
        "$HOME/openmpi-4.1.8" "$HOME/openmpi"; do
        if [ -x "$cand/bin/mpirun" ]; then MPI_ROOT="$cand"; break; fi
    done
fi
if [ -n "$MPI_ROOT" ]; then
    export PATH="$MPI_ROOT/bin:$(dirname "$ORCA_BIN"):${PATH:-}"
    export LD_LIBRARY_PATH="$MPI_ROOT/lib64:$MPI_ROOT/lib:$(dirname "$ORCA_BIN"):${LD_LIBRARY_PATH:-}"
    echo "MPI_ROOT=$MPI_ROOT"
else
    export PATH="$(dirname "$ORCA_BIN"):${PATH:-}"
    export LD_LIBRARY_PATH="$(dirname "$ORCA_BIN"):${LD_LIBRARY_PATH:-}"
    echo "WARN: no OpenMPI found; ORCA may fail to launch. Set MPI_ROOT in env or edit worker.sh."
fi

# ORCA MPI stability (single-node, no IB/PSM issues)
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp,self,sm

# ---- work pool ---------------------------------------------------
mapfile -t RXNS < <(ls -1 "$BUNDLE/inputs/" | grep '^rxn_' | sort)
TOTAL=${#RXNS[@]}
if [ $TOTAL -eq 0 ]; then
    echo "FATAL: no rxn dirs found in $BUNDLE/inputs/" >&2; exit 3
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID:-manual}
JOB_START=$(date +%s)
WALL_S=$((48*3600))
SAFETY_S=1800                    # exit cleanly if <30 min left before wall
NODE=$(hostname -s)

echo "=== worker IDX=$IDX JOB=$JOB_ID NODE=$NODE TOTAL=$TOTAL start=$(date -Is) ==="

processed=0; skipped=0; claimed_by_other=0

for OFFSET in $(seq 0 $((TOTAL - 1))); do
    ELAPSED=$(( $(date +%s) - JOB_START ))
    REMAIN=$(( WALL_S - ELAPSED ))
    if [ $REMAIN -lt $SAFETY_S ]; then
        echo "wall guard: ${REMAIN}s < ${SAFETY_S}s — exiting cleanly"
        break
    fi
    N=$(( (IDX + OFFSET) % TOTAL ))
    RID=${RXNS[$N]}
    DONE_FLAG="$BUNDLE/state/done/$RID"
    CLAIM_DIR="$BUNDLE/state/claim/$RID"
    RUN_DIR="$BUNDLE/inputs/$RID"

    [ -f "$DONE_FLAG" ] && { skipped=$((skipped+1)); continue; }
    [ -d "$RUN_DIR" ] || continue
    if [ -f "$RUN_DIR/eda.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$RUN_DIR/eda.out" 2>/dev/null; then
        touch "$DONE_FLAG"; skipped=$((skipped+1)); continue
    fi
    if ! mkdir "$CLAIM_DIR" 2>/dev/null; then
        claimed_by_other=$((claimed_by_other+1)); continue
    fi
    echo "$JOB_ID $IDX $NODE $(date -Is)" > "$CLAIM_DIR/owner"

    T0=$(date +%s)
    cd "$RUN_DIR"
    rm -f eda.out eda.err
    "$ORCA_BIN" eda.inp > eda.out 2> eda.err
    RC=$?
    ELAPSED_RXN=$(( $(date +%s) - T0 ))

    if [ $RC -eq 0 ] && grep -q "ORCA TERMINATED NORMALLY" eda.out; then
        touch "$DONE_FLAG"
        echo "$RID $ELAPSED_RXN $NODE" >> "$BUNDLE/state/timing/w${IDX}.tsv"
        processed=$((processed+1))
        # Cleanup: keep .inp/.out/.err (esp. eda_frag*.out — ORCA auto-generated,
        # needed by downstream parser). Delete only intermediates.
        find . -maxdepth 1 -type f \
            ! -name "*.inp" ! -name "*.out" ! -name "*.err" \
            -delete
        echo "[OK]   $RID  ${ELAPSED_RXN}s  processed=$processed"
    else
        echo "[FAIL] $RID rc=$RC  ${ELAPSED_RXN}s"
    fi
    rm -rf "$CLAIM_DIR" 2>/dev/null || true
    cd "$BUNDLE"
done

echo "=== worker $IDX done processed=$processed skipped=$skipped stolen=$claimed_by_other end=$(date -Is) ==="
"""


LAUNCH_SH = r"""#!/bin/bash
# One-command launcher for the SMD-relabel bundle.
# Usage:  bash launch.sh
# Requirements: SLURM (sbatch) on target cluster.
#
# Submits a 10-element SLURM array with 48h walltime per element.
# Idempotent: if some rxns already have "ORCA TERMINATED NORMALLY" in eda.out,
# workers skip them. Re-run this script anytime to resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' not found in PATH. Run this on a SLURM cluster." >&2
    exit 1
fi

mkdir -p logs state/done state/claim state/timing

# Sanity checks
[ -x ./orca_6_1_1_avx2/orca ] || { echo "ERROR: ORCA binary missing/not executable at ./orca_6_1_1_avx2/orca" >&2; exit 2; }
[ -d ./inputs ] && [ "$(ls -1 ./inputs | grep -c '^rxn_')" -gt 0 ] || { echo "ERROR: no rxn dirs in ./inputs/" >&2; exit 3; }
[ -f ./worker.sh ] || { echo "ERROR: worker.sh missing" >&2; exit 4; }

N_RXN=$(ls -1 ./inputs | grep -c '^rxn_')
N_DONE=$(ls -1 ./state/done 2>/dev/null | wc -l)
N_PENDING=$((N_RXN - N_DONE))
echo "Bundle: $(pwd)"
echo "  rxns: $N_RXN  done: $N_DONE  pending: $N_PENDING"

if [ $N_PENDING -le 0 ]; then
    echo "All reactions already done. Nothing to submit."
    exit 0
fi

# Verify the SBATCH block is acceptable to sbatch (catch missing partition etc.)
if ! sbatch --test-only ./worker.sh 2>/tmp/sbatch_err_$$; then
    echo "ERROR: sbatch --test-only rejected worker.sh:" >&2
    cat /tmp/sbatch_err_$$ >&2
    echo >&2
    echo "Common fix: uncomment/edit --partition= or --account= inside worker.sh" >&2
    rm -f /tmp/sbatch_err_$$
    exit 5
fi
rm -f /tmp/sbatch_err_$$

# Submit
JID=$(sbatch --parsable ./worker.sh)
echo ""
echo "Submitted array job: $JID (10 workers, 48h walltime each)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls state/done | wc -l          # progress out of $N_RXN"
echo "  tail -f logs/wrk_${JID}_0.out  # worker 0 log"
"""


README = """# SMD-relabel bundle {n}

Portable execution package: {count} reactions each requiring one ORCA
EDA-NOCV single point with SMD(water) solvation (auto-generates fragment
SCFs internally).

## Contents

- `inputs/rxn_XXXX/eda.inp` — {count} pre-built ORCA input files
- `orca_6_1_1_avx2/`         — ORCA 6.1.1 shared build (AVX2, OpenMPI 4.1.8)
- `worker.sh`                — SLURM array script (48h walltime, 10-way)
- `launch.sh`                — one-command launcher

## Usage on target SLURM cluster

    bash launch.sh

That's it. The launcher submits a 10-element SLURM array. Each element
runs for up to 48h and dynamically claims pending reactions from a shared
pool. Idempotent — re-run `bash launch.sh` anytime to resume.

## Progress

    ls state/done | wc -l   # done (out of {count})
    squeue -u $USER

## Adjustments for your cluster

Edit `worker.sh` header:

- `--partition=<your_partition>` — uncomment the line if your cluster requires it
- `--account=<your_account>`     — uncomment if needed
- MPI location — auto-detected from common paths; set `MPI_ROOT` env or edit
  the `MPI auto-detect` block if OpenMPI 4.1.x sits elsewhere

## After completion

Each `inputs/rxn_XXXX/` will contain:

- `eda.out`         — the ORCA main output
- `eda_frag1.out`   — ORCA-generated fragment 1 SCF (BSSE-corrected with SMD)
- `eda_frag2.out`   — ORCA-generated fragment 2 SCF
- `eda.err`         — stderr (typically empty)

Transfer results back to the origin cluster's `label_true/work/inputs/rxn_XXXX/`
for the downstream parser (`label_true/scripts/stage3_parse.py`).
"""


def make_bundle(k: int, chunk: list) -> None:
    bundle = INPUTS / f"bundle_{k+1}"
    log(f"bundle_{k+1}: rxn_{chunk[0]:04d} .. rxn_{chunk[-1]:04d} ({len(chunk)} rxns)")
    (bundle / "inputs").mkdir(parents=True, exist_ok=True)

    # Move rxn dirs INTO this bundle (no copy — same filesystem, atomic rename)
    moved = 0
    for rid in chunk:
        src = INPUTS / f"rxn_{rid:04d}"
        dst = bundle / "inputs" / f"rxn_{rid:04d}"
        if dst.exists():
            continue
        if not src.exists():
            log(f"  WARN: rxn_{rid:04d} missing at source (already moved?)")
            continue
        shutil.move(str(src), str(dst))
        moved += 1
    log(f"  relocated {moved} rxn dirs")

    # Copy ORCA installation into the bundle (~17 GB)
    orca_dst = bundle / "orca_6_1_1_avx2"
    if orca_dst.exists():
        log(f"  ORCA already present, skipping copy")
    else:
        log(f"  copying ORCA (17 GB) → {orca_dst.name}/ ...")
        t0 = time.time()
        shutil.copytree(ORCA_SRC, orca_dst, symlinks=True)
        log(f"  ORCA copy done in {time.time()-t0:.1f}s")

    # Write scripts + README
    (bundle / "worker.sh").write_text(WORKER_SH)
    os.chmod(bundle / "worker.sh", 0o755)
    (bundle / "launch.sh").write_text(LAUNCH_SH)
    os.chmod(bundle / "launch.sh", 0o755)
    (bundle / "README.md").write_text(README.format(n=k+1, count=len(chunk)))


def main():
    all_rxns = sorted(
        (int(d.name.split("_")[1]) for d in INPUTS.glob("rxn_*") if d.is_dir()),
        reverse=True,
    )
    total = CHUNK_SIZE * N_CHUNKS
    if len(all_rxns) < total:
        sys.exit(f"Not enough top-level rxn dirs: {len(all_rxns)} < {total}")
    top = all_rxns[:total]
    log(f"Top {total} rxns: rxn_{top[0]:04d} → rxn_{top[-1]:04d}")

    for k in range(N_CHUNKS):
        chunk = top[k*CHUNK_SIZE:(k+1)*CHUNK_SIZE]
        make_bundle(k, chunk)

    log("done. Bundle summary:")
    for k in range(N_CHUNKS):
        bundle = INPUTS / f"bundle_{k+1}"
        n_rxn = sum(1 for d in (bundle / "inputs").glob("rxn_*") if d.is_dir())
        log(f"  {bundle}: {n_rxn} rxns")


if __name__ == "__main__":
    main()
