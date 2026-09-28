#!/usr/bin/env python3
"""Bundle top-600 pending rxns (highest number first) into 2 bundles of 300 each.
No ORCA installation this time — target user already has ORCA.
"""
import os
import shutil
import stat
import sys
import time
from pathlib import Path

INPUTS = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
STATE_DONE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/state/done")

CHUNK_SIZE = 300
N_CHUNKS = 2

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


WORKER_SH = r"""#!/bin/bash
# SMD-relabel bundle worker (no bundled ORCA — expects ORCA at $ORCA_BIN or in PATH).
# SLURM array, 48h walltime, 10-way (adjust --array=0-9%N below).
#
# ---- edit resource block for your target cluster if needed ----
#SBATCH --job-name=smd_bundle
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9%10
#SBATCH --output=logs/wrk_%A_%a.out
#SBATCH --partition=cpu1,cpu2
# #SBATCH --account=your_account    # uncomment + set if needed
# --------------------------------------------------------------
set -uo pipefail

BUNDLE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$BUNDLE"
mkdir -p logs state/done state/claim state/timing

# ---- ORCA binary ----
# Priority: ORCA_BIN env var → $HOME/orca_6_1_1_avx2/orca → which orca
if [ -z "${ORCA_BIN:-}" ]; then
    for cand in \
        "$HOME/orca_6_1_1_avx2/orca" \
        "$HOME/orca/orca" \
        "/opt/orca/orca" \
        "$(command -v orca 2>/dev/null || true)"; do
        if [ -x "$cand" ]; then ORCA_BIN="$cand"; break; fi
    done
fi
if [ -z "${ORCA_BIN:-}" ] || [ ! -x "$ORCA_BIN" ]; then
    echo "FATAL: ORCA not found. Set ORCA_BIN env var or install at \$HOME/orca_6_1_1_avx2/orca" >&2
    exit 2
fi
export ORCA_BIN
echo "ORCA_BIN=$ORCA_BIN"

# ---- MPI runtime auto-detect (OpenMPI 4.1.x) ----
if [ -z "${MPI_ROOT:-}" ]; then
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
if [ -n "${MPI_ROOT:-}" ]; then
    export PATH="$MPI_ROOT/bin:$(dirname "$ORCA_BIN"):${PATH:-}"
    export LD_LIBRARY_PATH="$MPI_ROOT/lib64:$MPI_ROOT/lib:$(dirname "$ORCA_BIN"):${LD_LIBRARY_PATH:-}"
    echo "MPI_ROOT=$MPI_ROOT"
else
    export PATH="$(dirname "$ORCA_BIN"):${PATH:-}"
    export LD_LIBRARY_PATH="$(dirname "$ORCA_BIN"):${LD_LIBRARY_PATH:-}"
    echo "WARN: no OpenMPI found — ORCA may fail. Set MPI_ROOT or edit worker.sh."
fi

# ORCA MPI stability flags
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp,self,sm

# ---- work pool ----
mapfile -t RXNS < <(ls -1 "$BUNDLE/inputs/" | grep '^rxn_' | sort)
TOTAL=${#RXNS[@]}
if [ $TOTAL -eq 0 ]; then
    echo "FATAL: no rxn dirs in $BUNDLE/inputs/" >&2; exit 3
fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID:-manual}
JOB_START=$(date +%s)
WALL_S=$((48*3600))
SAFETY_S=1800

NODE=$(hostname -s)
echo "=== worker IDX=$IDX JOB=$JOB_ID NODE=$NODE TOTAL=$TOTAL start=$(date -Is) ==="

processed=0; skipped=0; claimed_by_other=0
for OFFSET in $(seq 0 $((TOTAL - 1))); do
    ELAPSED=$(( $(date +%s) - JOB_START ))
    REMAIN=$(( WALL_S - ELAPSED ))
    if [ $REMAIN -lt $SAFETY_S ]; then
        echo "wall guard: ${REMAIN}s < ${SAFETY_S}s — exit"; break
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
        find . -maxdepth 1 -type f ! -name "*.inp" ! -name "*.out" ! -name "*.err" -delete
        echo "[OK]   $RID  ${ELAPSED_RXN}s  processed=$processed"
    else
        echo "[FAIL] $RID rc=$RC  ${ELAPSED_RXN}s"
    fi
    rm -rf "$CLAIM_DIR" 2>/dev/null || true
    cd "$BUNDLE"
done
echo "=== worker $IDX done processed=$processed skipped=$skipped stolen=$claimed_by_other ==="
"""


LAUNCH_SH = r"""#!/bin/bash
# One-command launcher (no ORCA extraction — target has ORCA already).
# Usage:  bash launch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' not found. Target must be a SLURM cluster." >&2; exit 1
fi
[ -d ./inputs ] && [ "$(ls -1 ./inputs | grep -c '^rxn_')" -gt 0 ] || { echo "ERROR: no rxn dirs in ./inputs/" >&2; exit 3; }
[ -f ./worker.sh ] || { echo "ERROR: worker.sh missing" >&2; exit 4; }
mkdir -p logs state/done state/claim state/timing

# Optional ORCA check (workers will retry anyway with more paths)
if [ -z "${ORCA_BIN:-}" ]; then
    for cand in "$HOME/orca_6_1_1_avx2/orca" "$HOME/orca/orca" "/opt/orca/orca"; do
        [ -x "$cand" ] && { echo "found ORCA at $cand"; break; }
    done
fi

N_RXN=$(ls -1 ./inputs | grep -c '^rxn_')
N_DONE=$(ls -1 ./state/done 2>/dev/null | wc -l)
N_PENDING=$((N_RXN - N_DONE))
echo "Bundle: $(pwd)"
echo "  rxns: $N_RXN  done: $N_DONE  pending: $N_PENDING"
if [ $N_PENDING -le 0 ]; then
    echo "All done. Nothing to submit."; exit 0
fi

if ! sbatch --test-only ./worker.sh 2>/tmp/sbatch_err_$$; then
    echo "ERROR: sbatch rejected worker.sh:" >&2
    cat /tmp/sbatch_err_$$ >&2
    echo "" >&2
    echo "Hint: edit --partition= / --account= inside worker.sh" >&2
    rm -f /tmp/sbatch_err_$$; exit 5
fi
rm -f /tmp/sbatch_err_$$

JID=$(sbatch --parsable ./worker.sh)
echo ""
echo "Submitted array job: $JID (10 workers × 48h)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls state/done | wc -l    # progress out of $N_RXN"
echo "  tail -f logs/wrk_${JID}_0.out"
"""


README = """# SMD-relabel bundle {n} (pending 300 rxns, no ORCA bundled)

## Usage

Prerequisites on target:
- SLURM (sbatch)
- ORCA 6.1.1 AVX2 build available at `$ORCA_BIN` env var or `$HOME/orca_6_1_1_avx2/orca` or in PATH
- OpenMPI 4.1.x (auto-detected from common paths)

Then:

    bash launch.sh

Submits a 10-element SLURM array with 48h walltime each.

## Progress

    ls state/done | wc -l    # done out of {count}
    squeue -u $USER

## Resume

Just re-run `bash launch.sh` — idempotent, skips already-done rxns.
"""


def make_bundle(k: int, chunk: list) -> Path:
    bundle = INPUTS / f"bundle_{k+3}"  # bundle_3, bundle_4
    (bundle / "inputs").mkdir(parents=True, exist_ok=True)
    log(f"bundle_{k+3}: rxn_{chunk[0]:04d} → rxn_{chunk[-1]:04d} ({len(chunk)} rxns)")
    moved = 0
    for rid in chunk:
        src = INPUTS / f"rxn_{rid:04d}"
        dst = bundle / "inputs" / f"rxn_{rid:04d}"
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            moved += 1
    log(f"  moved {moved} rxn dirs")
    (bundle / "worker.sh").write_text(WORKER_SH)
    os.chmod(bundle / "worker.sh", 0o755)
    (bundle / "launch.sh").write_text(LAUNCH_SH)
    os.chmod(bundle / "launch.sh", 0o755)
    (bundle / "README.md").write_text(README.format(n=k+3, count=len(chunk)))
    return bundle


def main():
    # Get pending rxns (dirs without done flag)
    all_dirs = [d.name for d in INPUTS.glob("rxn_*") if d.is_dir()]
    done_flags = {p.name for p in STATE_DONE.iterdir()}
    pending = sorted(
        [int(d.split("_")[1]) for d in all_dirs if d not in done_flags],
        reverse=True,   # 높은 번호부터
    )
    log(f"pending 총: {len(pending)}")
    log(f"  최고 rxn 번호: {pending[0]}, 최저: {pending[-1]}")

    top = pending[:CHUNK_SIZE * N_CHUNKS]
    log(f"상위 {len(top)} 개 선정: rxn_{top[0]:04d} → rxn_{top[-1]:04d}")

    for k in range(N_CHUNKS):
        chunk = top[k*CHUNK_SIZE:(k+1)*CHUNK_SIZE]
        make_bundle(k, chunk)


if __name__ == "__main__":
    main()
