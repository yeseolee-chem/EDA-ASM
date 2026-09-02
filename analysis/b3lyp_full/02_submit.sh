#!/bin/bash
# SPEC16rev Step 2 — per-rxn ORCA runner (array element).
# 5 SPEs per rxn, each with idempotent skip; rm -f before rerun to prevent
# FSPE-append leaks (spec15 rxn 1479 lesson).
#
#SBATCH --job-name=b3full
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=40G
#SBATCH --exclude=n045
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/logs/%A_%a.out

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full

mapfile -t DIRS < <(ls -d $BASE/inputs/rxn_*/ | sort)
BASE_OFFSET=${BASE_OFFSET:-0}
IDX=$((BASE_OFFSET + SLURM_ARRAY_TASK_ID))
D="${DIRS[$IDX]}"
D="${D%/}"
[ -z "$D" ] && exit 0
rid=$(basename "$D")

export ORCA_BIN=/home1/yeseo1ee/orca_6_1_1_avx2/orca
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1
export PATH=$MPI_ROOT/bin:$(dirname $ORCA_BIN):$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib64:$(dirname $ORCA_BIN):${LD_LIBRARY_PATH:-}
# Allow OpenMPI to oversubscribe: SLURM gives us 10 CPUs as 1 task; MPI
# needs to launch 5 procs. Without this it errors "not enough slots".
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=self,vader,tcp   # avoid IB/PSM issues on non-IB nodes
# UCX/hcoll silently crash MPI mid-run on n044/n048 (563 stale partial
# eda.out with only "UCX WARN IB Async event" tails). Force OB1 PML
# (uses BTL above) and disable Mellanox collective library.
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp,self,sm

cd "$D"
echo "=== $rid $(date -Is) ==="

# Option B: run all 5 SPEs in PARALLEL (background + wait).
# Each ORCA process is single-threaded here (no %pal in input), so 5 procs
# fit in 5 CPUs per task. All independent inputs, no shared scratch files.
pids=()
for STEM in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
    [ -f "$STEM.inp" ] || continue
    # idempotent skip
    if [ -f "$STEM.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$STEM.out"; then
        echo "  skip $STEM"
        continue
    fi
    # Remove stale output to prevent FSPE-count anomalies from partial reruns
    rm -f "$STEM.out" "$STEM.err"
    ( $ORCA_BIN "$STEM.inp" > "$STEM.out" 2> "$STEM.err"; echo "  $STEM rc=$?" ) &
    pids+=($!)
done
# Wait for all parallel SPEs to finish
for p in "${pids[@]}"; do wait $p; done

# ---- Disk cleanup: keep .inp/.out/.err/.gbw, delete densities/cpcm/tmp ----
# Quota raised to 300 GB → keep .gbw wavefunctions (~10 MB/rxn total,
# ~52 GB for full 5262) for future re-analysis without SCF re-run.
# Cleanup only runs if all 5 SPEs terminated normally.
all_ok=1
for s in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
    [ -f "$s.inp" ] || continue
    grep -q "ORCA TERMINATED NORMALLY" "$s.out" 2>/dev/null || { all_ok=0; break; }
done
if [ "$all_ok" -eq 1 ]; then
    find . -maxdepth 1 -type f \
        ! -name "*.inp" ! -name "*.out" ! -name "*.err" ! -name "*.gbw" \
        -delete
    echo "  cleanup done"
fi

echo "=== end $(date -Is) ==="
