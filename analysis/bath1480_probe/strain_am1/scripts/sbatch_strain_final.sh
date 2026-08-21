#!/bin/bash
# Final wave: process ALL remaining rxns (no strain.json) evenly distributed
# across 10 concurrent tasks. Each task reads its slice from final_slices/slice_N.txt.
#
# CLAUDE.md compliance: 48h wall, idempotent (skip if strain.json exists),
# max 10 running (SLURM QoS cap), per-node local /tmp scratch.
#
#SBATCH --job-name=strain_final
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/final.%A_%a.log

# --array and --partition set at submit time to allow splitting across cpu1/cpu2.

set -uo pipefail

module load openmpi/4.1.5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRATCH=/tmp/strain_final_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts
RUNNER=$SCRIPTS/strain_one_rxn.sh
SLICE=$SCRIPTS/final_slices/slice_${SLURM_ARRAY_TASK_ID}.txt

if [ ! -f "$SLICE" ]; then
    echo "FAIL: slice file $SLICE missing"
    exit 2
fi

RXN_IDS=$(cat "$SLICE")
echo "=== final task=$SLURM_ARRAY_TASK_ID  host=$(hostname)  scratch=$SCRATCH  time=$(date -Is) ==="
echo "=== rxns in slice: $(echo $RXN_IDS | wc -w) ==="

# Clean stale partial state (crashed workers may leave stale .done or .lock).
# Only clean rxns that DON'T already have strain.json.
for RID in $RXN_IDS; do
    RXN_TAG=$(printf "rxn_%04d" "$RID")
    W=$STRAIN/work/$RXN_TAG
    if [ -f "$W/strain.json" ]; then
        continue
    fi
    # remove stale lock
    rmdir "$W/.lock" 2>/dev/null || true
    # wipe partial files so runner starts each stage fresh
    find "$W" -maxdepth 1 -type f -delete 2>/dev/null
done

n_done=0
n_skip=0
n_fail=0
for RID in $RXN_IDS; do
    RXN_TAG=$(printf "rxn_%04d" "$RID")
    if [ -f "$STRAIN/work/$RXN_TAG/strain.json" ]; then
        n_skip=$((n_skip+1))
        continue
    fi
    bash $RUNNER $RID
    RC=$?
    if [ $RC -eq 0 ]; then
        n_done=$((n_done+1))
    else
        n_fail=$((n_fail+1))
        echo "!! rxn_$RID failed rc=$RC"
    fi
done

echo "=== final task=$SLURM_ARRAY_TASK_ID  done=$n_done  skipped=$n_skip  failed=$n_fail  time=$(date -Is) ==="

cd / && rm -rf $SCRATCH
exit 0
