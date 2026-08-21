#!/bin/bash
# Retry sbatch for known-failed rxns. Uses STRAIN_RETRY_MODE=1
# (SlowConv + MaxIter 500). Explicit rxn_id list, no bucketing.
#
# Usage: submit with --export=ALL,STRAIN_RETRY_IDS="1538 1750 4322"
# Or edit RETRY_IDS default below.
#
#SBATCH --job-name=strain_retry
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/retry.%A_%a.log

set -uo pipefail

module load openmpi/4.1.5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRATCH=/tmp/strain_retry_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts
RUNNER=$SCRIPTS/strain_one_rxn.sh

export STRAIN_RETRY_MODE=1

RETRY_IDS=${STRAIN_RETRY_IDS:-"1538 1750 4322"}

echo "=== retry task=${SLURM_ARRAY_TASK_ID:-solo}  host=$(hostname)  time=$(date -Is) ==="
echo "=== retry ids: $RETRY_IDS ==="

# Clean partial state so runner re-executes from scratch (keep strain.json if any).
for RID in $RETRY_IDS; do
    RXN_TAG=$(printf "rxn_%04d" "$RID")
    WORK=$STRAIN/work/$RXN_TAG
    if [ -f "$WORK/strain.json" ]; then
        echo "SKIP $RXN_TAG: strain.json already exists (nothing to retry)"
        continue
    fi
    echo "CLEAN $RXN_TAG: wiping partial state in $WORK"
    find "$WORK" -maxdepth 1 -type f -delete 2>/dev/null
    find "$WORK" -maxdepth 1 -mindepth 1 -type d -name ".lock" -exec rmdir {} + 2>/dev/null
done

n_done=0
n_fail=0
for RID in $RETRY_IDS; do
    RXN_TAG=$(printf "rxn_%04d" "$RID")
    if [ -f "$STRAIN/work/$RXN_TAG/strain.json" ]; then
        continue
    fi
    bash $RUNNER $RID
    RC=$?
    if [ $RC -eq 0 ]; then
        n_done=$((n_done+1))
    else
        n_fail=$((n_fail+1))
        echo "!! rxn_$RID retry failed rc=$RC"
    fi
done

echo "=== retry done=$n_done  failed=$n_fail  time=$(date -Is) ==="

cd / && rm -rf $SCRATCH
exit 0
