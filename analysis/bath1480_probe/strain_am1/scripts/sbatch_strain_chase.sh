#!/bin/bash
# Chase task: iterate ALL remaining rxns; runner's atomic lock skips
# any rxn currently held by another worker (bucketed array or another chaser).
#
# This exists because the bucketed array leaves parallelism to drop as
# short buckets drain. Chase tasks can be added freely (up to 10-running cap)
# to keep utilisation high.
#
#SBATCH --job-name=strain_chase
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/chase.%A_%a.log

set -uo pipefail

module load openmpi/4.1.5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRATCH=/tmp/strain_chase_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts
RUNNER=$SCRIPTS/strain_one_rxn.sh

# STRAIN_CHASE_ORDER: forward (default), reverse, or random.
# forward chases lowest rxn_ids first; reverse chases highest first.
# Use reverse for chasers to reduce collisions with the forward-going bucketed array.
ORDER=${STRAIN_CHASE_ORDER:-reverse}

echo "=== chase task=${SLURM_ARRAY_TASK_ID:-solo}  host=$(hostname)  order=$ORDER  scratch=$SCRATCH  time=$(date -Is) ==="

case "$ORDER" in
    forward) ALL_IDS=($(ls -1 $STRAIN/data | sed -E 's/^rxn_0*([0-9]+)$/\1/' | sort -n));;
    reverse) ALL_IDS=($(ls -1 $STRAIN/data | sed -E 's/^rxn_0*([0-9]+)$/\1/' | sort -nr));;
    random)  ALL_IDS=($(ls -1 $STRAIN/data | sed -E 's/^rxn_0*([0-9]+)$/\1/' | shuf));;
    *) echo "unknown ORDER=$ORDER"; exit 2;;
esac
echo "=== total rxns available: ${#ALL_IDS[@]} ==="

n_done=0
n_skip=0
n_fail=0
for RID in "${ALL_IDS[@]}"; do
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

echo "=== chase done=$n_done  skipped=$n_skip  failed=$n_fail  time=$(date -Is) ==="

cd / && rm -rf $SCRATCH
exit 0
