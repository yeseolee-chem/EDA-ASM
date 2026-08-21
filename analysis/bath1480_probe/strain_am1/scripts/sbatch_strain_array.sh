#!/bin/bash
# Strain (distortion) energies for 3504 tt reactions on AM1 TS geometry.
# 10-task array, round-robin distribution over rxn_ids.
#
# CLAUDE.md: 48h wall, idempotent, MPI-configured for OpenMPI 4.x.
#
#SBATCH --job-name=strain_am1
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/strain.%A_%a.log

# STRAIN_BUCKET_OFFSET (0 or 5) picks which half of the 10 round-robin buckets this array covers.
# Chain the second half via sbatch --dependency=afterany:<jid_A> --export=ALL,STRAIN_BUCKET_OFFSET=5

set -uo pipefail

module load openmpi/4.1.5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Node-local /tmp scratch (GPFS TMPDIR fills strain_am1/work quota fast when
# 10 tasks × ORCA 8-proc × ~300MB scratch = ~24 GB concurrent).
SCRATCH=/tmp/strain_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts
RUNNER=$SCRIPTS/strain_one_rxn.sh

# Round-robin: task i handles rxn_ids where rxn_id % 10 == (i + STRAIN_BUCKET_OFFSET)
TASK=$SLURM_ARRAY_TASK_ID
NTASK=10
OFFSET=${STRAIN_BUCKET_OFFSET:-0}
BUCKET=$(( (TASK + OFFSET) % NTASK ))
echo "=== bucket_offset=$OFFSET  task=$TASK  bucket=$BUCKET ==="

# Enumerate rxn_ids present in data/
ALL_IDS=($(ls -1 $STRAIN/data | sed -E 's/^rxn_0*([0-9]+)$/\1/' | sort -n))
echo "=== task=$TASK  host=$(hostname)  scratch=$SCRATCH  time=$(date -Is) ==="
echo "=== total rxns available: ${#ALL_IDS[@]} ==="

n_done=0
n_skip=0
n_fail=0
for RID in "${ALL_IDS[@]}"; do
    if [ $((RID % NTASK)) -ne $BUCKET ]; then
        continue
    fi
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

echo "=== task=$TASK  done=$n_done  skipped=$n_skip  failed=$n_fail  time=$(date -Is) ==="

cd / && rm -rf $SCRATCH
exit 0
