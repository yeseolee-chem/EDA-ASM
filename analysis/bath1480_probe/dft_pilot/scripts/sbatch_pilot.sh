#!/bin/bash
# Pilot: 5 reactions, DFT TS reopt + EDA-NOCV SPE.
# CLAUDE.md: 48h wall, idempotent, no login-node processes.
#
#SBATCH --job-name=dft_pilot
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot/logs/pilot.%A_%a.log

set -uo pipefail

RXN_IDS=(3 7 56 489 1217)
RXN_ID=${RXN_IDS[$SLURM_ARRAY_TASK_ID]}

# ORCA 6.x needs OpenMPI 4.x at runtime (libmpi.so.40) for %pal nprocs.
module load openmpi/4.1.5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Node-local scratch (avoid GPFS churn on Hessian recompute)
SCRATCH=/tmp/dft_pilot_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

# reactot env for python (parser + build_eda_input)
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRIPTS=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot/scripts

echo "=== task=$SLURM_ARRAY_TASK_ID  rxn=$RXN_ID  host=$(hostname)  scratch=$SCRATCH  time=$(date -Is) ==="
bash $SCRIPTS/run_one.sh $RXN_ID
RC=$?
echo "=== task=$SLURM_ARRAY_TASK_ID  rxn=$RXN_ID  rc=$RC  time=$(date -Is) ==="

# Cleanup node-local scratch
cd /tmp && rm -rf $SCRATCH

exit $RC
