#!/bin/bash
#SBATCH --job-name=adf_frag
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=ADF_800/logs/define_fragments_%j.out
#SBATCH --error=ADF_800/logs/define_fragments_%j.err

set -eo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
cd "$REPO"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=== define_fragments on $(hostname) at $(date -Iseconds) ==="
echo "Slurm job: ${SLURM_JOB_ID:-NOJOB}  cpus=${SLURM_CPUS_PER_TASK}"
echo ""

python scripts/define_fragments.py \
    --config configs/fragment_partitioning.yaml \
    --force \
    --log-level INFO

echo ""
echo "=== done at $(date -Iseconds) ==="
