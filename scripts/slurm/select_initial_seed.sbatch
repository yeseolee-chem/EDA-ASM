#!/bin/bash
#SBATCH --job-name=asr_seed_sel
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
cd "$REPO"

# Restrict NumPy / sklearn BLAS threads to match --cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

echo "=== Environment ==="
echo "  host         : $(hostname)"
echo "  date         : $(date -Iseconds)"
echo "  python       : $(which python)"
echo "  cpus         : ${SLURM_CPUS_PER_TASK}"
echo "  job id       : ${SLURM_JOB_ID:-NOJOB}"
echo ""

python scripts/select_initial_seed.py \
    --config configs/seed_selection.yaml \
    --output-dir data/selection/initial_seed_v1 \
    --seed 42 \
    --log-level INFO

echo ""
echo "=== Done at $(date -Iseconds) ==="
