#!/bin/bash
# SPEC_07 — λ=1 base-only (xgb 28-d), 5 folds, CPU only. Idempotent.
# CLAUDE.md: --time=48:00:00, ≤10 concurrent overall.
#
# Array: 0..4 → fold index. Single λ = 1.0.

#SBATCH --job-name=s7_base
#SBATCH --array=0-4%2
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_base_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_base_%A_%a.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)   [SLURM_ARRAY_TASK_ID] ${SLURM_ARRAY_TASK_ID}"

MEMBER=${MEMBER:-0}

python -u spec/spec07_lambda_contribution/code/run_lambda.py \
    --lam 1.0 \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --member "${MEMBER}" \
    --device cpu
