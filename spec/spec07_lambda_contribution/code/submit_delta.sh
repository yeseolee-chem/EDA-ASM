#!/bin/bash
# SPEC_07 — δ training for λ ∈ {0.0, 0.25, 0.5, 0.75}, 5 folds each = 20 cells.
# Array id → (λ_idx, fold): lam = LAMBDAS[i//5], fold = i%5.
# Idempotent, ≤10 concurrent, spread gpu3/gpu4/gpu5.

#SBATCH --job-name=s7_delta
#SBATCH --array=0-19%5
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_delta_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_delta_%A_%a.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)   [SLURM_ARRAY_TASK_ID] ${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

LAMBDAS=(0.0 0.25 0.5 0.75)
TASK=${SLURM_ARRAY_TASK_ID}
LAM_IDX=$(( TASK / 5 ))
FOLD=$(( TASK % 5 ))
LAM=${LAMBDAS[$LAM_IDX]}
MEMBER=${MEMBER:-0}

echo "[dispatch] lam=${LAM}  fold=${FOLD}  member=${MEMBER}"

python -u spec/spec07_lambda_contribution/code/run_lambda.py \
    --lam "${LAM}" \
    --fold "${FOLD}" \
    --member "${MEMBER}" \
    --device cuda
