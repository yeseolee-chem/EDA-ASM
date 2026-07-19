#!/bin/bash
# SPEC_07 — delta training with MEMBER as the array axis.
# Fold and λ are fixed per submission via env vars FOLD and LAM.
# Array element = MEMBER (0..4 depending on --array override).
#
# CLAUDE.md: --time=48:00:00, ≤10 concurrent.
# Idempotent: run_lambda.py skips if member{M}.json already exists.

#SBATCH --job-name=s7_dm
#SBATCH --array=0-4
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_dm_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_dm_%A_%a.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)   [SLURM_ARRAY_TASK_ID] ${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

: "${FOLD:?FOLD env var required (0..4)}"
: "${LAM:?LAM env var required (e.g. 0.0, 0.25, 0.5, 0.75)}"
MEMBER=${SLURM_ARRAY_TASK_ID}

echo "[dispatch] lam=${LAM}  fold=${FOLD}  member=${MEMBER}"

python -u spec/spec07_lambda_contribution/code/run_lambda.py \
    --lam "${LAM}" \
    --fold "${FOLD}" \
    --member "${MEMBER}" \
    --device cuda
