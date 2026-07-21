#!/bin/bash
# SPEC_11 - 2-step xgb33 + delta training. One task per outer fold.
# Idempotent: existing member{M}.json is skipped.
# CLAUDE.md: --time=48:00:00, <=10 concurrent, spread gpu3/gpu4/gpu5.
#SBATCH --job-name=s11_x33d
#SBATCH --array=0-4
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_x33d_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_x33d_%A_%a.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)   [SLURM_ARRAY_TASK_ID] ${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

MEMBER=${MEMBER:-0}

python -u spec/spec11_electronic_33d/code/train_xgb33_delta.py \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --member "${MEMBER}" \
    --device cuda
