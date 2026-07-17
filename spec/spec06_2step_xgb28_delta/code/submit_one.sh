#!/bin/bash
# SPEC_06 — SINGLE (fold, member) cell. No array. Submitted individually
# so each cell competes for scheduling independently (better backfill
# with other users' jobs than a %-throttled array).
# CLAUDE.md: --time=48:00:00; idempotent (skips if output JSON exists).

#SBATCH --job-name=s6_x28d
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec06_one_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec06_one_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)   [FOLD] ${FOLD}   [MEMBER] ${MEMBER}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python -u spec/spec06_2step_xgb28_delta/code/train_xgb28_delta.py \
    --fold "${FOLD}" --member "${MEMBER}" --device cuda
