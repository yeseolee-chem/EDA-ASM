#!/bin/bash
# SPEC_08 whole-dataset LC — single (SIZE, FOLD, MEMBER) cell.
# CLAUDE.md: --time=48:00:00, idempotent. Broadened to all gpu partitions
# because gpu3/gpu4/gpu5 were fully occupied by higher-priority jobs.

#SBATCH --job-name=s08w_lc
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

: "${SIZE:?SIZE env var required (100..786)}"
: "${FOLD:?FOLD env var required (0..4)}"
MEMBER="${MEMBER:-0}"

echo "[node] $(hostname)  [SIZE] ${SIZE}  [FOLD] ${FOLD}  [MEMBER] ${MEMBER}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python -u spec/spec08_whole_dataset_learning_curve/code/train_lc_cell.py \
    --size "${SIZE}" --fold "${FOLD}" --member "${MEMBER}" --device cuda
rc=$?
echo "=== spec08w size=${SIZE} fold=${FOLD} member=${MEMBER} done rc=${rc} $(date '+%F %T') ==="
exit "${rc}"
