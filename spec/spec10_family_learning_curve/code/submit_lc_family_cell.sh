#!/bin/bash
# SPEC_10 — one (FAMILY, SIZE, FOLD, MEMBER) family-restricted learning-curve cell.
# CLAUDE.md rules: --time=48:00:00; idempotent (skips if output JSON exists);
# distributed across gpu3/gpu4/gpu5.

#SBATCH --job-name=s10_flc
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec10_flc_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec10_flc_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

: "${FAMILY:?FAMILY env var required (dipolar|qmrxn20_e2|qmrxn20_sn2|rgd1)}"
: "${SIZE:?SIZE env var required (50|100|150)}"
: "${FOLD:?FOLD env var required (0..4)}"
MEMBER="${MEMBER:-0}"

echo "[node] $(hostname)  [FAM] ${FAMILY}  [SIZE] ${SIZE}  [FOLD] ${FOLD}  [MEMBER] ${MEMBER}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python -u spec/spec10_family_learning_curve/code/train_lc_family_cell.py \
    --family "${FAMILY}" --size "${SIZE}" --fold "${FOLD}" \
    --member "${MEMBER}" --device cuda
rc=$?
echo "=== spec10_flc ${FAMILY} size=${SIZE} fold=${FOLD} member=${MEMBER} done rc=${rc} $(date '+%F %T') ==="
exit "${rc}"
