#!/bin/bash
# SPEC_06 B1 — one job per (FAMILY, FOLD); trains members 0..4 sequentially.
# CLAUDE.md: --time=48:00:00; idempotent per-cell skip.

#SBATCH --job-name=s6b1
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec06_b1_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec06_b1_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

: "${FAMILY:?FAMILY env var required}"
: "${FOLD:?FOLD env var required (0..4)}"
MEMBERS="${MEMBERS:-0 1 2 3 4}"

echo "[node] $(hostname)   [FAMILY] ${FAMILY}   [FOLD] ${FOLD}   [MEMBERS] ${MEMBERS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

for M in ${MEMBERS}; do
    echo "=== ${FAMILY} fold=${FOLD} member=${M} start $(date '+%F %T') ==="
    python -u spec/spec09_per_family_xgb28_delta/code/train_family_xgb28_delta.py \
        --family "${FAMILY}" --fold "${FOLD}" --member "${M}" --device cuda
    rc=$?
    echo "=== ${FAMILY} fold=${FOLD} member=${M} done rc=${rc} $(date '+%F %T') ==="
    if [ "${rc}" -ne 0 ]; then
        echo "cell failed (rc=${rc}); continuing"
    fi
done
