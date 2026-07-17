#!/bin/bash
# SPEC_06 — one job per fold, sequentially trains member 2, 3, 4 on the
# same GPU. Idempotent: each member call skips if JSON already exists,
# so parallel array runs (member 0/1 already done, member 2 folds 0-2
# still running) won't collide.
# CLAUDE.md: --time=48:00:00; each member cell ~1-3h → 3× ≈ 3-9h wall.

#SBATCH --job-name=s6_x28d
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec06_fold_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec06_fold_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

: "${FOLD:?FOLD env var required (0..4)}"
MEMBERS="${MEMBERS:-2 3 4}"

echo "[node] $(hostname)   [FOLD] ${FOLD}   [MEMBERS] ${MEMBERS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

for M in ${MEMBERS}; do
    echo "=== fold=${FOLD} member=${M} start $(date '+%F %T') ==="
    python -u spec/spec06_2step_xgb28_delta/code/train_xgb28_delta.py \
        --fold "${FOLD}" --member "${M}" --device cuda
    rc=$?
    echo "=== fold=${FOLD} member=${M} done rc=${rc} $(date '+%F %T') ==="
    if [ "${rc}" -ne 0 ]; then
        echo "member ${M} failed (rc=${rc}); continuing to next member"
    fi
done
