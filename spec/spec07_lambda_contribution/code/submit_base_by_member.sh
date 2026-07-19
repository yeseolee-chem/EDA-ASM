#!/bin/bash
# SPEC_07 — base-only (λ=1.0) training with MEMBER as the array axis.
# Fold fixed per submission via env var FOLD. Array element = MEMBER.
# Very cheap CPU job (~1s per cell) since it just fits XGBoost.

#SBATCH --job-name=s7_bm
#SBATCH --array=0-4
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_bm_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_bm_%A_%a.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

: "${FOLD:?FOLD env var required (0..4)}"
MEMBER=${SLURM_ARRAY_TASK_ID}

echo "[dispatch] base fold=${FOLD} member=${MEMBER}"

python -u spec/spec07_lambda_contribution/code/run_lambda.py \
    --lam 1.0 \
    --fold "${FOLD}" \
    --member "${MEMBER}" \
    --device cpu
