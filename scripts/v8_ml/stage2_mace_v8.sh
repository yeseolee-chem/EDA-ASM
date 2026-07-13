#!/bin/bash
#SBATCH --job-name=stage2_v8
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/stage2_v8_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/stage2_v8_%A_%a.out

# Stage 2 (v8) — MACE-OFF23_medium feature extraction for the 799-rxn v8 cohort.
# 4-way array; each shard picks rxns where row_index % 4 == SLURM_ARRAY_TASK_ID.
# 48 h walltime (project rule); resubmit if the wall clips a shard — outputs
# are per-rxn .pt files and the script skips any that already exist.

set -euo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8

# Conda env
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$REPO"

echo "[stage2_v8] host=$(hostname) job=${SLURM_JOB_ID} shard=${SLURM_ARRAY_TASK_ID}/4 start=$(date -Iseconds)"
nvidia-smi || true

python -u scripts/v8_ml/stage2_mace_v8.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards 4

echo "[stage2_v8] end=$(date -Iseconds)"
