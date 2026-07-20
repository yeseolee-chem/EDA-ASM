#!/bin/bash
# SPEC_11 Stage 1 - compute d29..d33 for the v9 783-rxn cohort.
# 8-shard array on cpu partitions (~98 rxns/shard, 3 xTB SPs each).
# Idempotent: shard rows already present are skipped.
#SBATCH --job-name=s11_d2933
#SBATCH --array=0-7%8
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_d2933_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_d2933_%A_%a.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)   [shard] ${SLURM_ARRAY_TASK_ID}"
python -u spec/spec11_electronic_33d/code/compute_d29_d33.py \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards 8
