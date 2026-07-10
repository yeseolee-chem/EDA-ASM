#!/bin/bash
# MACE-OFF23_medium feature extraction for v7 776-reaction cohort.
# 2-way sharded on GPU. Idempotent (skips complete .pt files).
#SBATCH --job-name=st2_v7
#SBATCH --array=0-1%2
#SBATCH --partition=gpu1,gpu3,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st2v7_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st2v7_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python -u scripts/stage2_v7_mace.py --model-size medium \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 2
