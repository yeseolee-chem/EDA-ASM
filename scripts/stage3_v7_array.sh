#!/bin/bash
# 8-way v7 descriptor array (xTB single-points, ~97 rxns per shard).
#SBATCH --job-name=st3_v7
#SBATCH --array=0-7%8
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3v7_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3v7_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u scripts/stage3_v7_descriptors.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 8
