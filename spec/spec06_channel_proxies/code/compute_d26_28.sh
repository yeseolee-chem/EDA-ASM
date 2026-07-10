#!/bin/bash
# 4-shard xTB SP array; each shard does ~194 rxns x 3 SP (complex + fragA + fragB).
#SBATCH --job-name=d26_28
#SBATCH --array=0-3%4
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/d26_28_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/d26_28_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u spec/spec06_channel_proxies/code/compute_d26_28.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
