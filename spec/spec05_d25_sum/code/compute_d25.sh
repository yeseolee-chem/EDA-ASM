#!/bin/bash
# 4-shard xTB single-points on R geometry (~1550 SPs total, ~3-8s each)
#SBATCH --job-name=d25
#SBATCH --array=0-3%4
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/d25_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/d25_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u spec/spec05_d25_sum/code/compute_d25.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
