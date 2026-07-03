#!/bin/bash
#SBATCH --job-name=st2_mace
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_rebuild/logs/st2_%j.out
#SBATCH --error=pipeline_rebuild/logs/st2_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p pipeline_rebuild/logs pipeline_rebuild/results

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python -u pipeline_rebuild/stage2_mace_features.py --model-size medium
