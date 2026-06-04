#!/bin/bash
# One-shot SLURM job: precompute NequIP backbone features for all 134
# labelled dipolar reactions. Output → outputs/asr_v1/features_dipolar_ep29.pt
#
# Usage:   sbatch scripts/asr_v1/submit_cache_features.sh
# Walltime budget: ~15 min (caching CPU was ~30s/reaction; on 1 GPU expect ≪ 5min)

#SBATCH --job-name=asr_v1_cache
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/asr_v1/logs/cache-%j.out
#SBATCH --error=outputs/asr_v1/logs/cache-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === asr_v1 feature cache start (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/asr_v1/cache_features.py --config configs/asr_v1.yaml --device cuda
echo "[$(date)] === asr_v1 feature cache done ==="
