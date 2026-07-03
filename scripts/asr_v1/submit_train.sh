#!/bin/bash
# Production training launch for ASR v1 (dipolar POC).
#
# Runs in sequence on a single allocation:
#   1. train_cv.py --model b0
#   2. train_cv.py --model m1
#   3. learning_curve.py --model b0
#   4. learning_curve.py --model m1
#
# All inputs come from the cached feature .pt file produced by
# submit_cache_features.sh — this job does NOT touch the NequIP backbone.
#
# Usage:   sbatch scripts/asr_v1/submit_train.sh
# Walltime budget: ~30 min (head training is sub-second per fold; the
# learning curve does K × |sizes| × M training runs).

#SBATCH --job-name=asr_v1_train
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/asr_v1/logs/train-%j.out
#SBATCH --error=outputs/asr_v1/logs/train-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === asr_v1 production training start (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo
echo ">>> Step 1/4: train_cv B0 <<<"
python -u scripts/asr_v1/train_cv.py --config configs/asr_v1.yaml --model b0

echo
echo ">>> Step 2/4: train_cv M1 <<<"
python -u scripts/asr_v1/train_cv.py --config configs/asr_v1.yaml --model m1

echo
echo ">>> Step 3/4: learning_curve B0 <<<"
python -u scripts/asr_v1/learning_curve.py --config configs/asr_v1.yaml --model b0

echo
echo ">>> Step 4/4: learning_curve M1 <<<"
python -u scripts/asr_v1/learning_curve.py --config configs/asr_v1.yaml --model m1

echo
echo "[$(date)] === asr_v1 production training done ==="
