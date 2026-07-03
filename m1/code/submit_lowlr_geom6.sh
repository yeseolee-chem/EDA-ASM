#!/bin/bash
# Track B / m1_delta + geom6 (NO xTB) — 100k max epochs, 10k patience.

#SBATCH --job-name=lr5_g6
#SBATCH --array=0-4%5
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=analysis/exp_6arm_redesign_v2/slurm/logs/lr5_g6-%A_%a.out
#SBATCH --error=analysis/exp_6arm_redesign_v2/slurm/logs/lr5_g6-%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p analysis/exp_6arm_redesign_v2/slurm/logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export BASELINE=geom6
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=lowlr_no_ood

echo "[$(date)] === lr5_g6 task ${SLURM_ARRAY_TASK_ID} start (BASELINE=geom6, no-OOD pool) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u analysis/exp_6arm_redesign_v2/runner_lowlr_trackB_m1delta.py

echo "[$(date)] === lr5_g6 task ${SLURM_ARRAY_TASK_ID} done ==="
