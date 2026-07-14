#!/bin/bash
# m3 v2 / Track B / m1_delta + xtb_geom6_plus_v2 (cache rebuilt with index-based extractor)
# Only trains member 0 per fold (5 jobs total, matches m1/m2 comparison set).

#SBATCH --job-name=lr5_m3v2
#SBATCH --array=0-4%5
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=analysis/exp_6arm_redesign_v2/slurm/logs/lr5_m3v2-%A_%a.out
#SBATCH --error=analysis/exp_6arm_redesign_v2/slurm/logs/lr5_m3v2-%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p analysis/exp_6arm_redesign_v2/slurm/logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export BASELINE=xtb_geom6_plus_v2
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=lowlr_no_ood

echo "[$(date)] === lr5_m3v2 task ${SLURM_ARRAY_TASK_ID} start (BASELINE=${BASELINE}, fold=${SLURM_ARRAY_TASK_ID}, member 0 only) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u analysis/exp_6arm_redesign_v2/runner_lowlr_trackB_m1delta.py --fold ${SLURM_ARRAY_TASK_ID} --member 0

echo "[$(date)] === lr5_m3v2 task ${SLURM_ARRAY_TASK_ID} done ==="
