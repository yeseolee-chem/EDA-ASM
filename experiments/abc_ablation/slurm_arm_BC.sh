#!/bin/bash
# Arm B (ridge_delta) + Arm C (xgb_delta) as a single SLURM array.
# task_id 0..4 → arm B fold 0..4 ; task_id 5..9 → arm C fold 0..4.
# 10 cells total. Distributed across gpu3/4/5 (≤10 concurrent per CLAUDE.md).

#SBATCH --job-name=abc_bc
#SBATCH --array=0-9%9
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/abc_ablation/logs/bc-%A_%a.out
#SBATCH --error=experiments/abc_ablation/logs/bc-%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p experiments/abc_ablation/logs experiments/abc_ablation/results/cells/{B,C}

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === abc_bc task ${SLURM_ARRAY_TASK_ID} start ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u experiments/abc_ablation/arm_BC_run.py \
    --task ${SLURM_ARRAY_TASK_ID} \
    --descriptor-set m3

echo "[$(date)] === abc_bc task ${SLURM_ARRAY_TASK_ID} done ==="
