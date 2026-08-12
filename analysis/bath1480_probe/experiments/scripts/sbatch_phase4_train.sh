#!/bin/bash
# Phase 4: XGBoost + MACE residual training. 5 folds × 5 seeds = 25 cells.
# 3-way GPU parallel (gpu3/4/5). Array 0-24 with %3 concurrent cap.
#
#SBATCH --job-name=exp_p4trn
#SBATCH --time=48:00:00
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-24%3
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase4_train.%A_%a.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Cell index → (fold, seed)
CELL=$SLURM_ARRAY_TASK_ID
export FOLD=$((CELL / 5))
export SEED=$((CELL % 5))

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
echo "=== Phase4 train cell=$CELL (fold=$FOLD seed=$SEED) host=$(hostname) start=$(date -Is) ==="
python3 -u $SCRIPTS/05_train_e3.py
echo "=== end $(date -Is) ==="
