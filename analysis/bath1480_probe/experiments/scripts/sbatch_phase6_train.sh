#!/bin/bash
# Phase 6: Phase 5 XGB + MACE residual. 5 cells (one per seed, paper 80/10/10 split).
# Direct comparison to Phase 5 XGB with matching seeds.
#
#SBATCH --job-name=ph6_xm
#SBATCH --time=48:00:00
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase6.%A_%a.log

set -uo pipefail

SEEDS=(22 23 14 1 2)
export SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== phase6 task=$SLURM_ARRAY_TASK_ID seed=$SEED $(date -Is) ==="
python3 -u /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts/09_train_phase6.py
echo "=== end rc=$? $(date -Is) ==="
