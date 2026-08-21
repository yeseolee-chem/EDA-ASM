#!/bin/bash
# Phase 2/3/5 sklearn (paper's ml_analysis.py) — v2 datasets, oracle-clean + strain swap.
# 5 seeds × 3 phases = 15 tasks, run at most 10 concurrent per SLURM cap.
#
#SBATCH --job-name=ph235v2
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --array=0-14%10
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/ph235v2.%A_%a.log

set -uo pipefail

SEEDS=(22 23 14 1 2)
PHASES=(phase2 phase3 phase5)
CELL=$SLURM_ARRAY_TASK_ID
P=${PHASES[$((CELL/5))]}
S=${SEEDS[$((CELL%5))]}

EXP=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments
DATASET=$EXP/cohort_v1/${P}_dataset_v2.pkl
HPS=$EXP/cohort_v1/hps_phase23_v2.pkl
WORK=$EXP/results/${P}_paper_v2/seed_$S
mkdir -p "$WORK"; cd "$WORK"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ML=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/ml_analysis.py
echo "=== $P v2 seed=$S cell=$CELL $(date -Is) ==="
python3 -u "$ML" $S "$DATASET" "$HPS"
echo "=== end rc=$? $(date -Is) ==="
