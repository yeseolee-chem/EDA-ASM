#!/bin/bash
# Phase 2 paper-exact: τ=1e-10 variance-filtered (61 features) + 5 seeds
#SBATCH --job-name=ph2_paper
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase2_paper.%A_%a.log

set -uo pipefail
SEEDS=(22 23 14 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EXP=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments
DATASET=$EXP/cohort_v1/phase2_dataset.pkl
HPS=$EXP/cohort_v1/hps_phase23.pkl
ML_SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/ml_analysis.py

WORK=$EXP/results/phase2_paper/seed_$SEED
mkdir -p "$WORK"; cd "$WORK"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== phase2 (τ=1e-10, 61 features) seed=$SEED $(date -Is) ==="
python3 -u "$ML_SCRIPT" $SEED "$DATASET" "$HPS"
echo "=== end rc=$? $(date -Is) ==="
