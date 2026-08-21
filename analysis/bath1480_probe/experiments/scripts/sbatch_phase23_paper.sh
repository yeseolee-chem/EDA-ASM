#!/bin/bash
# Phase 2/3 paper-exact reproduction: uses paper's ml_analysis.py with
# our extended dataset (features_armB + our new d1_own/d2_own labels).
# 5 seeds matching paper (22, 23, 14, 1, 2).
#
#SBATCH --job-name=ph23_paper
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase23_paper.%A_%a.log

set -uo pipefail

SEEDS=(22 23 14 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EXP=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments
DATASET=$EXP/cohort_v1/phase23_dataset.pkl
HPS=$EXP/cohort_v1/hps_phase23.pkl
ML_SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/ml_analysis.py

WORK=$EXP/results/phase23_paper/seed_$SEED
mkdir -p "$WORK"
cd "$WORK"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== phase23 paper-exact seed=$SEED  host=$(hostname)  time=$(date -Is) ==="
python3 -u "$ML_SCRIPT" $SEED "$DATASET" "$HPS"
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="
ls -la ml_results.pkl 2>&1
exit $RC
