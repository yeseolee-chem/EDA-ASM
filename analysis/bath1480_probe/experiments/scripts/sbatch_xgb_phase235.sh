#!/bin/bash
# XGB baseline for Phase 2 (τ=1e-10), 3 (τ=0.05), 5 (no filter) — 5 seeds each.
# 15 tasks total. All using paper's split + StandardScaler.
#
#SBATCH --job-name=xgb_p235
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-14%10
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/xgb_p235.%A_%a.log

set -uo pipefail

# Cell 0-4  → Phase 2, seeds [22,23,14,1,2]
# Cell 5-9  → Phase 3
# Cell 10-14 → Phase 5
CELL=$SLURM_ARRAY_TASK_ID
PHASE_IDX=$((CELL / 5))
SEED_IDX=$((CELL % 5))
SEEDS=(22 23 14 1 2)
PHASES=("phase2" "phase3" "phase5")
PHASE=${PHASES[$PHASE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

EXP=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments
export DATASET=$EXP/cohort_v1/${PHASE}_dataset_v2.pkl
export SEED=$SEED
export OUT_DIR=$EXP/results/${PHASE}_paper_v2/seed_$SEED

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== xgb $PHASE seed=$SEED  cell=$CELL  host=$(hostname)  time=$(date -Is) ==="
python3 -u $EXP/scripts/xgb_one_seed.py
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="
exit $RC
