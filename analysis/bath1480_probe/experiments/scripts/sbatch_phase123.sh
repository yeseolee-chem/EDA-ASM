#!/bin/bash
# Shared sbatch for Phase 1/2/3 (sklearn 3 models × N targets × 5 seeds).
# Configure via env vars: PHASE_NAME, LABELS_PKL, FEATURES_PKL, TARGET_COLS, VAR_TAU
# Array 0-4: one task per seed. 5 tasks parallel per phase.
#
#SBATCH --job-name=exp_ph
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase_%x.%A_%a.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SEED=$SLURM_ARRAY_TASK_ID

# All required env vars must be set by submitter via --export=ALL,PHASE_NAME=...,LABELS_PKL=...

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
echo "=== $PHASE_NAME seed=$SEED  host=$(hostname)  start=$(date -Is) ==="
python3 -u $SCRIPTS/02_train_sklearn.py
echo "=== $PHASE_NAME seed=$SEED  end=$(date -Is) ==="
