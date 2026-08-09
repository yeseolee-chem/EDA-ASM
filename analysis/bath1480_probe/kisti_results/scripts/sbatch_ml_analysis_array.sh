#!/bin/bash
# Phase 7: ml_analysis 5-seed array — uses hps_7ch.pkl
#SBATCH --job-name=tt_ml7
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/ml7.%A_%a.log

set -uo pipefail

SEEDS=(22 23 14 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
DATASET=$BASE/manual_tt_7ch_xtb.pkl
HPS=$BASE/pipeline_7ch/hps_7ch.pkl
ML_SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/ml_analysis.py

WORK=$BASE/pipeline_7ch/ml_results/seed_$SEED
mkdir -p "$WORK"
cd "$WORK"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== ml7 seed=$SEED  host=$(hostname)  time=$(date -Is) ==="
python3 -u "$ML_SCRIPT" $SEED "$DATASET" "$HPS"
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="
ls -la ml_results.pkl 2>&1
exit $RC
