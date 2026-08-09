#!/bin/bash
#SBATCH --job-name=A_ml
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_arm_a/ml.%A_%a.log

set -uo pipefail
SEEDS=(22 23 14 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
DATASET=$BASE/manual_tt_7ch.pkl
HPS=$BASE/pipeline_arm_a/hps_arm_a.pkl
ML=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/ml_analysis.py
WORK=$BASE/pipeline_arm_a/ml_results/seed_$SEED
mkdir -p "$WORK" && cd "$WORK"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Arm A ml seed=$SEED  host=$(hostname)  time=$(date -Is) ==="
python3 -u "$ML" $SEED "$DATASET" "$HPS"
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="
exit $RC
