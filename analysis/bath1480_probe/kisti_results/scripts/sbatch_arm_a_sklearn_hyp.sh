#!/bin/bash
#SBATCH --job-name=A_hyp_sk
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_arm_a/arm_a_sklearn.%j.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
WORK=$BASE/pipeline_arm_a/sklearn
mkdir -p "$WORK" && cd "$WORK"
SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/hyp_tuning_sklearn_only.py
DATASET=$BASE/manual_tt_7ch.pkl   # Arm A: no xTB features

echo "=== Arm A sklearn hyp  host=$(hostname)  time=$(date -Is) ==="
python3 -u "$SCRIPT" "$DATASET"
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="
ls -la hps.pkl checkpoint.pkl 2>&1
