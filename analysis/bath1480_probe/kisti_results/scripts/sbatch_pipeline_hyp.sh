#!/bin/bash
# Espley hyp_tuning.py on manual_tt_7ch.pkl (patched: ε extended, C extended, VarianceThreshold fixed)
# Runs seed=23 (hardcoded in hyp_tuning.py) — sklearn (Ridge/KRR/SVR/RF) + NN 2L/4L for all _dft targets.

#SBATCH --job-name=tt_hyp_7ch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_hyp.%j.log

set -uo pipefail

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
WORK=$BASE/pipeline_work_hyp
mkdir -p "$WORK"
cd "$WORK"

HYP_TUNING=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/hyp_tuning.py
DATASET=$BASE/manual_tt_7ch.pkl

echo "=== hyp_tuning start  host=$(hostname)  cwd=$(pwd)  time=$(date -Is) ==="
python3 -u "$HYP_TUNING" "$DATASET"
RC=$?
echo "=== end  rc=$RC  time=$(date -Is) ==="
ls -la hps.pkl checkpoint.pkl hyp_tuning.log 2>&1
