#!/bin/bash
#SBATCH --job-name=tt_ml_repro
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_repro/logs/mlrun.%A_%a.log

set -uo pipefail

SEEDS=(22 23 14 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_repro
DATASET=$BASE/data/manual_tt_solvent.pkl
HPS=$BASE/data/hps.pkl
ML_SCRIPT=$BASE/scripts/ml_analysis.py

WORKDIR=$BASE/results/seed_$SEED
mkdir -p "$WORKDIR"
cd "$WORKDIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro

# Suppress TF chatter
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

echo "=== seed=$SEED  host=$(hostname)  cwd=$(pwd)  start=$(date -Is) ==="

# Idempotency: ml_analysis.py itself checks ml_results.pkl for completed models
# and skips them. So safe to re-run this script.
python -u "$ML_SCRIPT" $SEED "$DATASET" "$HPS"
RC=$?

echo "=== seed=$SEED  rc=$RC  end=$(date -Is) ==="
if [ $RC -ne 0 ]; then
    echo "FAIL"
    exit $RC
fi
ls -la ml_results.pkl 2>&1
echo "OK"
