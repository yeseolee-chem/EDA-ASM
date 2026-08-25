#!/bin/bash
# spec11_delta_mae_baseline — full pipeline
#SBATCH --job-name=delta_mae
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts/run.%j.log

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline
cd "$BASE"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export XGB_NJOBS=4

echo "=== spec11 delta_mae_baseline start $(date -Is) ==="

for STEP in 00_check.py 01_oof_predict.py 02_delta_mae.py 03_rho_pair.py \
            04_error_corr.py 05_barrier_delta.py 06_report.py; do
    echo ""
    echo "--- $STEP ---"
    python3 -u "$BASE/$STEP" || { echo "ABORT: $STEP failed rc=$?"; exit 1; }
done

echo ""
echo "=== spec11 DONE $(date -Is) ==="
