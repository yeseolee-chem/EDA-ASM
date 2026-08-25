#!/bin/bash
# spec12_pairwise_delta — full pipeline (3 arms × 2 splits × 5 seeds × 5 folds × 7 channels)
#SBATCH --job-name=pw_delta
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta/artifacts/run.%j.log

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta
cd "$BASE"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export XGB_NJOBS=4

echo "=== spec12 pairwise_delta start $(date -Is) ==="

for STEP in 00_build_pairs.py 01_split.py 02_train_arms.py \
            03_invariants.py 04_evaluate.py 05_rho_check.py 06_report.py; do
    echo ""
    echo "--- $STEP ---"
    python3 -u "$BASE/$STEP" || { echo "ABORT: $STEP failed rc=$?"; exit 1; }
done

echo ""
echo "=== spec12 DONE $(date -Is) ==="
