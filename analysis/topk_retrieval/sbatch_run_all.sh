#!/bin/bash
# spec13_topk_retrieval — full pipeline (5 steps, < 5 min total).
# Reuses OOF predictions from delta_mae_baseline. No new QM.
#
#SBATCH --job-name=topk_retr
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval/artifacts/run.%j.log

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval
cd "$BASE"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== spec13 topk_retrieval start $(date -Is) ==="

for STEP in 00_setup.py 01_build_candidates.py 02_topk.py 03_figures.py 04_report.py; do
    echo ""
    echo "--- $STEP ---"
    python3 -u "$BASE/$STEP" || { echo "ABORT: $STEP failed rc=$?"; exit 1; }
done

echo ""
echo "=== spec13 DONE $(date -Is) ==="
