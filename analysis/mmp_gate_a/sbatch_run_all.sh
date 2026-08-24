#!/bin/bash
# spec10_mmp_gate_a — full pipeline. Gates handled inside individual scripts.
#SBATCH --job-name=mmp_gate_a
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a/artifacts/run.%j.log

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a
cd "$BASE"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== spec10_mmp_gate_a start $(date -Is) ==="

for STEP in 00_load_join.py 01_fragment.py 02_validate_library.py 03_validate_fragments.py \
            04_build_mmp.py 05_intersect.py 06_delta_dist.py 07_outlier_audit.py 08_report.py; do
    echo ""
    echo "--- $STEP ---"
    python3 -u "$BASE/$STEP" || { echo "ABORT: $STEP failed rc=$?"; exit 1; }
done

echo ""
echo "=== spec10_mmp_gate_a DONE $(date -Is) ==="
