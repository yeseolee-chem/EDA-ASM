#!/bin/bash
# spec10rev_mmp_core_filter — v2 patch. Reuses cohort_join.pkl from v1.
#SBATCH --job-name=mmp_v2
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a/artifacts/run_v2.%j.log

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a
cd "$BASE"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== spec10rev v2 start $(date -Is) ==="

for STEP in 01_fragment_v2.py 02_validate_library_v2.py 04_build_mmp_v2.py \
            05_intersect_v2.py 06_delta_dist_v2.py 07_outlier_audit_v2.py 08_report_v2.py; do
    echo ""
    echo "--- $STEP ---"
    python3 -u "$BASE/$STEP" || { echo "ABORT: $STEP failed rc=$?"; exit 1; }
done

echo ""
echo "=== spec10rev v2 DONE $(date -Is) ==="
