#!/bin/bash
#SBATCH --job-name=frag_review
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/frag_review.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/frag_review.%j.err

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Ensure Flask is installed (add to conda env if missing).
python -c "import flask" 2>/dev/null || pip install --quiet flask

REVIEW_PORT="${REVIEW_PORT:-5788}"
# Visualise everything as R (reactant geometry) per user request.
# Fragmentation itself is still computed with family-specific logic
# (dipolar/rgd1 at R, qmrxn20 at TS) — that decoupling is in
# refine_fragments.py, this env var only controls what the 3D viewer shows.
VIEW_GEOM="${VIEW_GEOM:-R}"
export REVIEW_PORT VIEW_GEOM

NODE="$(hostname -s)"
echo "================================================================"
echo " Fragment review app starting on compute node: $NODE"
echo " Port:                                          $REVIEW_PORT"
echo ""
echo " Port-forward from your LAPTOP terminal:"
echo "     ssh -N -L ${REVIEW_PORT}:${NODE}:${REVIEW_PORT} yeseo1ee@gate1.hpc"
echo ""
echo " Then open in your browser:"
echo "     http://localhost:${REVIEW_PORT}"
echo ""
echo " Auto-save target:"
echo "     /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/frag_review/manual_partitions.json"
echo "================================================================"

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python scripts/frag_review_app.py
