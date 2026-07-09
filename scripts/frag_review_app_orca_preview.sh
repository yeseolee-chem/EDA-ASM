#!/bin/bash
#SBATCH --job-name=frag_review_orca
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/frag_review_orca.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/frag_review_orca.%j.err
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
python -c "import flask" 2>/dev/null || pip install --quiet flask
REVIEW_PORT="${REVIEW_PORT:-5789}"
VIEW_GEOM=TS
PART_FILE=orca_inp_partitions.json
export REVIEW_PORT VIEW_GEOM PART_FILE
NODE="$(hostname -s)"
echo "================================================================"
echo " ORCA input file visualization on compute node: $NODE"
echo " Port:                                          $REVIEW_PORT"
echo " Showing: TS geometry + fragment assignment as-is from eda.inp"
echo ""
echo " Port-forward from your laptop:"
echo "     ssh -N -L ${REVIEW_PORT}:${NODE}:${REVIEW_PORT} gate1"
echo " Then open: http://localhost:${REVIEW_PORT}"
echo "================================================================"
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python scripts/frag_review_app.py
