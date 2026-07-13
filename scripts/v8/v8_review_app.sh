#!/bin/bash
#SBATCH --job-name=v8_review
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v8_review.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v8_review.%j.err

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -c "import flask" 2>/dev/null || pip install --quiet flask

REVIEW_PORT="${REVIEW_PORT:-5788}"
export REVIEW_PORT

NODE="$(hostname -s)"
echo "================================================================"
echo " v8 Fragment review app on compute node: $NODE"
echo " Port:                                   $REVIEW_PORT"
echo ""
echo " Laptop SSH tunnel:"
echo "     ssh -N -L ${REVIEW_PORT}:${NODE}:${REVIEW_PORT} yeseo1ee@<gate1_hostname>"
echo " Browser:  http://localhost:${REVIEW_PORT}"
echo ""
echo " State: outputs/v8_review/manual_partitions.json"
echo " ORCA:  outputs/v8_review/orca_inputs/{rid}/eda.inp"
echo "================================================================"

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python scripts/v8/v8_review_app.py
