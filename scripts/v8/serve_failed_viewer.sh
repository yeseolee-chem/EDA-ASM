#!/bin/bash
#SBATCH --job-name=serve_view
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/serve_view.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/serve_view.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

PORT="${VIEWER_PORT:-5578}"
NODE="$(hostname -s)"
DIR=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v8_review

echo "================================================================"
echo " failed rxn viewer on: $NODE:$PORT"
echo " serving:  $DIR"
echo " URL:      http://<node>:$PORT/failed_viewer.html"
echo "================================================================"

cd "$DIR"
exec python -u -m http.server "$PORT" --bind 0.0.0.0
