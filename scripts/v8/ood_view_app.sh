#!/bin/bash
#SBATCH --job-name=ood_view
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --nodelist=n116
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/ood_view.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/ood_view.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export FIX_PORT="${FIX_PORT:-5578}"
export FILTER_FILE="${FILTER_FILE:-}"
NODE="$(hostname -s)"
echo "================================================================"
echo " OOD investigation viewer on: $NODE:$FIX_PORT"
echo "================================================================"
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python -u scripts/v8/ood_view_app.py
