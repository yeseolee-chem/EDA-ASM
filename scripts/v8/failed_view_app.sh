#!/bin/bash
#SBATCH --job-name=fail_view
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --nodelist=n116
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/fail_view.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/fail_view.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export FIX_PORT="${FIX_PORT:-5578}"
NODE="$(hostname -s)"
echo "================================================================"
echo " Failed-EDA viewer on: $NODE:$FIX_PORT"
echo "================================================================"
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python -u scripts/v8/failed_view_app.py
