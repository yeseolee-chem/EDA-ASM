#!/bin/bash
#SBATCH --job-name=fix_single
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/fix_single.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/fix_single.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export RID="${RID:-dipolar_000658}"
export FIX_PORT="${FIX_PORT:-5578}"

NODE="$(hostname -s)"
echo "================================================================"
echo " fix single-rxn app on: $NODE:$FIX_PORT"
echo " editing RID: $RID"
echo "================================================================"

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python -u scripts/v8/fix_single_app.py
