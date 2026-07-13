#!/bin/bash
#SBATCH --job-name=rt_review
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/rt_review.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/rt_review.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export FIX_PORT="${FIX_PORT:-5578}"

NODE="$(hostname -s)"
echo "================================================================"
echo " R+TS partition review (258 needs decision) app on: $NODE:$FIX_PORT"
echo " 800 rxn cohort — click any atom to toggle A↔B"
echo "================================================================"

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
exec python -u scripts/v8/rt_review_app.py
