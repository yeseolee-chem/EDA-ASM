#!/bin/bash
# SPEC17rev2 Step 7 — cross-fit TS generation, one array task per fold.
#
#SBATCH --job-name=otfm07
#SBATCH --time=48:00:00
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --array=0-4%5
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/07_%A_%a.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export SAMPLER_NFE="${SAMPLER_NFE:-25}"

cd "$BASE"
python 07_generate_crossfit.py --fold "$SLURM_ARRAY_TASK_ID"
