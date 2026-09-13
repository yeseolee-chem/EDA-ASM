#!/bin/bash
# GUESS-mode TS generation via cross-fit. Array 0-4%5 covers the 5 folds.
#
# Submit example:
#   sbatch --partition=gpu3,gpu4,gpu5 --array=0-4%5 sbatch/06_generate_guess.sh
#
#SBATCH --job-name=guess06
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess/logs/06_%A_%a.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export SAMPLER_NFE="${SAMPLER_NFE:-25}"

cd "$BASE"
python 06_generate_guess.py --fold "${SLURM_ARRAY_TASK_ID:-0}"
