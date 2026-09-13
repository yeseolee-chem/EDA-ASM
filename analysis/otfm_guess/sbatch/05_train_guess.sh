#!/bin/bash
# GUESS-mode training. Array 0-4%5 covers the 5 folds.
#
# Submit example:
#   sbatch --partition=gpu3,gpu4,gpu5 \
#     --export=ALL,SAMPLER_MAX_NUM=8000,BATCH_SIZE=8,\
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     --array=0-4%5 sbatch/05_train_guess.sh
#
#SBATCH --job-name=guess05
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess/logs/05_%A_%a.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export SAMPLER_MAX_NUM="${SAMPLER_MAX_NUM:-8000}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$BASE"
python 05_train_guess.py --fold "${SLURM_ARRAY_TASK_ID:-0}"
