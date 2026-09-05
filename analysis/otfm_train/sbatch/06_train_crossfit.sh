#!/bin/bash
# SPEC17rev2 Step 6 — cross-fit training, one array task per fold.
# 5 array tasks × 1 GPU. Fits under SLURM 10-running / 20-submit caps.
#
#SBATCH --job-name=otfm06
#SBATCH --time=48:00:00
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --array=0-4%5
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/06_%A_%a.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# tune BATCH_SIZE / SAMPLER_MAX_NUM here if OOM — spec §8 warns Coley
# median-44-atom rxns push node^2 samplers past 16 GB defaults.
export BATCH_SIZE="${BATCH_SIZE:-8}"
export SAMPLER_MAX_NUM="${SAMPLER_MAX_NUM:-1600}"

cd "$BASE"
python 06_train_crossfit.py --fold "$SLURM_ARRAY_TASK_ID"
