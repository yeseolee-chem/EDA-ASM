#!/bin/bash
#SBATCH --job-name=otfm08
#SBATCH --time=48:00:00
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/08_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export BATCH_SIZE="${BATCH_SIZE:-8}"
export SAMPLER_MAX_NUM="${SAMPLER_MAX_NUM:-1600}"

cd "$BASE"
python 08_train_final.py
