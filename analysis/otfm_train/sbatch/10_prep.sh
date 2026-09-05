#!/bin/bash
# SPEC17rev2 Step 10 (prep) — stratified-sample + build 200×5 ORCA inputs.
#
#SBATCH --job-name=otfm10p
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/10p_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$BASE"
python 10_channel_impact.py --stage sample && \
python 10_channel_impact.py --stage inputs
