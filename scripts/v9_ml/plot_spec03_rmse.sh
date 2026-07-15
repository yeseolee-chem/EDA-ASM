#!/bin/bash
#SBATCH --job-name=spec03_rmse
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec03_rmse.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec03_rmse.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python -u spec/spec03_bmax/code/plot_rmse_bars.py
