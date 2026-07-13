#!/bin/bash
#SBATCH --job-name=regen_m3
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/regen_m3.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/regen_m3.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python -u scripts/v9_ml/regen_m3_figures_v9.py
