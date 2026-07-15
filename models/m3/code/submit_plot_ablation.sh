#!/bin/bash
# Regenerate m3 ablation figures (matplotlib-only, quick).
#SBATCH --job-name=plot_abl_m3
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/plot_abl_m3_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/plot_abl_m3_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === regenerating m3 ablation figures ==="
python -u models/m3/code/plot_ablation.py
echo "[$(date)] === done ==="
