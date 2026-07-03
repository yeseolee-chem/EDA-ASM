#!/bin/bash
#SBATCH --job-name=st3_bundle
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=pipeline_rebuild/logs/st3_%j.out
#SBATCH --error=pipeline_rebuild/logs/st3_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p pipeline_rebuild/logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === node $(hostname) ==="
python -u pipeline_rebuild/stage3_bundles_and_splits.py
