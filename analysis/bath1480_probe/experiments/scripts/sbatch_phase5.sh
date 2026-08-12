#!/bin/bash
# Phase 5: aggregate + figures.
#
#SBATCH --job-name=exp_p5
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase5.%j.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
echo "=== Phase 5 start=$(date -Is) ==="
python3 -u $SCRIPTS/06_aggregate.py
echo "=== end $(date -Is) ==="
