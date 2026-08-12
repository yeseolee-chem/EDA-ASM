#!/bin/bash
# Phase 0: scratch cleanup + cohort/labels/features build.
# Runs immediately after strain 48h wall hit.
#
#SBATCH --job-name=exp_p0
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase0.%j.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
python3 -u $SCRIPTS/00_build_cohort.py
echo "=== Phase 0 done at $(date -Is) ==="
