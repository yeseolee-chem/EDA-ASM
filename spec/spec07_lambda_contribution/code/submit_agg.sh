#!/bin/bash
# SPEC_07 — aggregate 5-fold OOF JSONs across all λ + render figures.
# Cheap CPU job, but per CLAUDE.md we still submit via sbatch.

#SBATCH --job-name=s7_agg
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_agg_%A.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_agg_%A.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)"

python -u spec/spec07_lambda_contribution/code/aggregate.py
python -u spec/spec07_lambda_contribution/code/plot_lambda.py

echo "[done] results/ + figures/ populated."
ls -la spec/spec07_lambda_contribution/results/
ls -la spec/spec07_lambda_contribution/figures/
