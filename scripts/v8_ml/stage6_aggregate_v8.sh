#!/bin/bash
# Stage 6 (v8) — aggregate 3 × 5 × 5 = 75 cell JSONs into NMAE / RMSE / parity
# plots + summary CSVs. Single non-array job on cpu2.

#SBATCH --job-name=v8_stage6
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v8_stage6_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v8_stage6_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
mkdir -p outputs/v8_review/results

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u scripts/v8_ml/stage6_aggregate_v8.py
