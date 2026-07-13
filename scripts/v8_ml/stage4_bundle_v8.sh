#!/bin/bash
# Stage 4 (v8) — assemble m1/m2/m3 bundles + 5-fold stratified splits for the
# 799-reaction v8 cohort (no OOD filtering). Single non-array job on cpu2.

#SBATCH --job-name=v8_stage4
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v8_stage4_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v8_stage4_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v8

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u scripts/v8_ml/stage4_bundle_v8.py
