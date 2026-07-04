#!/bin/bash
# m2 (d1..d21 baseline) — spec: LR 1e-5, EPOCHS 100k, PATIENCE 10k.
# 5 array tasks × 5 members = 25 cells.

#SBATCH --job-name=sv1_m2
#SBATCH --array=0-4%3
#SBATCH --partition=gpu1,gpu4,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/m2_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/m2_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v1
mkdir -p m2/code/bundles outputs/asr_v1/phase3/subsamples
# BASELINE=xtb_geom6 → runner reads bundles/features_v6_delta_xtb_geom6.pt
ln -sf "$BROOT/features_v6_delta_m2.pt"                  m2/code/bundles/features_v6_delta_xtb_geom6.pt
ln -sf "$BROOT/features_v6_delta_m2.families.json"       m2/code/bundles/features_v6_delta_xtb_geom6.families.json
ln -sfn "$SROOT/trackB_no_ood"                            outputs/asr_v1/phase3/subsamples/trackB_no_ood

export BASELINE=xtb_geom6
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=lowlr_no_ood
export EPOCHS_MAX=100000
export PATIENCE=10000

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python -u m2/code/runner_lowlr_trackB_m1delta.py
