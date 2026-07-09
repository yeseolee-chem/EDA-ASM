#!/bin/bash
# m1 v7 — use ORCA v7 5-channel labels + m1 descriptors (d1..d6, 6-d).
# 5 array tasks × 5 members = 25 cells.

#SBATCH --job-name=m1_v7
#SBATCH --array=0-4%3
#SBATCH --partition=gpu1,gpu3,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/m1_v7_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/m1_v7_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Bind the v7 bundles + splits into the runner's expected paths.
# Wipe old symlinks so cached ones don't leak.
BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v7
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v7
mkdir -p m1/code/bundles outputs/asr_v1/phase3/subsamples
rm -f m1/code/bundles/features_v6_delta_geom6.pt m1/code/bundles/features_v6_delta_geom6.families.json
ln -sf "$BROOT/features_v7_delta_m1.pt"                 m1/code/bundles/features_v6_delta_geom6.pt
ln -sf "$BROOT/features_v7_delta_m1.families.json"      m1/code/bundles/features_v6_delta_geom6.families.json
rm -f outputs/asr_v1/phase3/subsamples/trackB_no_ood
ln -sfn "$SROOT/trackB_no_ood"                           outputs/asr_v1/phase3/subsamples/trackB_no_ood

export BASELINE=geom6
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=v7_lowlr_no_ood
export EPOCHS_MAX=100000
export PATIENCE=10000

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python -u m1/code/runner_lowlr_trackB_m1delta.py
