#!/bin/bash
# m1 (geom6 baseline, d1..d6) — spec: LR 1e-5, EPOCHS 100k, PATIENCE 10k,
# grad-clip 5.0, batch 16, wd 1e-3.  5 array tasks × 5 members = 25 cells.

#SBATCH --job-name=sv1_m1
#SBATCH --array=0-4%3
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/m1_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/m1_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Symlink the spec-v1 bundle + splits into the paths the runner expects.
BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v1
mkdir -p m1/code/bundles outputs/asr_v1/phase3/subsamples
# BASELINE=geom6 in the runner will load bundles/features_v6_delta_geom6.pt
ln -sf "$BROOT/features_v6_delta_m1.pt"                  m1/code/bundles/features_v6_delta_geom6.pt
ln -sf "$BROOT/features_v6_delta_m1.families.json"       m1/code/bundles/features_v6_delta_geom6.families.json
ln -sfn "$SROOT/trackB_no_ood"                            outputs/asr_v1/phase3/subsamples/trackB_no_ood

export BASELINE=geom6
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=lowlr_no_ood
# Spec: full budget
export EPOCHS_MAX=100000
export PATIENCE=10000

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python -u m1/code/runner_lowlr_trackB_m1delta.py
