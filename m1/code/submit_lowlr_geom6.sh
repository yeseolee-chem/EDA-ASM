#!/bin/bash
# m1 Track B / geom6 baseline — 100k epochs, 10k patience.
# 5 array tasks, each runs all 5 members for one fold (25 cells total).

#SBATCH --job-name=m1_geom6
#SBATCH --array=0-4%5
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=m1/logs/m1_geom6-%A_%a.out
#SBATCH --error=m1/logs/m1_geom6-%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p m1/logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export BASELINE=geom6
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=lowlr_no_ood

# Symlinks are created by the smoke test — re-assert them defensively:
BUNDLE_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles/features_v6_delta_geom6.pt
FAMS_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles/features_v6_delta_geom6.families.json
SPLITS_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples/trackB_no_ood
mkdir -p m1/code/bundles outputs/asr_v1/phase3/subsamples
ln -sf "$BUNDLE_SRC"  m1/code/bundles/features_v6_delta_geom6.pt
ln -sf "$FAMS_SRC"    m1/code/bundles/features_v6_delta_geom6.families.json
ln -sfn "$SPLITS_SRC" outputs/asr_v1/phase3/subsamples/trackB_no_ood

echo "[$(date)] === m1 geom6 task ${SLURM_ARRAY_TASK_ID} start on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

python -u m1/code/runner_lowlr_trackB_m1delta.py

echo "[$(date)] === m1 geom6 task ${SLURM_ARRAY_TASK_ID} done ==="
