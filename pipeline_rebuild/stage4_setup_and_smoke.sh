#!/bin/bash
#SBATCH --job-name=st4_smoke
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=pipeline_rebuild/logs/st4_%j.out
#SBATCH --error=pipeline_rebuild/logs/st4_%j.err

# Stage 4 — link the Stage-3 bundle/splits into the paths the runner expects,
# then train fold 0 member 0 of m1 (geom6) as an end-to-end smoke test.

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

BUNDLE_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles/features_v6_delta_geom6.pt
FAMS_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles/features_v6_delta_geom6.families.json
SPLITS_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples/trackB_no_ood

# 1) symlink bundle into m1/code/bundles/
mkdir -p m1/code/bundles
ln -sf "$BUNDLE_SRC"  m1/code/bundles/features_v6_delta_geom6.pt
ln -sf "$FAMS_SRC"    m1/code/bundles/features_v6_delta_geom6.families.json

# 2) symlink fold splits into outputs/asr_v1/phase3/subsamples/trackB_no_ood/
mkdir -p outputs/asr_v1/phase3/subsamples
ln -sfn "$SPLITS_SRC" outputs/asr_v1/phase3/subsamples/trackB_no_ood

echo "[$(date)] paths linked:"
ls -la m1/code/bundles/
ls -la outputs/asr_v1/phase3/subsamples/trackB_no_ood/

# 3) Fold 0 member 0 smoke test
export BASELINE=geom6
export SUBSAMPLES_TAG=trackB_no_ood
export OUT_TAG=lowlr_no_ood

# Cap epochs low for the smoke test — real run uses 100k.
export EPOCHS_MAX=1500
export PATIENCE=200

# The runner reads --fold/--member via argparse (m3 flow) — force it.
echo "[$(date)] running m1/code/runner_lowlr_trackB_m1delta.py --fold 0 --member 0"
python -u m1/code/runner_lowlr_trackB_m1delta.py --fold 0 --member 0 || echo "!! runner failed (smoke)"

echo "[$(date)] Stage 4 smoke test done."
ls -la m1/code/trackB_lowlr_no_ood_geom6/m1_delta/fold0/ 2>/dev/null || echo "no fold0 output"
