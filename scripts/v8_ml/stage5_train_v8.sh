#!/bin/bash
# Stage 5 (v8) — unified 3-model × 5-fold × 5-member training launcher.
#
# 75 cells total = 3 models × 5 folds × 5 members.
# Task ID mapping (SLURM_ARRAY_TASK_ID in [0, 74]):
#   model_idx  = TASK / 25          (0 = m1, 1 = m2, 2 = m3)
#   cell       = TASK % 25
#   fold       = cell / 5           (0..4)
#   member     = cell % 5           (0..4)
#
# The existing runner (m{k}/code/runner_lowlr_trackB_m1delta.py) processes ONE
# fold's members at a time (loops 0..4 internally when --member is omitted).
# We call it with an explicit --fold and --member so each array task = 1 cell.
#
# NO OOD filtering. SUBSAMPLES_TAG=v8_all points at subsamples_v8/ which was
# built by stage4 from all 799 reactions.
#
# Output dir per model (as written by the runner):
#   m1/code/trackB_lowlr_v8_geom6/m1_delta/fold{F}/member{M}.json
#   m2/code/trackB_lowlr_v8_xtb_geom6/m1_delta/fold{F}/member{M}.json
#   m3/code/trackB_lowlr_v8_xtb_geom6_plus_v2/m1_delta/fold{F}/member{M}.json

#SBATCH --job-name=v8_train
#SBATCH --array=0-74%15
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v8_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v8_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$((TASK / 25))
CELL=$((TASK % 25))
FOLD=$((CELL / 5))
MEMBER=$((CELL % 5))

# ---- pick model + baseline name ----
case "$MODEL_IDX" in
  0) MODEL=m1; BASELINE=geom6            ;;
  1) MODEL=m2; BASELINE=xtb_geom6        ;;
  2) MODEL=m3; BASELINE=xtb_geom6_plus_v2;;
  *) echo "bad MODEL_IDX=$MODEL_IDX (TASK=$TASK)"; exit 2 ;;
esac
echo "[dispatch] TASK=$TASK -> MODEL=$MODEL BASELINE=$BASELINE FOLD=$FOLD MEMBER=$MEMBER"

# ---- symlink v8 bundle + splits into runner-expected paths ----
BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v8

mkdir -p "$MODEL/code/bundles" outputs/asr_v1/phase3/subsamples
ln -sf "$BROOT/features_v6_delta_${MODEL}.pt" \
       "$MODEL/code/bundles/features_v6_delta_${BASELINE}.pt"
ln -sf "$BROOT/features_v6_delta_${MODEL}.families.json" \
       "$MODEL/code/bundles/features_v6_delta_${BASELINE}.families.json"
# The runner reads $ROOT/outputs/asr_v1/phase3/subsamples/$SUBSAMPLES_TAG.
# NEW tag = v8_all (renamed from the trackB_no_ood folder). No OOD filtering.
ln -sfn "$SROOT" outputs/asr_v1/phase3/subsamples/v8_all

# ---- training hyperparameters (spec) ----
export BASELINE
export SUBSAMPLES_TAG=v8_all
export OUT_TAG=lowlr_v8
export EPOCHS_MAX=100000
export PATIENCE=10000
export LR_LOW=1e-5
# SIZE_FULL default in runner is 509. Our splits use full pool (~640/fold),
# named size_{N}.json. Setting SIZE_FULL=0 forces the "pick largest size_*.json"
# fallback, which correctly grabs the whole train pool for each fold.
export SIZE_FULL=0

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# One cell per array task.
python -u "${MODEL}/code/runner_lowlr_trackB_m1delta.py" \
    --fold "$FOLD" --member "$MEMBER"
