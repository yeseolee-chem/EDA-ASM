#!/bin/bash
# Backbone-comparison run: MACE-OFF (medium) vs ep29 NequIP baseline.
# Implements ASR_Backbone_Comparison_Spec_v1.0 §10 (single-allocation
# sequence: cache → B0 → M1 → LC B0 → LC M1 → compare).
#
# Pre-req: ep29 outputs already on disk under outputs/asr_v1/{b0,m1,
# learning_curve_b0,learning_curve_m1}/summary.json (they are — June 1 run).
#
# Usage:    sbatch scripts/asr_v1/submit_maceoff.sh
# Walltime: feature extraction is fast (MACE is MD-grade); head training
#           re-uses the same ~30 min budget as the ep29 run.

#SBATCH --job-name=asr_v1_maceoff
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/asr_v1/logs/maceoff-%j.out
#SBATCH --error=outputs/asr_v1/logs/maceoff-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

MODEL_SIZE="${MODEL_SIZE:-medium}"
FEATURES="outputs/asr_v1/features_dipolar_maceoff_${MODEL_SIZE}.pt"
CONFIG="configs/asr_v1_maceoff.yaml"

echo "[$(date)] === MACE-OFF backbone comparison start (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo
echo ">>> Step 1/6: cache MACE-OFF features ($MODEL_SIZE) <<<"
python -u scripts/asr_v1/cache_features_maceoff.py \
    --config "$CONFIG" \
    --model-size "$MODEL_SIZE" \
    --device cuda \
    --out "$FEATURES"

echo
echo ">>> Step 2/6: train_cv B0 (MACE-OFF) <<<"
python -u scripts/asr_v1/train_cv.py \
    --config "$CONFIG" \
    --model b0 \
    --features "$FEATURES" \
    --output-dir outputs/asr_v1/maceoff_b0

echo
echo ">>> Step 3/6: train_cv M1 (MACE-OFF) <<<"
python -u scripts/asr_v1/train_cv.py \
    --config "$CONFIG" \
    --model m1 \
    --features "$FEATURES" \
    --output-dir outputs/asr_v1/maceoff_m1

echo
echo ">>> Step 4/6: learning_curve B0 (MACE-OFF) <<<"
python -u scripts/asr_v1/learning_curve.py \
    --config "$CONFIG" \
    --model b0 \
    --features "$FEATURES" \
    --output-dir outputs/asr_v1/maceoff_learning_curve_b0

echo
echo ">>> Step 5/6: learning_curve M1 (MACE-OFF) <<<"
python -u scripts/asr_v1/learning_curve.py \
    --config "$CONFIG" \
    --model m1 \
    --features "$FEATURES" \
    --output-dir outputs/asr_v1/maceoff_learning_curve_m1

echo
echo ">>> Step 6/6: compare_backbones (ep29 vs MACE-OFF) <<<"
python -u scripts/asr_v1/compare_backbones.py \
    --ep29-root outputs/asr_v1 \
    --maceoff-root outputs/asr_v1 \
    --out outputs/asr_v1/backbone_comparison/summary.json

echo
echo "[$(date)] === MACE-OFF backbone comparison done ==="
