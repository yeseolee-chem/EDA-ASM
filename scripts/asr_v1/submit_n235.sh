#!/bin/bash
# Re-run the ASR v1 backbone comparison on the updated N=235 dipolar label set
# (asr_labels.parquet regenerated 2026-06-03 after disk had 235 5/5-converged
# dipolar reactions; the cached features at N=134 are stale).
#
# Pipeline (single allocation):
#   1. cache ep29 NequIP  features @ N=235
#   2. cache MACE-OFF     features @ N=235
#   3. train_cv  B0+M1 ep29
#   4. train_cv  B0+M1 MACE-OFF
#   5. LC        B0+M1 ep29
#   6. LC        B0+M1 MACE-OFF
#   7. compare_backbones → outputs/asr_v1/backbone_comparison_n235/summary.json
#
# Usage: sbatch scripts/asr_v1/submit_n235.sh

#SBATCH --job-name=asr_v1_n235
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/asr_v1/logs/n235-%j.out
#SBATCH --error=outputs/asr_v1/logs/n235-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

CFG_EP29="configs/asr_v1_n235.yaml"
CFG_MACE="configs/asr_v1_maceoff_n235.yaml"
FEAT_EP29="outputs/asr_v1/features_dipolar_ep29_n235.pt"
FEAT_MACE="outputs/asr_v1/features_dipolar_maceoff_medium_n235.pt"

OUT_EP29_B0="outputs/asr_v1/b0_n235"
OUT_EP29_M1="outputs/asr_v1/m1_n235"
OUT_EP29_LC_B0="outputs/asr_v1/learning_curve_b0_n235"
OUT_EP29_LC_M1="outputs/asr_v1/learning_curve_m1_n235"

OUT_MACE_B0="outputs/asr_v1/maceoff_b0_n235"
OUT_MACE_M1="outputs/asr_v1/maceoff_m1_n235"
OUT_MACE_LC_B0="outputs/asr_v1/maceoff_learning_curve_b0_n235"
OUT_MACE_LC_M1="outputs/asr_v1/maceoff_learning_curve_m1_n235"

OUT_CMP="outputs/asr_v1/backbone_comparison_n235/summary.json"

echo "[$(date)] === asr_v1 N=235 re-run (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo
echo ">>> 1/7: cache ep29 NequIP features (N=235) <<<"
python -u scripts/asr_v1/cache_features.py --config "$CFG_EP29" --output "$FEAT_EP29" --device cuda

echo
echo ">>> 2/7: cache MACE-OFF features (N=235) <<<"
python -u scripts/asr_v1/cache_features_maceoff.py --config "$CFG_MACE" --model-size medium --device cuda --out "$FEAT_MACE"

echo
echo ">>> 3/7: train_cv B0+M1 ep29 <<<"
python -u scripts/asr_v1/train_cv.py --config "$CFG_EP29" --model b0 --features "$FEAT_EP29" --output-dir "$OUT_EP29_B0"
python -u scripts/asr_v1/train_cv.py --config "$CFG_EP29" --model m1 --features "$FEAT_EP29" --output-dir "$OUT_EP29_M1"

echo
echo ">>> 4/7: train_cv B0+M1 MACE-OFF <<<"
python -u scripts/asr_v1/train_cv.py --config "$CFG_MACE" --model b0 --features "$FEAT_MACE" --output-dir "$OUT_MACE_B0"
python -u scripts/asr_v1/train_cv.py --config "$CFG_MACE" --model m1 --features "$FEAT_MACE" --output-dir "$OUT_MACE_M1"

echo
echo ">>> 5/7: learning_curve B0+M1 ep29 <<<"
python -u scripts/asr_v1/learning_curve.py --config "$CFG_EP29" --model b0 --features "$FEAT_EP29" --output-dir "$OUT_EP29_LC_B0"
python -u scripts/asr_v1/learning_curve.py --config "$CFG_EP29" --model m1 --features "$FEAT_EP29" --output-dir "$OUT_EP29_LC_M1"

echo
echo ">>> 6/7: learning_curve B0+M1 MACE-OFF <<<"
python -u scripts/asr_v1/learning_curve.py --config "$CFG_MACE" --model b0 --features "$FEAT_MACE" --output-dir "$OUT_MACE_LC_B0"
python -u scripts/asr_v1/learning_curve.py --config "$CFG_MACE" --model m1 --features "$FEAT_MACE" --output-dir "$OUT_MACE_LC_M1"

echo
echo ">>> 7/7: compare_backbones (ep29 vs MACE-OFF @ N=235) <<<"
python -u scripts/asr_v1/compare_backbones.py \
    --ep29-root outputs/asr_v1 \
    --maceoff-root outputs/asr_v1 \
    --ep29-b0-dir b0_n235 --ep29-m1-dir m1_n235 \
    --ep29-lc-b0-dir learning_curve_b0_n235 --ep29-lc-m1-dir learning_curve_m1_n235 \
    --maceoff-b0-dir maceoff_b0_n235 --maceoff-m1-dir maceoff_m1_n235 \
    --maceoff-lc-b0-dir maceoff_learning_curve_b0_n235 --maceoff-lc-m1-dir maceoff_learning_curve_m1_n235 \
    --out "$OUT_CMP"

echo
echo "[$(date)] === asr_v1 N=235 re-run done ==="
