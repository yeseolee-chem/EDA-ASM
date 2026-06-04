#!/bin/bash
# Re-run ASR v1 — MACE-OFF only @ N=250 (asr_labels.parquet reparsed
# 2026-06-04 after disk had 250 5/5-converged dipolar reactions).
# ep29 NequIP is NOT re-run: the prior N=134 backbone comparison already
# established MACE-OFF as the winner on hard components.
#
# Pipeline:
#   1. cache MACE-OFF features (N=250)
#   2. train_cv B0
#   3. train_cv M1
#   4. learning_curve B0
#   5. learning_curve M1
#
# Outputs land under outputs/asr_v1/maceoff_{b0,m1,learning_curve_b0,
# learning_curve_m1}_n250/summary.json.
#
# Usage: sbatch scripts/asr_v1/submit_maceoff_n250.sh

#SBATCH --job-name=asr_v1_mace_n250
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/asr_v1/logs/mace-n250-%j.out
#SBATCH --error=outputs/asr_v1/logs/mace-n250-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

CFG="configs/asr_v1_maceoff_n250.yaml"
FEAT="outputs/asr_v1/features_dipolar_maceoff_medium_n250.pt"

OUT_B0="outputs/asr_v1/maceoff_b0_n250"
OUT_M1="outputs/asr_v1/maceoff_m1_n250"
OUT_LC_B0="outputs/asr_v1/maceoff_learning_curve_b0_n250"
OUT_LC_M1="outputs/asr_v1/maceoff_learning_curve_m1_n250"

echo "[$(date)] === MACE-OFF N=250 re-run (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo
echo ">>> 1/5: cache MACE-OFF features (N=250) <<<"
python -u scripts/asr_v1/cache_features_maceoff.py --config "$CFG" --model-size medium --device cuda --out "$FEAT"

echo
echo ">>> 2/5: train_cv B0 (MACE-OFF) <<<"
python -u scripts/asr_v1/train_cv.py --config "$CFG" --model b0 --features "$FEAT" --output-dir "$OUT_B0"

echo
echo ">>> 3/5: train_cv M1 (MACE-OFF) <<<"
python -u scripts/asr_v1/train_cv.py --config "$CFG" --model m1 --features "$FEAT" --output-dir "$OUT_M1"

echo
echo ">>> 4/5: learning_curve B0 (MACE-OFF) <<<"
python -u scripts/asr_v1/learning_curve.py --config "$CFG" --model b0 --features "$FEAT" --output-dir "$OUT_LC_B0"

echo
echo ">>> 5/5: learning_curve M1 (MACE-OFF) <<<"
python -u scripts/asr_v1/learning_curve.py --config "$CFG" --model m1 --features "$FEAT" --output-dir "$OUT_LC_M1"

echo
echo "[$(date)] === MACE-OFF N=250 re-run done ==="
