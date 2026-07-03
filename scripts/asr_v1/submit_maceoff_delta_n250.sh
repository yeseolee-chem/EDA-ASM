#!/bin/bash
# Δ-learning pipeline @ N=250 — MACE-OFF medium (R, TS, P) features + a 6-
# descriptor physics ridge baseline. The ML head learns the residual.
#
# Pipeline:
#   1. cache MACE features + descriptors (N=250)
#   2. train_cv B0_Delta
#   3. train_cv M1_Delta
#   4. learning_curve B0_Delta
#   5. learning_curve M1_Delta

#SBATCH --job-name=asr_v1_mace_delta_n250
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=outputs/asr_v1/logs/mace-delta-n250-%j.out
#SBATCH --error=outputs/asr_v1/logs/mace-delta-n250-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

CFG="configs/asr_v1_maceoff_delta_n250.yaml"
FEAT="outputs/asr_v1/features_dipolar_maceoff_medium_delta_n250.pt"

OUT_B0="outputs/asr_v1/maceoff_b0_delta_n250"
OUT_M1="outputs/asr_v1/maceoff_m1_delta_n250"
OUT_LC_B0="outputs/asr_v1/maceoff_learning_curve_b0_delta_n250"
OUT_LC_M1="outputs/asr_v1/maceoff_learning_curve_m1_delta_n250"

echo "[$(date)] === MACE-OFF Δ-learning N=250 (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo
echo ">>> 1/5: cache MACE features + physics descriptors <<<"
python -u scripts/asr_v1/cache_features_maceoff_delta.py --config "$CFG" --model-size medium --device cuda --out "$FEAT"

echo
echo ">>> 2/5: train_cv B0_Delta <<<"
python -u scripts/asr_v1/train_cv_delta.py --config "$CFG" --model b0_delta --features "$FEAT" --output-dir "$OUT_B0"

echo
echo ">>> 3/5: train_cv M1_Delta <<<"
python -u scripts/asr_v1/train_cv_delta.py --config "$CFG" --model m1_delta --features "$FEAT" --output-dir "$OUT_M1"

echo
echo ">>> 4/5: learning_curve B0_Delta <<<"
python -u scripts/asr_v1/learning_curve_delta.py --config "$CFG" --model b0_delta --features "$FEAT" --output-dir "$OUT_LC_B0"

echo
echo ">>> 5/5: learning_curve M1_Delta <<<"
python -u scripts/asr_v1/learning_curve_delta.py --config "$CFG" --model m1_delta --features "$FEAT" --output-dir "$OUT_LC_M1"

echo
echo "[$(date)] === MACE-OFF Δ-learning N=250 done ==="
