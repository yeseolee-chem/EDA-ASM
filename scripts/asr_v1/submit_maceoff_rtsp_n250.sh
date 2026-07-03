#!/bin/bash
# RTSP pipeline @ N=250 — MACE-OFF medium with (R, TS, P) input.
# Tests the hypothesis that explicitly supplying the DFT-converged TS
# geometry breaks the ~9.4 kcal/mol overall-MAE plateau hit by the R/P-only
# medium model on the same N=250 dipolar set.
#
# Pipeline:
#   1. cache MACE-OFF features for R, TS, P (N=250)
#   2. train_cv B0_RTSP
#   3. train_cv M1_RTSP
#   4. learning_curve B0_RTSP
#   5. learning_curve M1_RTSP

#SBATCH --job-name=asr_v1_mace_rtsp_n250
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=outputs/asr_v1/logs/mace-rtsp-n250-%j.out
#SBATCH --error=outputs/asr_v1/logs/mace-rtsp-n250-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

CFG="configs/asr_v1_maceoff_rtsp_n250.yaml"
FEAT="outputs/asr_v1/features_dipolar_maceoff_medium_rtsp_n250.pt"

OUT_B0="outputs/asr_v1/maceoff_b0_rtsp_n250"
OUT_M1="outputs/asr_v1/maceoff_m1_rtsp_n250"
OUT_LC_B0="outputs/asr_v1/maceoff_learning_curve_b0_rtsp_n250"
OUT_LC_M1="outputs/asr_v1/maceoff_learning_curve_m1_rtsp_n250"

echo "[$(date)] === MACE-OFF RTSP N=250 (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo
echo ">>> 1/5: cache MACE-OFF features (R, TS, P) <<<"
python -u scripts/asr_v1/cache_features_maceoff_rtsp.py --config "$CFG" --model-size medium --device cuda --out "$FEAT"

echo
echo ">>> 2/5: train_cv B0_RTSP <<<"
python -u scripts/asr_v1/train_cv_rtsp.py --config "$CFG" --model b0_rtsp --features "$FEAT" --output-dir "$OUT_B0"

echo
echo ">>> 3/5: train_cv M1_RTSP <<<"
python -u scripts/asr_v1/train_cv_rtsp.py --config "$CFG" --model m1_rtsp --features "$FEAT" --output-dir "$OUT_M1"

echo
echo ">>> 4/5: learning_curve B0_RTSP <<<"
python -u scripts/asr_v1/learning_curve_rtsp.py --config "$CFG" --model b0_rtsp --features "$FEAT" --output-dir "$OUT_LC_B0"

echo
echo ">>> 5/5: learning_curve M1_RTSP <<<"
python -u scripts/asr_v1/learning_curve_rtsp.py --config "$CFG" --model m1_rtsp --features "$FEAT" --output-dir "$OUT_LC_M1"

echo
echo "[$(date)] === MACE-OFF RTSP N=250 done ==="
