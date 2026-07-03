#!/bin/bash
# MACE-OFF LARGE @ N=250 — capacity-bump sanity check on the dipolar set.
# Same pipeline as submit_maceoff_n250.sh; differs only by model size (large
# vs medium). Decision: adopt large only if overall MAE improves >0.5 kcal/mol
# AND hard channels (Pauli, E_orb) drop noticeably vs medium n250 baseline.
#
# Usage: sbatch scripts/asr_v1/submit_maceoff_large_n250.sh

#SBATCH --job-name=asr_v1_mace_large_n250
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=outputs/asr_v1/logs/mace-large-n250-%j.out
#SBATCH --error=outputs/asr_v1/logs/mace-large-n250-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

CFG="configs/asr_v1_maceoff_large_n250.yaml"
FEAT="outputs/asr_v1/features_dipolar_maceoff_large_n250.pt"

OUT_B0="outputs/asr_v1/maceoff_large_b0_n250"
OUT_M1="outputs/asr_v1/maceoff_large_m1_n250"
OUT_LC_B0="outputs/asr_v1/maceoff_large_learning_curve_b0_n250"
OUT_LC_M1="outputs/asr_v1/maceoff_large_learning_curve_m1_n250"

echo "[$(date)] === MACE-OFF LARGE N=250 (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo
echo ">>> 1/5: cache MACE-OFF LARGE features (N=250) <<<"
python -u scripts/asr_v1/cache_features_maceoff.py --config "$CFG" --model-size large --device cuda --out "$FEAT"

echo
echo ">>> 2/5: train_cv B0 (LARGE) <<<"
python -u scripts/asr_v1/train_cv.py --config "$CFG" --model b0 --features "$FEAT" --output-dir "$OUT_B0"

echo
echo ">>> 3/5: train_cv M1 (LARGE) <<<"
python -u scripts/asr_v1/train_cv.py --config "$CFG" --model m1 --features "$FEAT" --output-dir "$OUT_M1"

echo
echo ">>> 4/5: learning_curve B0 (LARGE) <<<"
python -u scripts/asr_v1/learning_curve.py --config "$CFG" --model b0 --features "$FEAT" --output-dir "$OUT_LC_B0"

echo
echo ">>> 5/5: learning_curve M1 (LARGE) <<<"
python -u scripts/asr_v1/learning_curve.py --config "$CFG" --model m1 --features "$FEAT" --output-dir "$OUT_LC_M1"

echo
echo "[$(date)] === MACE-OFF LARGE N=250 done ==="
