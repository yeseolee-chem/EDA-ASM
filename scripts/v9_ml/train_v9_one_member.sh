#!/bin/bash
# Train ONE (mk, fold, member) — single member, no serial loop.
# Env: MK, FOLD, MEMBER

#SBATCH --job-name=v9_1mem
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_1mem_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_1mem_%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

: "${MK:?MK required (m1|m2|m3)}"
: "${FOLD:?FOLD required (0..4)}"
: "${MEMBER:?MEMBER required (0..4)}"
case "$MK" in
  m1) BASELINE=geom6 ;;
  m2) BASELINE=xtb_geom6 ;;
  m3) BASELINE=xtb_geom6_plus_v2 ;;
esac

BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9
mkdir -p "models/$MK/code/bundles" outputs/asr_v1/phase3/subsamples
ln -sf "$BROOT/features_v6_delta_${MK}.pt" "models/$MK/code/bundles/features_v6_delta_${BASELINE}.pt"
ln -sf "$BROOT/features_v6_delta_${MK}.families.json" "models/$MK/code/bundles/features_v6_delta_${BASELINE}.families.json"
ln -sfn "$SROOT" outputs/asr_v1/phase3/subsamples/v9_all
export BASELINE SUBSAMPLES_TAG=v9_all OUT_TAG=lowlr_v9 EPOCHS_MAX=100000 PATIENCE=10000 LR_LOW=1e-5 SIZE_FULL=0

OUT="models/$MK/code/trackB_${OUT_TAG}_${BASELINE}/m1_delta/fold${FOLD}/member${MEMBER}.json"
if [ -f "$OUT" ]; then echo "SKIP (already done)"; exit 0; fi

echo "[$(date +%H:%M:%S)] START $MK fold=$FOLD member=$MEMBER on $(hostname -s)"
python -u "models/$MK/code/runner_lowlr_trackB_m1delta.py" --fold "$FOLD" --member "$MEMBER"
