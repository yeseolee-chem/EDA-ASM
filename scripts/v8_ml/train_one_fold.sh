#!/bin/bash
# Train ONE (model, fold) pair — loops 5 members serially inside this job.
# Env vars required: MODEL_IDX (0=m1, 1=m2, 2=m3), FOLD (0..4)
#
# SBATCH configured for 48h so all 5 members fit within walltime.

#SBATCH --job-name=v8_train
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v8_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v8_%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

: "${MODEL_IDX:?MODEL_IDX required (0=m1,1=m2,2=m3)}"
: "${FOLD:?FOLD required (0..4)}"

MODEL_NAMES=(m1 m2 m3)
BASELINES=(geom6 xtb_geom6 xtb_geom6_plus_v2)
MK="${MODEL_NAMES[$MODEL_IDX]}"
BASELINE="${BASELINES[$MODEL_IDX]}"

echo "[$(date +%H:%M:%S)] MODEL=$MK  FOLD=$FOLD  BASELINE=$BASELINE  node=$(hostname -s)"

# Symlink v8 bundle + splits into paths runner expects
mkdir -p "$MK/code/bundles" outputs/asr_v1/phase3/subsamples
BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v8
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v8
ln -sf "$BROOT/features_v6_delta_${MK}.pt" "$MK/code/bundles/features_v6_delta_${BASELINE}.pt"
ln -sf "$BROOT/features_v6_delta_${MK}.families.json" "$MK/code/bundles/features_v6_delta_${BASELINE}.families.json"
ln -sfn "$SROOT" outputs/asr_v1/phase3/subsamples/v8_all

export BASELINE
export SUBSAMPLES_TAG=v8_all
export OUT_TAG=lowlr_v8
export EPOCHS_MAX=100000
export PATIENCE=10000
export LR_LOW=1e-5
export SIZE_FULL=0

RUNNER="$MK/code/runner_lowlr_trackB_m1delta.py"
[ ! -f "$RUNNER" ] && { echo "ERROR: $RUNNER missing"; exit 1; }

n_ok=0; n_fail=0
for MEM in 0 1 2 3 4; do
  OUT="$MK/code/trackB_${OUT_TAG}_${BASELINE}/m1_delta/fold${FOLD}/member${MEM}.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $MK fold=$FOLD member=$MEM (exists)"
    n_ok=$((n_ok+1)); continue
  fi
  echo "[$(date +%H:%M:%S)] START $MK fold=$FOLD member=$MEM"
  python -u "$RUNNER" --fold "$FOLD" --member "$MEM" 2>&1 | tail -40
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] OK    $MK fold=$FOLD member=$MEM"
    n_ok=$((n_ok+1))
  else
    echo "[$(date +%H:%M:%S)] FAIL  $MK fold=$FOLD member=$MEM"
    n_fail=$((n_fail+1))
  fi
done

echo "[$(date +%H:%M:%S)] $MK fold=$FOLD done: ok=$n_ok fail=$n_fail"
