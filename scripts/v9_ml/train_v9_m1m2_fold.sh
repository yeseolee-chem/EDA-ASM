#!/bin/bash
# Train ONE (mk, fold) on v9 (783 rxns), 5 members serially.
# Env: MK (m1 or m2), FOLD (0..4)

#SBATCH --job-name=v9m12_tr
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v9m12_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v9m12_%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

: "${MK:?MK required (m1 or m2)}"
: "${FOLD:?FOLD required (0..4)}"
case "$MK" in
  m1) BASELINE=geom6 ;;
  m2) BASELINE=xtb_geom6 ;;
  *) echo "unknown MK $MK"; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] v9 $MK fold=$FOLD BASELINE=$BASELINE node=$(hostname -s)"

BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9
mkdir -p "models/$MK/code/bundles" outputs/asr_v1/phase3/subsamples
ln -sf "$BROOT/features_v6_delta_${MK}.pt" "models/$MK/code/bundles/features_v6_delta_${BASELINE}.pt"
ln -sf "$BROOT/features_v6_delta_${MK}.families.json" "models/$MK/code/bundles/features_v6_delta_${BASELINE}.families.json"
ln -sfn "$SROOT" outputs/asr_v1/phase3/subsamples/v9_all

export BASELINE SUBSAMPLES_TAG=v9_all OUT_TAG=lowlr_v9 EPOCHS_MAX=100000 PATIENCE=10000 LR_LOW=1e-5 SIZE_FULL=0
RUNNER=models/$MK/code/runner_lowlr_trackB_m1delta.py
n_ok=0; n_fail=0
for MEM in 0 1 2 3 4; do
  OUT="models/$MK/code/trackB_${OUT_TAG}_${BASELINE}/m1_delta/fold${FOLD}/member${MEM}.json"
  if [ -f "$OUT" ]; then echo "SKIP $MK fold=$FOLD member=$MEM"; n_ok=$((n_ok+1)); continue; fi
  echo "[$(date +%H:%M:%S)] START $MK fold=$FOLD member=$MEM"
  python -u "$RUNNER" --fold "$FOLD" --member "$MEM" 2>&1 | tail -30
  [ -f "$OUT" ] && { echo "OK $MK fold=$FOLD member=$MEM"; n_ok=$((n_ok+1)); } || { echo "FAIL $MK fold=$FOLD member=$MEM"; n_fail=$((n_fail+1)); }
done
echo "$MK fold=$FOLD done: ok=$n_ok fail=$n_fail"
