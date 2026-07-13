#!/bin/bash
# Train ONE (m3, fold) on v9 (783 rxns), 5 members serially.
# Targets fastest GPU (a6000ada = gpu3) preferentially.

#SBATCH --job-name=v9m3_train
#SBATCH --partition=gpu3,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v9m3_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/train_v9m3_%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

: "${FOLD:?FOLD required (0..4)}"
BASELINE=xtb_geom6_plus_v2

echo "[$(date +%H:%M:%S)] v9 m3 fold=$FOLD  node=$(hostname -s)  gpu=$SLURMD_NODENAME"
nvidia-smi -L 2>/dev/null | head -1

# Symlink v9 bundle + splits into paths runner expects
mkdir -p m3/code/bundles outputs/asr_v1/phase3/subsamples
BROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9
SROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9
ln -sf "$BROOT/features_v6_delta_m3.pt" "m3/code/bundles/features_v6_delta_${BASELINE}.pt"
ln -sf "$BROOT/features_v6_delta_m3.families.json" "m3/code/bundles/features_v6_delta_${BASELINE}.families.json"
ln -sfn "$SROOT" outputs/asr_v1/phase3/subsamples/v9_all

export BASELINE
export SUBSAMPLES_TAG=v9_all
export OUT_TAG=lowlr_v9
export EPOCHS_MAX=100000
export PATIENCE=10000
export LR_LOW=1e-5
export SIZE_FULL=0

RUNNER=m3/code/runner_lowlr_trackB_m1delta.py
n_ok=0; n_fail=0
for MEM in 0 1 2 3 4; do
  OUT="m3/code/trackB_${OUT_TAG}_${BASELINE}/m1_delta/fold${FOLD}/member${MEM}.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP m3 fold=$FOLD member=$MEM"
    n_ok=$((n_ok+1)); continue
  fi
  echo "[$(date +%H:%M:%S)] START m3 fold=$FOLD member=$MEM"
  python -u "$RUNNER" --fold "$FOLD" --member "$MEM" 2>&1 | tail -30
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] OK    m3 fold=$FOLD member=$MEM"; n_ok=$((n_ok+1))
  else
    echo "[$(date +%H:%M:%S)] FAIL  m3 fold=$FOLD member=$MEM"; n_fail=$((n_fail+1))
  fi
done
echo "[$(date +%H:%M:%S)] m3 fold=$FOLD done: ok=$n_ok fail=$n_fail"
