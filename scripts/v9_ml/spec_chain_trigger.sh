#!/bin/bash
# Chained decision trigger after SPEC 1 + SPEC 3 finish:
#   1. Read SPEC 3 baseline_leaderboard.csv
#   2. If xgb is the best per-channel method (majority of 5 channels), sbatch SPEC 5
#   3. Then sbatch SPEC 4 + SPEC 2 (with dep on SPEC 5 if it ran)

#SBATCH --job-name=spec_trig
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec_trig.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec_trig.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

CSV="spec/spec03_bmax/results/baseline_leaderboard.csv"
export BUNDLE_PT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt
export SPLIT_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9

if [ ! -f "$CSV" ]; then
  echo "SPEC 3 result missing → sbatch SPEC 5 anyway (assume xgb wins), then SPEC 4+2"
  XGB_WIN=1
else
  XGB_WIN=$(python -c "
import pandas as pd
df = pd.read_csv('$CSV')
# expect columns: channel, best_method (or similar). count how many channels xgb wins
if 'best_method' in df.columns:
    v = (df['best_method'].astype(str).str.lower()=='xgb').sum()
else:
    # try 'best' column
    for c in ['best','method','model']:
        if c in df.columns:
            v = (df[c].astype(str).str.lower()=='xgb').sum(); break
    else:
        v = 0
print(int(v >= 3))  # xgb wins if majority (≥3 of 5)
")
  echo "XGB channel wins: $XGB_WIN"
fi

# Step 4: SPEC 5 (conditional)
if [ "$XGB_WIN" = "1" ]; then
  echo "[$(date +%H:%M:%S)] xgb wins → submitting SPEC 5"
  SPEC5=$(sbatch --parsable --export=ALL,BUNDLE_PT=$BUNDLE_PT,SPLIT_ROOT=$SPLIT_ROOT \
    scripts/v9_ml/run_spec05_v9.sh)
  echo "  SPEC 5 job: $SPEC5"
  DEP_ARG="--dependency=afterany:$SPEC5"
else
  echo "[$(date +%H:%M:%S)] xgb NOT best → skipping SPEC 5"
  DEP_ARG=""
fi

# Step 5: SPEC 4 + SPEC 2
echo "[$(date +%H:%M:%S)] submitting SPEC 4 + SPEC 2"
SPEC4=$(sbatch --parsable $DEP_ARG --export=ALL,BUNDLE_PT=$BUNDLE_PT,SPLIT_ROOT=$SPLIT_ROOT \
  scripts/v9_ml/run_spec04_v9.sh)
echo "  SPEC 4 job: $SPEC4"
SPEC2=$(sbatch --parsable $DEP_ARG --export=ALL,BUNDLE_PT=$BUNDLE_PT,SPLIT_ROOT=$SPLIT_ROOT \
  scripts/v9_ml/run_spec02_v9.sh)
echo "  SPEC 2 job: $SPEC2"
echo "[$(date +%H:%M:%S)] chain done"
