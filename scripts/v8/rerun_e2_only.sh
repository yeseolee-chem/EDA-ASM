#!/bin/bash
# Rerun ORCA EDA only for the failed qmrxn20_e2 rxns (charge -1 fixed inp).
#
# Manifest: outputs/v8_review/orca_inputs/manifest_e2_rerun.txt
# Idempotent: skips rxns whose eda.out already contains "ORCA TERMINATED NORMALLY".

#SBATCH --job-name=orca_eda_e2
#SBATCH --partition=cpu2
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_eda_e2_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_eda_e2_%A_%a.err

set -uo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
INPUT_ROOT="$REPO/outputs/v8_review/orca_inputs"
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$INPUT_ROOT/manifest_e2_rerun.txt"

TOTAL=$(wc -l < "$MANIFEST")
NSHARDS=9
SHARD=$SLURM_ARRAY_TASK_ID
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS  total=$TOTAL  node=$(hostname -s)"

if [ -f "$HOME/orca6/orca-env.sh" ]; then
  source "$HOME/orca6/orca-env.sh"
fi

LINE=0
n_ok=0; n_skip=0; n_fail=0
while IFS= read -r RID; do
  LINE=$((LINE + 1))
  if (( (LINE - 1) % NSHARDS != SHARD )); then continue; fi

  DIR="$INPUT_ROOT/$RID"
  INP="$DIR/eda.inp"
  OUT="$DIR/eda.out"

  if [ ! -f "$INP" ]; then
    echo "[SKIP] $RID: no eda.inp"; n_skip=$((n_skip+1)); continue
  fi
  if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[SKIP] $RID: already done"; n_skip=$((n_skip+1)); continue
  fi

  echo "[$(date +%H:%M:%S)] START $RID (shard $SHARD)"
  cd "$DIR" || continue
  "$ORCA_BIN" eda.inp > eda.out 2> eda.err
  if grep -q "ORCA TERMINATED NORMALLY" eda.out 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK    $RID"
    n_ok=$((n_ok+1))
    rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
  else
    echo "[$(date +%H:%M:%S)] FAIL  $RID"
    n_fail=$((n_fail+1))
  fi
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] shard $SHARD done: ok=$n_ok skip=$n_skip fail=$n_fail"
