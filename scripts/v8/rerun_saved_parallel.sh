#!/bin/bash
# 8-way parallel array for the combined saved-rxns manifest.
# Manifest: manifest_saved_all.txt (OOD 30 + non-OOD 54 = 84 tasks)
# Each shard: processes items where (LINE-1) % 8 == SHARD.
# Idempotent: skips items already TERMINATED NORMALLY.

#SBATCH --job-name=saved_par
#SBATCH --partition=cpu2
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/saved_par_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/saved_par_%A_%a.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$REPO/outputs/v8_review/manifest_saved_all.txt"

NSHARDS=8
SHARD=$SLURM_ARRAY_TASK_ID
TOTAL=$(wc -l < "$MANIFEST")
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS  total=$TOTAL  node=$(hostname -s)"

LINE=0; n_ok=0; n_skip=0; n_fail=0
while IFS= read -r LINE_TXT; do
  LINE=$((LINE + 1))
  [ -z "$LINE_TXT" ] && continue
  if (( (LINE - 1) % NSHARDS != SHARD )); then continue; fi

  TYPE=$(echo "$LINE_TXT" | awk '{print $1}')
  RID=$(echo "$LINE_TXT" | awk '{print $2}')
  FRAG=$(echo "$LINE_TXT" | awk '{print $3}')
  if [ "$TYPE" = "EDA" ]; then
    DIR="$REPO/outputs/v8_review/orca_inputs/$RID"; INP=eda.inp; OUT=eda.out
  elif [ "$TYPE" = "SP" ]; then
    DIR="$REPO/outputs/v8_review/strain_sp/$RID"; INP="${FRAG}_R.inp"; OUT="${FRAG}_R.out"
  else
    continue
  fi
  if [ ! -f "$DIR/$INP" ]; then n_skip=$((n_skip+1)); continue; fi
  if [ -f "$DIR/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$DIR/$OUT" 2>/dev/null; then
    n_skip=$((n_skip+1)); continue
  fi
  echo "[$(date +%H:%M:%S)] shard $SHARD START $TYPE $RID $FRAG"
  cd "$DIR"
  "$ORCA_BIN" "$INP" > "$OUT" 2> "${INP%.inp}.err"
  if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] shard $SHARD OK   $TYPE $RID $FRAG"; n_ok=$((n_ok+1))
    if [ "$TYPE" = "EDA" ]; then
      rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
    else
      rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp 2>/dev/null
    fi
  else
    echo "[$(date +%H:%M:%S)] shard $SHARD FAIL $TYPE $RID $FRAG"; n_fail=$((n_fail+1))
  fi
done < "$MANIFEST"
echo "[$(date +%H:%M:%S)] shard $SHARD done: ok=$n_ok skip=$n_skip fail=$n_fail"
