#!/bin/bash
# 4-way helper for the 8 tasks currently stuck on shard 7 of 731204.
# Manifest: manifest_shard7_pending.txt

#SBATCH --job-name=sh7_help
#SBATCH --partition=cpu2
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/sh7_help_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/sh7_help_%A_%a.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$REPO/outputs/v8_review/manifest_shard7_pending.txt"
NSHARDS=4
SHARD=$SLURM_ARRAY_TASK_ID
TOTAL=$(wc -l < "$MANIFEST")
echo "[$(date +%H:%M:%S)] helper shard $SHARD/$NSHARDS total=$TOTAL"

LINE=0
while IFS= read -r LINE_TXT; do
  LINE=$((LINE+1))
  [ -z "$LINE_TXT" ] && continue
  if (( (LINE - 1) % NSHARDS != SHARD )); then continue; fi
  TYPE=$(echo "$LINE_TXT" | awk '{print $1}')
  RID=$(echo "$LINE_TXT" | awk '{print $2}')
  FRAG=$(echo "$LINE_TXT" | awk '{print $3}')
  if [ "$TYPE" = "EDA" ]; then
    DIR="$REPO/outputs/v8_review/orca_inputs/$RID"; INP=eda.inp; OUT=eda.out
  elif [ "$TYPE" = "SP" ]; then
    DIR="$REPO/outputs/v8_review/strain_sp/$RID"; INP="${FRAG}_R.inp"; OUT="${FRAG}_R.out"
  else continue; fi
  if [ ! -f "$DIR/$INP" ]; then continue; fi
  if [ -f "$DIR/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$DIR/$OUT" 2>/dev/null; then continue; fi
  echo "[$(date +%H:%M:%S)] helper $SHARD START $TYPE $RID $FRAG"
  cd "$DIR"
  "$ORCA_BIN" "$INP" > "$OUT" 2> "${INP%.inp}.err"
  if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] helper $SHARD OK   $TYPE $RID $FRAG"
    if [ "$TYPE" = "EDA" ]; then
      rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
    else
      rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp 2>/dev/null
    fi
  else
    echo "[$(date +%H:%M:%S)] helper $SHARD FAIL $TYPE $RID $FRAG"
  fi
done < "$MANIFEST"
echo "[$(date +%H:%M:%S)] helper $SHARD done"
