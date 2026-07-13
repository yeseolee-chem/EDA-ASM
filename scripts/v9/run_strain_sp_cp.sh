#!/bin/bash
# Counterpoise-corrected fragment SP runner (v9).
# One work item per SLURM array task, format: "SP <rid> <fragA|fragB>".
# Reads manifest from outputs/v9_review/manifest_sp.txt.
# Each SP is small (~5-30 min), so ARRAY_MAX=8 keeps us under the 10-slot cap.

#SBATCH --job-name=v9_cpsp
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review/logs/cpsp_%A_%a.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review/logs/cpsp_%A_%a.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
V9="$REPO/outputs/v9_review"
MANIFEST="${MANIFEST:-$V9/manifest_sp.txt}"

TASK=$SLURM_ARRAY_TASK_ID
# Multiple items per task = pack density. If PACK is set, this task
# processes items [TASK*PACK .. TASK*PACK+PACK-1]. Default PACK=1.
PACK=${PACK:-1}
START=$((TASK * PACK))
END=$((START + PACK))

TOTAL_LINES=$(wc -l < "$MANIFEST")
if [ "$START" -ge "$TOTAL_LINES" ]; then
  echo "task $TASK: no work (start=$START, total=$TOTAL_LINES)"; exit 0
fi

for i in $(seq $START $((END-1))); do
  [ "$i" -ge "$TOTAL_LINES" ] && break
  LINE=$(awk "NR==$((i+1))" "$MANIFEST")
  [ -z "$LINE" ] && continue
  TYPE=$(echo "$LINE" | awk '{print $1}')
  RID=$(echo "$LINE" | awk '{print $2}')
  FRAG=$(echo "$LINE" | awk '{print $3}')
  DIR="$V9/strain_sp_cp/$RID"
  INP="${FRAG}_R.inp"
  OUT="${FRAG}_R.out"

  if [ ! -f "$DIR/$INP" ]; then
    echo "[$(date +%H:%M:%S)] MISS $RID $FRAG (no inp)"; continue
  fi
  if [ -f "$DIR/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$DIR/$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $RID $FRAG (already done)"; continue
  fi

  echo "[$(date +%H:%M:%S)] START $RID $FRAG on $(hostname -s)"
  cd "$DIR"
  "$ORCA_BIN" "$INP" > "$OUT" 2> "${FRAG}_R.err"
  if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK   $RID $FRAG"
    rm -f "${FRAG}_R.densities" "${FRAG}_R.gbw" "${FRAG}_R.bas"* "${FRAG}_R.tmp" 2>/dev/null
  else
    echo "[$(date +%H:%M:%S)] FAIL $RID $FRAG (see ${FRAG}_R.err)"
  fi
done

echo "[$(date +%H:%M:%S)] task $TASK done"
