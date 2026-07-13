#!/bin/bash
# Helper runner — identical to run_strain_sp_cp.sh but writes into
# strain_sp_helper/ instead of strain_sp_cp/. Purpose: race-free
# parallelism with the primary array 732264 (same inp files copied
# to a separate dir, no shared writer).
#
# Manifest format: "SP <rid> <fragA|fragB>", one per line.
# Assembler picks the primary strain_sp_cp/ first, then falls back
# to strain_sp_helper/ if primary .out is missing.

#SBATCH --job-name=v9_help
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review/logs/help_%A_%a.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review/logs/help_%A_%a.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
V9="$REPO/outputs/v9_review"
MANIFEST="${MANIFEST:-$V9/manifest_helper.txt}"

TASK=$SLURM_ARRAY_TASK_ID
PACK=${PACK:-1}
START=$((TASK * PACK))
END=$((START + PACK))

TOTAL_LINES=$(wc -l < "$MANIFEST")
if [ "$START" -ge "$TOTAL_LINES" ]; then
  echo "task $TASK: no work"; exit 0
fi

for i in $(seq $START $((END-1))); do
  [ "$i" -ge "$TOTAL_LINES" ] && break
  LINE=$(awk "NR==$((i+1))" "$MANIFEST")
  [ -z "$LINE" ] && continue
  RID=$(echo "$LINE" | awk '{print $2}')
  FRAG=$(echo "$LINE" | awk '{print $3}')
  DIR="$V9/strain_sp_helper/$RID"
  INP="${FRAG}_R.inp"
  OUT="${FRAG}_R.out"

  # Skip if EITHER dir has a completed SP (primary or helper) - safe deduplication
  if [ -f "$V9/strain_sp_cp/$RID/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$V9/strain_sp_cp/$RID/$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $RID $FRAG (primary already done)"; continue
  fi
  if [ -f "$DIR/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$DIR/$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $RID $FRAG (helper already done)"; continue
  fi
  if [ ! -f "$DIR/$INP" ]; then
    echo "[$(date +%H:%M:%S)] MISS $RID $FRAG (no inp in helper dir)"; continue
  fi

  echo "[$(date +%H:%M:%S)] START $RID $FRAG on $(hostname -s)"
  cd "$DIR"
  "$ORCA_BIN" "$INP" > "$OUT" 2> "${FRAG}_R.err"
  if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK   $RID $FRAG"
    rm -f "${FRAG}_R.densities" "${FRAG}_R.gbw" "${FRAG}_R.bas"* "${FRAG}_R.tmp" 2>/dev/null
  else
    echo "[$(date +%H:%M:%S)] FAIL $RID $FRAG"
  fi
done

echo "[$(date +%H:%M:%S)] helper task $TASK done"
