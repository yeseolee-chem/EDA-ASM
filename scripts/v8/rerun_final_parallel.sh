#!/bin/bash
# Parallel runner: each array task processes ONE work item from manifest_final_parallel.txt
# Format per line: either "EDA <rid>" or "SP <rid> <frag>"
# 12 tasks (or however many entries), max 8 concurrent to stay under 10-slot cap.

#SBATCH --job-name=orca_finpar
#SBATCH --partition=cpu2
#SBATCH --array=0-10%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_finpar_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_finpar_%A_%a.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$REPO/outputs/v8_review/manifest_final_parallel.txt"

TASK=$SLURM_ARRAY_TASK_ID
LINE=$(awk "NR==$((TASK+1))" "$MANIFEST")
if [ -z "$LINE" ]; then
  echo "no work for task $TASK"; exit 0
fi

TYPE=$(echo "$LINE" | awk '{print $1}')
RID=$(echo "$LINE" | awk '{print $2}')
FRAG=$(echo "$LINE" | awk '{print $3}')

echo "[$(date +%H:%M:%S)] task=$TASK type=$TYPE rid=$RID frag=$FRAG node=$(hostname -s)"

if [ "$TYPE" = "EDA" ]; then
  DIR="$REPO/outputs/v8_review/orca_inputs/$RID"
  INP=eda.inp; OUT=eda.out
elif [ "$TYPE" = "SP" ]; then
  DIR="$REPO/outputs/v8_review/strain_sp/$RID"
  INP="${FRAG}_R.inp"; OUT="${FRAG}_R.out"
else
  echo "unknown TYPE: $TYPE"; exit 1
fi

if [ ! -f "$DIR/$INP" ]; then
  echo "no inp: $DIR/$INP"; exit 1
fi
if [ -f "$DIR/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$DIR/$OUT" 2>/dev/null; then
  echo "already done: $DIR/$OUT"; exit 0
fi

cd "$DIR"
"$ORCA_BIN" "$INP" > "$OUT" 2> "${INP%.inp}.err"
if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] OK   $TYPE $RID $FRAG"
  # cleanup intermediates
  if [ "$TYPE" = "EDA" ]; then
    rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
  else
    rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp 2>/dev/null
  fi
else
  echo "[$(date +%H:%M:%S)] FAIL $TYPE $RID $FRAG"
fi
