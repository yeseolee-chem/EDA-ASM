#!/bin/bash
# Serial runner for OOD (30 tasks) — single sbatch submission.
# Uses manifest_ood_parallel.txt (10 EDA + 20 SP entries).

#SBATCH --job-name=ood_ser
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/ood_ser.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/ood_ser.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$REPO/outputs/v8_review/manifest_ood_parallel.txt"
TOTAL=$(wc -l < "$MANIFEST")
echo "[$(date +%H:%M:%S)] ood_serial total=$TOTAL node=$(hostname -s)"

n_ok=0; n_fail=0
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  TYPE=$(echo "$LINE" | awk '{print $1}')
  RID=$(echo "$LINE" | awk '{print $2}')
  FRAG=$(echo "$LINE" | awk '{print $3}')

  if [ "$TYPE" = "EDA" ]; then
    DIR="$REPO/outputs/v8_review/orca_inputs/$RID"
    INP=eda.inp; OUT=eda.out
  elif [ "$TYPE" = "SP" ]; then
    DIR="$REPO/outputs/v8_review/strain_sp/$RID"
    INP="${FRAG}_R.inp"; OUT="${FRAG}_R.out"
  else
    echo "[SKIP] unknown $LINE"; continue
  fi

  if [ ! -f "$DIR/$INP" ]; then echo "[SKIP] $RID $FRAG: no inp"; continue; fi
  if [ -f "$DIR/$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$DIR/$OUT" 2>/dev/null; then
    echo "[SKIP] $RID $FRAG: done"; continue
  fi
  echo "[$(date +%H:%M:%S)] START $TYPE $RID $FRAG"
  cd "$DIR"
  "$ORCA_BIN" "$INP" > "$OUT" 2> "${INP%.inp}.err"
  if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK   $TYPE $RID $FRAG"; n_ok=$((n_ok+1))
    if [ "$TYPE" = "EDA" ]; then
      rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
    else
      rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp 2>/dev/null
    fi
  else
    echo "[$(date +%H:%M:%S)] FAIL $TYPE $RID $FRAG"; n_fail=$((n_fail+1))
  fi
done < "$MANIFEST"
echo "[$(date +%H:%M:%S)] done: ok=$n_ok fail=$n_fail"
