#!/bin/bash
#SBATCH --job-name=orca_retry
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_retry.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_retry.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN=$HOME/orca_6_1_1_avx2/orca

# EDA rerun
if [ -s "$REPO/outputs/v8_review/orca_inputs/manifest_retry.txt" ]; then
  while IFS= read -r RID; do
    [ -z "$RID" ] && continue
    DIR="$REPO/outputs/v8_review/orca_inputs/$RID"
    [ -f "$DIR/eda.inp" ] || continue
    echo "[$(date +%H:%M:%S)] EDA START $RID"
    (cd "$DIR" && "$ORCA_BIN" eda.inp > eda.out 2> eda.err)
    if grep -q "ORCA TERMINATED NORMALLY" "$DIR/eda.out" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] EDA OK    $RID"
      (cd "$DIR" && rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null)
    else
      echo "[$(date +%H:%M:%S)] EDA FAIL  $RID"
    fi
  done < "$REPO/outputs/v8_review/orca_inputs/manifest_retry.txt"
fi

# SP rerun
if [ -s "$REPO/outputs/v8_review/strain_sp/manifest_retry.txt" ]; then
  while IFS= read -r LINE; do
    [ -z "$LINE" ] && continue
    RID=$(echo "$LINE" | awk '{print $1}')
    FRAG=$(echo "$LINE" | awk '{print $2}')
    DIR="$REPO/outputs/v8_review/strain_sp/$RID"
    [ -f "$DIR/${FRAG}_R.inp" ] || continue
    echo "[$(date +%H:%M:%S)] SP START $RID $FRAG"
    (cd "$DIR" && "$ORCA_BIN" "${FRAG}_R.inp" > "${FRAG}_R.out" 2> "${FRAG}_R.err")
    if grep -q "ORCA TERMINATED NORMALLY" "$DIR/${FRAG}_R.out" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] SP OK    $RID $FRAG"
      (cd "$DIR" && rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp 2>/dev/null)
    else
      echo "[$(date +%H:%M:%S)] SP FAIL  $RID $FRAG"
    fi
  done < "$REPO/outputs/v8_review/strain_sp/manifest_retry.txt"
fi
echo "[$(date +%H:%M:%S)] retry done"
