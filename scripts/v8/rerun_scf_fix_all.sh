#!/bin/bash
# Auto-detect currently-failed SCF-related dipolar rxns, regen inp with
# SlowConv + SOSCF + MaxIter 500, and rerun. Idempotent, serial.
# Runs AFTER 730612 finishes so no race conditions.

#SBATCH --job-name=orca_scfall
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_scfall.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_scfall.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
if [ -f "$HOME/orca6/orca-env.sh" ]; then source "$HOME/orca6/orca-env.sh"; fi

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
INPUT_ROOT="$REPO/outputs/v8_review/orca_inputs"
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$INPUT_ROOT/manifest_scf_fix.txt"

cd "$REPO"
python -u scripts/v8/fix_all_scf_dipolar.py

TOTAL=$(wc -l < "$MANIFEST")
echo "[$(date +%H:%M:%S)] scf-fix-all  total=$TOTAL  node=$(hostname -s)"
n_ok=0; n_fail=0
while IFS= read -r RID; do
  DIR="$INPUT_ROOT/$RID"
  INP="$DIR/eda.inp"; OUT="$DIR/eda.out"
  if [ ! -f "$INP" ]; then echo "[SKIP] $RID: no inp"; continue; fi
  if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[SKIP] $RID: done"; continue
  fi
  echo "[$(date +%H:%M:%S)] START $RID"
  cd "$DIR" || continue
  "$ORCA_BIN" eda.inp > eda.out 2> eda.err
  if grep -q "ORCA TERMINATED NORMALLY" eda.out 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK    $RID"; n_ok=$((n_ok+1))
    rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
  else
    echo "[$(date +%H:%M:%S)] FAIL  $RID"; n_fail=$((n_fail+1))
  fi
done < "$MANIFEST"
echo "[$(date +%H:%M:%S)] scf-fix-all done: ok=$n_ok fail=$n_fail"
