#!/bin/bash
# Final cleanup: 8 not-run EDA + 2 charge-fix EDA + 2 fragA_R strain SP
# Single shard, serial. Regen inp for charge-fix, then run.

#SBATCH --job-name=orca_finalfx
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_finalfx.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_finalfx.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
if [ -f "$HOME/orca6/orca-env.sh" ]; then source "$HOME/orca6/orca-env.sh"; fi

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
INPUT_ROOT="$REPO/outputs/v8_review/orca_inputs"
SP_ROOT="$REPO/outputs/v8_review/strain_sp"
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"

cd "$REPO"
python -u scripts/v8/fix_final_leftovers.py

# ---- Run EDA (10 rxns) ----
echo "===== EDA rerun ====="
n_ok=0; n_fail=0
while IFS= read -r RID; do
  DIR="$INPUT_ROOT/$RID"
  INP="$DIR/eda.inp"; OUT="$DIR/eda.out"
  if [ ! -f "$INP" ]; then echo "[SKIP] $RID: no inp"; continue; fi
  if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[SKIP] $RID: done"; continue
  fi
  echo "[$(date +%H:%M:%S)] EDA START $RID"
  cd "$DIR" || continue
  "$ORCA_BIN" eda.inp > eda.out 2> eda.err
  if grep -q "ORCA TERMINATED NORMALLY" eda.out 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] EDA OK    $RID"; n_ok=$((n_ok+1))
    rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
  else
    echo "[$(date +%H:%M:%S)] EDA FAIL  $RID"; n_fail=$((n_fail+1))
  fi
done < "$INPUT_ROOT/manifest_final_eda.txt"
echo "EDA total: ok=$n_ok fail=$n_fail"

# ---- Run strain SP fragA for the 2 charge-fix rxns ----
echo "===== SP rerun (fragA only) ====="
n_sp_ok=0; n_sp_fail=0
while IFS= read -r RID; do
  DIR="$SP_ROOT/$RID"
  INP="$DIR/fragA_R.inp"; OUT="$DIR/fragA_R.out"
  if [ ! -f "$INP" ]; then echo "[SKIP] $RID fragA: no inp"; continue; fi
  if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
    echo "[SKIP] $RID fragA: done"; continue
  fi
  echo "[$(date +%H:%M:%S)] SP START $RID fragA"
  cd "$DIR" || continue
  "$ORCA_BIN" fragA_R.inp > fragA_R.out 2> fragA_R.err
  if grep -q "ORCA TERMINATED NORMALLY" fragA_R.out 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SP OK    $RID fragA"; n_sp_ok=$((n_sp_ok+1))
    rm -f fragA_R.densities fragA_R.gbw fragA_R.bas* fragA_R.tmp fragA_R.smpso fragA_R.smpss 2>/dev/null
  else
    echo "[$(date +%H:%M:%S)] SP FAIL  $RID fragA"; n_sp_fail=$((n_sp_fail+1))
  fi
done < "$SP_ROOT/manifest_final_sp.txt"
echo "SP total: ok=$n_sp_ok fail=$n_sp_fail"

echo "[$(date +%H:%M:%S)] final leftovers done"
