#!/bin/bash
# Sharded ORCA EDA-NOCV runner: 9 workers, each processes its shard of 789
# reactions. Each reaction is a single ORCA EDA-NOCV run producing the
# 5-6 channel decomposition (Pauli, V_elst, E_orb, ΔE_XC, E_disp, E_int).
#
# Idempotent: skips reactions whose eda.out contains "ORCA TERMINATED NORMALLY".

#SBATCH --job-name=orca_eda_789
#SBATCH --partition=cpu2
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_eda_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_eda_%A_%a.err

set -uo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
INPUT_ROOT="$REPO/outputs/orca_eda/inputs"
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"

MANIFEST="$REPO/outputs/orca_eda/manifest.txt"
if [ ! -s "$MANIFEST" ]; then
  ls -1 "$INPUT_ROOT" | sort > "$MANIFEST"
fi

TOTAL=$(wc -l < "$MANIFEST")
NSHARDS=9
SHARD=$SLURM_ARRAY_TASK_ID
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS  total=$TOTAL  node=$(hostname -s)"

if [ -f "$HOME/orca6/orca-env.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/orca6/orca-env.sh"
fi

LINE=0
while IFS= read -r RID; do
  LINE=$((LINE + 1))
  if (( (LINE - 1) % NSHARDS != SHARD )); then continue; fi

  DIR="$INPUT_ROOT/$RID"
  INP="$DIR/eda.inp"
  OUT="$DIR/eda.out"

  if [ ! -f "$INP" ]; then
    echo "[SKIP] $RID: no eda.inp"; continue
  fi
  if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT"; then
    echo "[SKIP] $RID: already done"; continue
  fi

  echo "[$(date +%H:%M:%S)] START $RID (shard $SHARD)"
  cd "$DIR"
  "$ORCA_BIN" eda.inp > eda.out 2> eda.err
  STATUS=$?
  if grep -q "ORCA TERMINATED NORMALLY" eda.out 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK    $RID"
    # Cleanup: remove wavefunction/density files (kept only .out, .property.txt, .inp).
    # These are unused downstream (EDA channels parsed from eda.out only).
    rm -f eda.densities eda_frag1.densities eda_frag2.densities
    rm -f eda.gbw eda.nocv.gbw eda_frag1.gbw eda_frag2.gbw
    rm -f eda.bas0 eda.bas1 eda.bas2 eda.bas3 eda.bas4 eda.bas5
    rm -f eda_frag1.bas0 eda_frag1.bas1 eda_frag2.bas0 eda_frag2.bas1
    rm -f eda.densitiesinfo eda.bibtex eda_frag1.bibtex eda_frag2.bibtex
    rm -f eda.int.tmp eda.tmp
  else
    echo "[$(date +%H:%M:%S)] FAIL  $RID  (exit=$STATUS)"
  fi
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] shard $SHARD finished"
