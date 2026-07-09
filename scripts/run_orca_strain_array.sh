#!/bin/bash
# 9-shard array runner for strain-pass optimizations.
# Each task: single ORCA Opt on one fragment (fA or fB).
# Idempotent: skips if opt.out already contains "ORCA TERMINATED NORMALLY".
# Serial ORCA (no MPI on this cluster). Auto-cleans intermediate files.

#SBATCH --job-name=orca_strain
#SBATCH --partition=cpu2
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_strain_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_strain_%A_%a.err

set -uo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
INPUT_ROOT="$REPO/outputs/orca_strain/inputs"
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$REPO/outputs/orca_strain/manifest.txt"

TOTAL=$(wc -l < "$MANIFEST")
NSHARDS=9
SHARD=$SLURM_ARRAY_TASK_ID
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS  total=$TOTAL  node=$(hostname -s)"

if [ -f "$HOME/orca6/orca-env.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/orca6/orca-env.sh"
fi

LINE=0
while IFS= read -r ITEM; do
  LINE=$((LINE + 1))
  if (( (LINE - 1) % NSHARDS != SHARD )); then continue; fi

  DIR="$INPUT_ROOT/$ITEM"
  INP="$DIR/opt.inp"
  OUT="$DIR/opt.out"

  if [ ! -f "$INP" ]; then
    echo "[SKIP] $ITEM: no opt.inp"; continue
  fi
  if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT"; then
    echo "[SKIP] $ITEM: already done"; continue
  fi

  echo "[$(date +%H:%M:%S)] START $ITEM (shard $SHARD)"
  cd "$DIR"
  "$ORCA_BIN" opt.inp > opt.out 2> opt.err
  STATUS=$?
  if grep -q "ORCA TERMINATED NORMALLY" opt.out 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] OK    $ITEM"
    # Cleanup intermediate files, keep only opt.out, opt.inp, opt.property.txt
    rm -f opt.densities opt.densitiesinfo opt.gbw opt.bas* opt.tmp opt.int.tmp
    rm -f opt.bibtex opt.engrad opt.opt opt_trj.xyz opt_property.txt.tmp
  else
    echo "[$(date +%H:%M:%S)] FAIL  $ITEM  (exit=$STATUS)"
  fi
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] shard $SHARD finished"
