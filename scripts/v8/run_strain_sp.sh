#!/bin/bash
# Runs ORCA SP for E_A(R) and E_B(R) — the strain-channel fragment SPs.
# Idempotent: skips if fragA_R.out and fragB_R.out both contain "ORCA TERMINATED NORMALLY".
#
# Shard 0 regenerates inp files + manifest at start (picks up any late R-review changes).

#SBATCH --job-name=strain_sp
#SBATCH --partition=cpu2
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/strain_sp_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/strain_sp_%A_%a.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
SP_ROOT="$REPO/outputs/v8_review/strain_sp"
ORCA_BIN="$HOME/orca_6_1_1_avx2/orca"
MANIFEST="$SP_ROOT/manifest.txt"

# Shard 0 regenerates inp + manifest; others wait briefly
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  cd "$REPO"
  python -u scripts/v8/gen_strain_sp_inp.py
fi
if [ ! -s "$MANIFEST" ]; then
  for i in 1 2 3 4 5 6; do sleep 10; [ -s "$MANIFEST" ] && break; done
fi

if [ -f "$HOME/orca6/orca-env.sh" ]; then
  source "$HOME/orca6/orca-env.sh"
fi

NSHARDS=9
SHARD=$SLURM_ARRAY_TASK_ID
TOTAL=$(wc -l < "$MANIFEST")
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS  total=$TOTAL  node=$(hostname -s)"

LINE=0
n_ok=0; n_skip=0; n_fail=0
while IFS= read -r RID; do
  LINE=$((LINE + 1))
  if (( (LINE - 1) % NSHARDS != SHARD )); then continue; fi

  DIR="$SP_ROOT/$RID"
  cd "$DIR" || continue

  for FRAG in fragA fragB; do
    INP="${FRAG}_R.inp"
    OUT="${FRAG}_R.out"
    if [ ! -f "$INP" ]; then
      echo "[SKIP] $RID $FRAG: no inp"; n_skip=$((n_skip+1)); continue
    fi
    if [ -f "$OUT" ] && grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
      n_skip=$((n_skip+1)); continue
    fi
    "$ORCA_BIN" "$INP" > "$OUT" 2> "${FRAG}_R.err"
    if grep -q "ORCA TERMINATED NORMALLY" "$OUT" 2>/dev/null; then
      n_ok=$((n_ok+1))
      rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp ${FRAG}_R.smpso ${FRAG}_R.smpss ${FRAG}_R.opt ${FRAG}_R.hess ${FRAG}_R.engrad 2>/dev/null
    else
      echo "[FAIL] $RID $FRAG"
      n_fail=$((n_fail+1))
    fi
  done
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] shard $SHARD done: ok=$n_ok skip=$n_skip fail=$n_fail"
