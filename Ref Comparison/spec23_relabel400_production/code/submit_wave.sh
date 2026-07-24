#!/bin/bash
#SBATCH --job-name=s23_sweep
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_sweep_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_sweep_%j.err

# spec23 wave submitter: scans job_manifest.csv, classifies PENDING vs DONE,
# writes a wave manifest, submits a SLURM array over PENDING, then submits
# itself as an afterany dependency for the next wave. Cap at $WAVE_CAP.
#
# CLAUDE.md: 12-task queue cap → each array ≤ 11 tasks. This script runs
# as a compute-node sbatch job (option 2 in CLAUDE.md), NOT a login-node daemon.

set -uo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec23_relabel400_production"
JOB_MANIFEST="$STAGE/results/job_manifest.csv"
WAVE_LOG="$STAGE/logs/wave_history.csv"
WAVE_DIR="$STAGE/logs/waves"
GATES_LOG="$STAGE/logs/gates.log"

mkdir -p "$WAVE_DIR"

WAVE_N="${WAVE_N:-1}"
WAVE_CAP="${WAVE_CAP:-6}"
ARRAY_SIZE="${ARRAY_SIZE:-10}"   # ≤ 11 to stay under 12-cap (this sweep uses the 11th slot)
NPROCS="${NPROCS:-4}"
MAXCORE_MB="${MAXCORE_MB:-3500}"

echo "=== [$(date -Is)] wave $WAVE_N / cap $WAVE_CAP ==="

# --- Classify --------------------------------------------------------------
python3 - <<PY > "$WAVE_DIR/pending_wave_${WAVE_N}.csv"
import csv, os, sys
manifest = "$JOB_MANIFEST"
with open(manifest) as f:
    rows = list(csv.DictReader(f))
pending = []
for r in rows:
    out = r["out"]
    done = False
    try:
        with open(out) as g:
            for line in g:
                if "****ORCA TERMINATED NORMALLY****" in line:
                    done = True; break
    except FileNotFoundError:
        pass
    if not done:
        pending.append(r)
w = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
w.writeheader()
for r in pending:
    w.writerow(r)
PY

NPENDING=$(($(wc -l < "$WAVE_DIR/pending_wave_${WAVE_N}.csv") - 1))
echo "[wave $WAVE_N] pending = $NPENDING"

# Update wave history
if [[ ! -f "$WAVE_LOG" ]]; then
  echo "wave,pending,submitted_at" > "$WAVE_LOG"
fi
echo "$WAVE_N,$NPENDING,$(date -Is)" >> "$WAVE_LOG"

# --- Termination checks ----------------------------------------------------
if [[ "$NPENDING" -le 0 ]]; then
  echo "[wave $WAVE_N] all jobs complete — signalling done"
  echo "ALL_DONE at $(date -Is)" >> "$GATES_LOG"
  exit 0
fi
if [[ "$WAVE_N" -ge "$WAVE_CAP" ]]; then
  echo "[wave $WAVE_N] hit wave cap $WAVE_CAP with $NPENDING still pending — halting chain"
  echo "WAVE_CAP_HIT wave=$WAVE_N pending=$NPENDING at $(date -Is)" >> "$GATES_LOG"
  exit 0
fi

# --- Convergence check: pending must have decreased from the prior wave ----
if [[ "$WAVE_N" -gt 1 ]]; then
  PREV=$(awk -F',' -v w=$((WAVE_N - 1)) 'NR>1 && $1==w {print $2}' "$WAVE_LOG" | tail -1)
  if [[ -n "$PREV" ]] && [[ "$NPENDING" -ge "$PREV" ]]; then
    echo "[wave $WAVE_N] pending did not decrease ($PREV -> $NPENDING) — halting chain"
    echo "STUCK wave=$WAVE_N prev=$PREV curr=$NPENDING at $(date -Is)" >> "$GATES_LOG"
    exit 0
  fi
fi

# --- Submit array over pending (size ≤ ARRAY_SIZE) -------------------------
BATCH=$((NPENDING < ARRAY_SIZE ? NPENDING : ARRAY_SIZE))
WAVE_MANIFEST="$WAVE_DIR/wave_${WAVE_N}_manifest.csv"
head -1 "$JOB_MANIFEST" > "$WAVE_MANIFEST"
tail -n +2 "$WAVE_DIR/pending_wave_${WAVE_N}.csv" | head -n "$BATCH" >> "$WAVE_MANIFEST"

echo "[wave $WAVE_N] submitting array size $BATCH from $WAVE_MANIFEST"

RUN_SCRIPT="$STAGE/code/run_job.sh"
JOB_ID=$(sbatch --parsable \
  --export=ALL,MANIFEST="$WAVE_MANIFEST",NPROCS="$NPROCS",MAXCORE_MB="$MAXCORE_MB" \
  --array=0-$((BATCH - 1))%${ARRAY_SIZE} \
  "$RUN_SCRIPT")

echo "[wave $WAVE_N] array job id = $JOB_ID"

# --- Chain next wave as afterany dependency --------------------------------
NEXT_WAVE=$((WAVE_N + 1))
if [[ "$NEXT_WAVE" -le "$WAVE_CAP" ]]; then
  sbatch --parsable \
    --dependency=afterany:${JOB_ID} \
    --export=ALL,WAVE_N=$NEXT_WAVE,WAVE_CAP=$WAVE_CAP,ARRAY_SIZE=$ARRAY_SIZE,NPROCS=$NPROCS,MAXCORE_MB=$MAXCORE_MB \
    "$STAGE/code/submit_wave.sh"
  echo "[wave $WAVE_N] next-wave sweep queued"
fi

echo "=== [$(date -Is)] wave $WAVE_N submission complete ==="
