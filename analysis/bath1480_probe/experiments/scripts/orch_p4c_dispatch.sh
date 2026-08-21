#!/bin/bash
# Continuous dispatcher: submits one P4c cell at a time whenever queue has room.
# Keeps up to MAX_QUEUE tasks in queue (default 10). Idempotent — skips cells
# whose metrics.json already exists.
#
# Runs on compute node (CLAUDE.md compliant). Launches per-cell 1-task arrays
# so each cell is an independent sbatch submission.
#
#SBATCH --job-name=orch_p4c_disp
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/orch_p4c_disp.%j.log

set -o pipefail

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
RESULTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results
OUT_ROOT=$RESULTS/phase4_stacked_xgb_mace

MAX_QUEUE=${MAX_QUEUE:-10}   # keep queue at this size (leaves 5 slots for other user jobs at 15-cap)
POLL_SEC=${POLL_SEC:-30}

echo "=== orch_p4c dispatch start $(date -Is)  max_queue=$MAX_QUEUE ==="

# Build list of cells to run: skip those with metrics.json
declare -a TODO
for CELL in $(seq 0 24); do
    FOLD=$((CELL / 5))
    SEED=$((CELL % 5))
    if [ -f "$OUT_ROOT/fold_$FOLD/seed_$SEED/metrics.json" ]; then
        echo "SKIP cell $CELL (fold=$FOLD seed=$SEED) — metrics.json exists"
        continue
    fi
    TODO+=($CELL)
done
echo "=== ${#TODO[@]} cells to process: ${TODO[@]} ==="

# Track submitted job IDs so we can count "our" queued P4c jobs (exclude self + others)
declare -a SUBMITTED

count_our_p4c_in_queue() {
    local n=0
    if [ -n "${SUBMITTED+x}" ] && [ ${#SUBMITTED[@]} -gt 0 ]; then
        local ids
        ids=$(IFS=,; echo "${SUBMITTED[*]}")
        n=$(squeue -j "$ids" -h 2>/dev/null | wc -l)
    fi
    echo $n
}

for CELL in "${TODO[@]}"; do
    # Wait until queue has room
    while true; do
        n=$(count_our_p4c_in_queue)
        if [ "$n" -lt "$MAX_QUEUE" ]; then
            break
        fi
        sleep $POLL_SEC
    done
    # Submit single-task array (cell index as array element)
    JID=$(sbatch --parsable --array=$CELL $SCRIPTS/sbatch_phase4_train.sh)
    SUBMITTED+=($JID)
    echo "$(date -Is)  submitted cell $CELL as jid=$JID  (currently $((n+1))/$MAX_QUEUE in queue)"
done

echo "=== all cells submitted; waiting for completion $(date -Is) ==="

# Wait for all submitted jobs to complete
while true; do
    n=$(count_our_p4c_in_queue)
    if [ "$n" -eq 0 ]; then break; fi
    echo "$(date -Is)  $n cells still running"
    sleep $POLL_SEC
done

echo "=== orch_p4c dispatch DONE $(date -Is) ==="
echo "=== metrics summary ==="
find $OUT_ROOT -name metrics.json 2>/dev/null | wc -l
echo "cells with metrics.json (target 25)"
exit 0
