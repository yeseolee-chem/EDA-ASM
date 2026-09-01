#!/bin/bash
# spec14 orchestrator — continuous fill (no wave synchronization).
#
# Instead of waiting for a whole wave to drain, keep the submit queue full at
# 19 geomval tasks (= 20-cap − 1 for orch itself). As tasks finish, top up
# the queue with the next contiguous block. Slow tail-tasks no longer block
# future dispatches.
#
# Idempotent: each array element in 03_submit.sh skips if eda.out already
# contains "ORCA TERMINATED NORMALLY". Safe to re-run from 0; already-done
# tasks exit in seconds.
#
#SBATCH --job-name=geom_orch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation/logs/orch.%j.log

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation
SUBMIT=$BASE/03_submit.sh
POLL=30
TOTAL=218
CAP=19          # 20-submit − 1 orch
TARGET_QUEUE=19 # keep this many geomval tasks queued at all times

echo "=== orch_geomval CONTINUOUS start $(date -Is) ==="

NEXT=0
while [ $NEXT -lt $TOTAL ]; do
    # Count outstanding geomval tasks (queued + running, minus orch)
    Q=$(squeue -u $USER -h -r --name=geomval 2>/dev/null | wc -l)
    NEED=$((TARGET_QUEUE - Q))
    REM=$((TOTAL - NEXT))
    [ $NEED -gt $REM ] && NEED=$REM
    if [ $NEED -le 0 ]; then
        sleep $POLL
        continue
    fi
    END=$((NEXT + NEED - 1))

    JID=$(sbatch --parsable --array=$NEXT-$END%10 $SUBMIT 2>/dev/null)
    if [ -z "$JID" ]; then
        echo "$(date -Is)  sbatch failed for $NEXT-$END, retrying in ${POLL}s"
        sleep $POLL
        continue
    fi
    echo "$(date -Is)  filled $NEXT-$END ($NEED tasks) → jid=$JID"
    NEXT=$((END + 1))
    sleep $POLL   # let SLURM register submission before next probe
done

echo "$(date -Is)  all $TOTAL tasks dispatched; waiting for drain"
while [ -n "$(squeue -u $USER -h -r --name=geomval 2>/dev/null)" ]; do
    sleep $POLL
done

echo "=== orch_geomval DONE $(date -Is) ==="
