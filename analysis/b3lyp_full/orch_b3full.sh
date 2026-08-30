#!/bin/bash
# SPEC16rev orchestrator — continuous-fill + self-chaining for 5269 rxns.
#
# Each orch instance runs ≤48h. Near wall (47h elapsed), it stops dispatching,
# lets in-flight tasks continue draining, and submits the next orch instance
# with dependency=afterany so the chain persists across wall boundaries.
#
# Individual array elements have their own 48h wall + idempotent skip, so
# already-done work is preserved.
#
#SBATCH --job-name=b3_orch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/logs/orch.%j.log

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full
SUBMIT=$BASE/02_submit.sh
POLL=30
TARGET_QUEUE=19        # 20-cap − 1 orch
MAX_ELAPSED=$((47 * 3600))
SELF="$BASE/orch_b3full.sh"

START=$(date +%s)
TOTAL=$(ls -d $BASE/inputs/rxn_*/ 2>/dev/null | wc -l)
echo "=== orch_b3full start $(date -Is)  TOTAL=$TOTAL ==="

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no inputs/rxn_*. Run 00→01 first."; exit 1
fi

# Function: count fully-done rxns (all 5 SPEs terminated normally)
count_done() {
    local n=0
    for d in $BASE/inputs/rxn_*; do
        local ok=1
        for stem in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
            f="$d/$stem.out"
            if [ ! -f "$f" ]; then ok=0; break; fi
            grep -q "ORCA TERMINATED NORMALLY" "$f" || { ok=0; break; }
        done
        [ $ok -eq 1 ] && n=$((n+1))
    done
    echo $n
}

# Check if all rxns fully done before we even start dispatching
DONE=$(count_done)
echo "$(date -Is)  initial done=$DONE / $TOTAL"
if [ "$DONE" -ge "$TOTAL" ]; then
    echo "all done, no chain needed"; exit 0
fi

# Determine next incomplete rxn index (linear scan)
next_index() {
    local i=0
    for d in $BASE/inputs/rxn_*; do
        local ok=1
        for stem in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
            f="$d/$stem.out"
            if [ ! -f "$f" ] || ! grep -q "ORCA TERMINATED NORMALLY" "$f"; then
                ok=0; break
            fi
        done
        if [ $ok -eq 0 ]; then echo $i; return; fi
        i=$((i+1))
    done
    echo -1
}

NEXT=$(next_index)
echo "starting from index $NEXT"

while [ $NEXT -ge 0 ] && [ $NEXT -lt $TOTAL ]; do
    ELAPSED=$(($(date +%s) - START))
    if [ $ELAPSED -gt $MAX_ELAPSED ]; then
        echo "$(date -Is)  wall-time budget exhausted, stopping dispatch"
        break
    fi

    Q=$(squeue -u $USER -h -r --name=b3full 2>/dev/null | wc -l)
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
        echo "$(date -Is)  sbatch failed for $NEXT-$END, retry in ${POLL}s"
        sleep $POLL
        continue
    fi
    echo "$(date -Is)  filled $NEXT-$END ($NEED tasks) → jid=$JID"
    NEXT=$((END + 1))
    # Skip forward past any rxns already done to avoid resubmitting
    while [ $NEXT -lt $TOTAL ]; do
        d=$(ls -d $BASE/inputs/rxn_*/ | sed -n "$((NEXT+1))p")
        d="${d%/}"
        all_done=1
        for stem in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
            f="$d/$stem.out"
            if [ ! -f "$f" ] || ! grep -q "ORCA TERMINATED NORMALLY" "$f"; then
                all_done=0; break
            fi
        done
        [ $all_done -eq 0 ] && break
        NEXT=$((NEXT+1))
    done
    sleep $POLL
done

# Wait for dispatched jobs to finish (until wall or complete)
echo "$(date -Is)  wait for in-flight to drain (wall budget)"
while [ -n "$(squeue -u $USER -h -r --name=b3full 2>/dev/null)" ]; do
    ELAPSED=$(($(date +%s) - START))
    if [ $ELAPSED -gt $((47 * 3600 + 1800)) ]; then break; fi
    sleep $POLL
done

# Self-chain if work remains
DONE=$(count_done)
echo "$(date -Is)  final done=$DONE / $TOTAL"
if [ "$DONE" -lt "$TOTAL" ]; then
    NEXT_JID=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID $SELF)
    echo "chained next orch: $NEXT_JID"
else
    echo "all done, no chain"
fi

echo "=== orch_b3full DONE $(date -Is) ==="
