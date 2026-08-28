#!/bin/bash
# spec14 orchestrator — 218 ORCA EDA calcs in waves of 20 (%10 concurrent)
# to respect the 20-submit / 10-run QoS cap. Runs on a compute node (cpu2)
# per CLAUDE.md (no login-node loops).
#
# Idempotent: each array element checks "ORCA TERMINATED NORMALLY" and skips
# if already done, so re-running or resuming after wall clip is safe.
#
#SBATCH --job-name=geom_orch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation/logs/orch.%j.log

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation
SUBMIT=$BASE/03_submit.sh
POLL=60

echo "=== orch_geomval start $(date -Is) ==="

# 11 waves × 20 tasks (last wave = 18): 0-19, 20-39, ..., 200-217
WAVE_STARTS=(0 20 40 60 80 100 120 140 160 180 200)

for START in "${WAVE_STARTS[@]}"; do
    END=$((START + 19))
    [ $END -gt 217 ] && END=217

    # Wait until my other jobs have drained enough to allow a 20-task submission
    while true; do
        MY=$(squeue -u $USER -h -r --name=geomval 2>/dev/null | wc -l)
        NEED=$((END - START + 1))
        FREE=$((20 - MY))
        if [ "$FREE" -ge "$NEED" ]; then break; fi
        sleep $POLL
    done

    echo "$(date -Is)  submitting wave $START-$END (%10)"
    JID=$(sbatch --parsable --array=$START-$END%10 $SUBMIT)
    echo "  jid=$JID"

    # Wait for this wave to fully drain before submitting the next
    while [ -n "$(squeue -j $JID -h 2>/dev/null)" ]; do
        sleep $POLL
    done
    echo "$(date -Is)  wave $START-$END drained"
done

echo "=== orch_geomval DONE $(date -Is) ==="
