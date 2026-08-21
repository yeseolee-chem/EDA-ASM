#!/bin/bash
# Second-stage orch: submits Phase 4c chunks sequentially after P4b physics done.
# Split 25 cells into 5 chunks of 5 tasks (each %3 concurrent).
#SBATCH --job-name=orch_p4c
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/orch_p4c.%j.log

set -uo pipefail
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts

wait_jobs() {
    local ids="$1"
    while true; do
        n=$(squeue -j "$ids" -h 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && break
        sleep 30
    done
}

echo "=== orch_p4c start $(date -Is) ==="
for CHUNK in "0-4" "5-9" "10-14" "15-19" "20-24"; do
    echo "=== P4c chunk $CHUNK $(date -Is) ==="
    JID=$(sbatch --parsable --array=$CHUNK%3 $SCRIPTS/sbatch_phase4_train.sh)
    echo "P4c[$CHUNK] jid=$JID"
    wait_jobs "$JID"
    echo "P4c[$CHUNK] done $(date -Is)"
done

echo "=== orch_p4c DONE $(date -Is) ==="
exit 0
