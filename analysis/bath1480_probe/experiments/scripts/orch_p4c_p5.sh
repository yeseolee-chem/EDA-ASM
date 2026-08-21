#!/bin/bash
# Second-stage orchestrator: handles P4c (3 chunks) + P5.
# Runs after P4b physics_24 completes. Blocks previous P4b via SBATCH deps.
#
# Also waits for P1/P2/P3 to finish before submitting P5.

set -uo pipefail

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
RESULTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results

echo "=== orch2 start $(date -Is) host=$(hostname) ==="

wait_jobs() {
    local ids="$1"
    while true; do
        n=$(squeue -j "$ids" -h 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && break
        sleep 20
    done
}

# --- P4c chunk 1 ---
echo "=== P4c chunk 0-9 start $(date -Is) ==="
J4C1=$(sbatch --parsable --array=0-9%3 $SCRIPTS/sbatch_phase4_train.sh)
echo "P4c1 jid=$J4C1"
wait_jobs "$J4C1"
echo "P4c1 done at $(date -Is)"

# --- P4c chunk 2 ---
echo "=== P4c chunk 10-19 start $(date -Is) ==="
J4C2=$(sbatch --parsable --array=10-19%3 $SCRIPTS/sbatch_phase4_train.sh)
echo "P4c2 jid=$J4C2"
wait_jobs "$J4C2"
echo "P4c2 done at $(date -Is)"

# --- P4c chunk 3 ---
echo "=== P4c chunk 20-24 start $(date -Is) ==="
J4C3=$(sbatch --parsable --array=20-24%3 $SCRIPTS/sbatch_phase4_train.sh)
echo "P4c3 jid=$J4C3"
wait_jobs "$J4C3"
echo "P4c3 done at $(date -Is)"

# --- P5 aggregate (also waits for P1/P2/P3 which should be long done) ---
echo "=== P5 aggregate start $(date -Is) ==="
JP5=$(sbatch --parsable $SCRIPTS/sbatch_phase5.sh)
echo "P5 jid=$JP5"
wait_jobs "$JP5"
echo "P5 done at $(date -Is)"

echo "=== orch2 ALL DONE $(date -Is) ==="
ls -la $RESULTS/comparison_v1/ 2>/dev/null || echo "WARN: no comparison_v1 output"
exit 0
