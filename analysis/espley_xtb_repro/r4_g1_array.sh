#!/bin/bash
#SBATCH --job-name=r4_g1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# Phase 1-2: G1 geometries for the 5,260 ok rxns in 18 interleaved slices. Array element i = slices i and i+9,
# run one after the other (9 elements, so the shared MaxSubmit=20 keeps room for other work); per slice
# 8 concurrent ORCA OptTS runs x %pal nprocs 2. Idempotent: rxns with .done / .fail_* are skipped, so a task
# clipped by the 48 h wall is simply resubmitted.
#   sbatch --array=0-8 --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/g1.%A_%a.out r4_g1_array.sh
#   then: sbatch --dependency=afterany:<JID> ... r4_g1_summary.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
for s in ${SLURM_ARRAY_TASK_ID} $((SLURM_ARRAY_TASK_ID + 9)); do
    python g1_geom.py run $s 18
done
