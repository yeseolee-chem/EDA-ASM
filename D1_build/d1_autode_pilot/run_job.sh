#!/bin/bash
# run_job.sh — one manifest job per array task: ts -> ref -> inputs -> sp -> label.
# Submit only through submit_pilot.sh (it sets -p, --array, -o, the S0 dependency and PILOT_DIR).
#SBATCH --job-name=d1pilot
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
set -uo pipefail
source "$PILOT_DIR/pilot_env.sh"
MAN=${D1P_MANIFEST:-$PILOT_DIR/pilot_manifest.csv}
JOB=$(awk -F, -v k=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==k{print $1}' "$MAN")
[ -n "$JOB" ] || { echo "no manifest row for task $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "=== $JOB (task $SLURM_ARRAY_TASK_ID) $(date -Is) $(hostname) ==="
run_stage "$JOB" ts     python run_ts.py --job "$JOB"            || exit 0
run_stage "$JOB" ref    python make_reference.py --job "$JOB"    || exit 0
run_stage "$JOB" inputs python build_sp_inputs.py --job "$JOB"   || exit 0
run_stage "$JOB" sp     run_sp "$JOB"                            || exit 0
run_stage "$JOB" label  python assemble.py --job "$JOB"          || exit 0
echo "=== $JOB complete $(date -Is) ==="
