#!/bin/bash
# SPEC_07 — launcher for the second half of spec07 δ training.
#
# We are limited to MaxSubmitJobs=20 pending+running by SLURM. The 20-cell
# λ×fold sweep + spec06 arrays + s7_base + s7_agg cannot all fit at once, so
# this launcher waits for the first delta batch to finish and then submits
# the remaining cells, followed by the aggregator with the correct dep.
#
# Optionally re-submits spec06's agg (canceled earlier to free a slot).
#
# 48h wall per CLAUDE.md; polling loop with 60s sleep — cheap CPU task.

#SBATCH --job-name=s7_launch
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_launch_%A.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_launch_%A.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

echo "[launcher] $(date -Iseconds)  hostname=$(hostname)"
echo "[launcher] FIRST_BATCH_JID=${FIRST_BATCH_JID:-<unset>}"
echo "[launcher] BASE_JID=${BASE_JID:-<unset>}"

wait_jid () {
  local jid=$1; local label=$2
  while squeue -h -u "$USER" -j "$jid" 2>/dev/null | grep -q .; do
    echo "[wait $label $jid] $(date +%H:%M:%S) still running/pending"
    sleep 60
  done
  echo "[wait $label $jid] drained @ $(date +%H:%M:%S)"
}

# Wait for first batch (spec07 delta 0..12) to fully drain.
if [[ -n "${FIRST_BATCH_JID:-}" ]]; then
  wait_jid "$FIRST_BATCH_JID" "delta_batch1"
fi

# Submit second batch (spec07 delta 13..19). Retry a few times if we still
# temporarily blow the submit limit (e.g. spec06 not drained).
JID_D2=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  JID_D2=$(sbatch --parsable --array=13-19%5 spec/spec07_lambda_contribution/code/submit_delta.sh 2>&1) && break
  echo "[submit d2 retry] $JID_D2"
  JID_D2=""
  sleep 120
done
echo "[launcher] delta_batch2 jid=${JID_D2}"

# Submit aggregator with dependency on second batch (afterany so partial
# aggregations still run even if a cell dies).
DEP="afterany:${JID_D2}"
if [[ -n "${BASE_JID:-}" ]]; then
  DEP="${DEP}:${BASE_JID}"
fi
JID_AGG=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  JID_AGG=$(sbatch --parsable --dependency="$DEP" \
      spec/spec07_lambda_contribution/code/submit_agg.sh 2>&1) && break
  echo "[submit agg retry] $JID_AGG"
  JID_AGG=""
  sleep 60
done
echo "[launcher] agg jid=${JID_AGG}"

# Also re-submit spec06 agg that we cancelled to free a slot. Best-effort.
if [[ -f spec/spec06_2step_xgb28_delta/code/submit_aggregate.sh ]]; then
  JID_S6AGG=""
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    JID_S6AGG=$(sbatch --parsable \
        spec/spec06_2step_xgb28_delta/code/submit_aggregate.sh 2>&1) && break
    echo "[submit s6_agg retry] $JID_S6AGG"
    JID_S6AGG=""
    sleep 60
  done
  echo "[launcher] s6_agg resubmitted: ${JID_S6AGG}"
fi

echo "[launcher] done @ $(date -Iseconds)"
