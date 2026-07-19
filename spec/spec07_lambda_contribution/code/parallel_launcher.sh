#!/bin/bash
# SPEC_07 — parallel launcher (v4, fold-first / member-array).
#
# One sbatch = one (fold, λ). Array axis = MEMBER. So e.g.
#   "fold=0, λ=0.0" → sbatch --array=1-4 --export=FOLD=0,LAM=0.0 submit_delta_by_member.sh
#   trains members 1..4 as 4 concurrent GPU cells.
#
# Missing-member set is computed from the filesystem per (fold, λ) — so
# already-done members are skipped (e.g. member 0 is done for many pairs;
# the array becomes 1-4 in that case, or 0-4 for the m=0-missing pairs).
#
# Fires as many arrays in parallel as SLURM's MaxSubmitJobs=20 allows,
# throttled by wait_slots_free. SLURM's own MaxJobs=10 automatically caps
# concurrent RUNNING jobs.
#
# When all fired arrays finish, submits the final aggregation.

#SBATCH --job-name=s7_pl
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_pl_%A.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_pl_%A.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

REPO="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
OOF="$REPO/spec/spec07_lambda_contribution/oof"

echo "[pl] start @ $(date -Iseconds)  hostname=$(hostname)"

wait_slots_free () {
  local threshold=$1
  while true; do
    local total=$(squeue -h -u "$USER" -r 2>/dev/null | wc -l)
    if [[ $total -le $threshold ]]; then
      echo "[pl] slots ok: total=$total ≤ $threshold @ $(date +%H:%M:%S)"
      break
    fi
    echo "[pl] waiting slots: total=$total > $threshold @ $(date +%H:%M:%S)"
    sleep 60
  done
}

wait_job () {
  local jid=$1
  [[ -z "$jid" ]] && return 0
  while squeue -h -u "$USER" -j "$jid" 2>/dev/null | grep -q .; do sleep 60; done
}

submit_retry () {
  local label=$1; shift
  local jid=""
  for _ in $(seq 1 60); do
    jid=$("$@" 2>&1)
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
      echo "[pl] submitted $label -> jid=$jid"
      printf "%s" "$jid"
      return 0
    fi
    echo "[pl] submit $label FAILED: $jid  (retry in 60s)" >&2
    sleep 60
  done
  echo ""; return 1
}

# Return a comma-separated list of MEMBER indices that need training for
# a given (fold, lam_tag), by checking filesystem for existing outputs.
missing_members () {
  local FOLD=$1 TAG=$2
  local out=""
  for M in 0 1 2 3 4; do
    if [[ ! -f "$OOF/lam${TAG}/fold${FOLD}/member${M}.json" ]]; then
      if [[ -z "$out" ]]; then out="$M"; else out="${out},${M}"; fi
    fi
  done
  echo "$out"
}

# --- Build the target list: one entry per (fold, λ) with at least one
# missing member. Order: iterate λ outer, fold inner so we spread work
# across λ values evenly.
LAMBDAS_STR=("1.0" "0.0" "0.25" "0.5" "0.75")
LAMBDAS_TAG=("1p00" "0p00" "0p25" "0p50" "0p75")

declare -a TARGETS
for i in 0 1 2 3 4; do
  LAM="${LAMBDAS_STR[$i]}"
  TAG="${LAMBDAS_TAG[$i]}"
  for FOLD in 0 1 2 3 4; do
    MISSING=$(missing_members "$FOLD" "$TAG")
    if [[ -n "$MISSING" ]]; then
      TARGETS+=("$FOLD|$LAM|$TAG|$MISSING")
    fi
  done
done

echo "[pl] targets to fire: ${#TARGETS[@]}  (each = one (fold,λ), array over MEMBER)"
for t in "${TARGETS[@]}"; do echo "  $t"; done

# --- Fire all targets in parallel, throttled by wait_slots_free ---
declare -a ALL_JIDS
for t in "${TARGETS[@]}"; do
  IFS='|' read -r FOLD LAM TAG MISSING <<< "$t"

  # Guard: 5 = worst-case array size when all members missing.
  # threshold 14 → 14 + 5 = 19 ≤ 20 cap. If missing set is smaller we're
  # even safer.
  wait_slots_free 14

  if [[ "$LAM" == "1.0" ]]; then
    JID=$(submit_retry "base fold=$FOLD members=$MISSING" sbatch --parsable \
      --array=${MISSING} \
      --export=ALL,FOLD=$FOLD \
      spec/spec07_lambda_contribution/code/submit_base_by_member.sh)
  else
    JID=$(submit_retry "delta fold=$FOLD λ=$LAM members=$MISSING" sbatch --parsable \
      --array=${MISSING} \
      --export=ALL,FOLD=$FOLD,LAM=$LAM \
      spec/spec07_lambda_contribution/code/submit_delta_by_member.sh)
  fi
  [[ -n "$JID" ]] && ALL_JIDS+=("$JID")
done

echo "[pl] all targets submitted: ${#ALL_JIDS[@]} arrays"
echo "[pl] waiting for all to drain before final aggregation ..."

for jid in "${ALL_JIDS[@]}"; do
  wait_job "$jid"
done

echo "[pl] all cells done — submitting final aggregation @ $(date -Iseconds)"
wait_slots_free 18
JID_AGG=$(submit_retry "final agg" sbatch --parsable \
  spec/spec07_lambda_contribution/code/submit_agg.sh)
echo "[pl] final agg jid=$JID_AGG"
echo "[pl] end @ $(date -Iseconds)"
