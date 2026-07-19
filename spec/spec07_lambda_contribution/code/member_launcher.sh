#!/bin/bash
# SPEC_07 — member launcher (v2, batch=5).
#
# Processes ONE member's worth of cells (5 base + 20 delta) sequentially,
# then chains to the next member via sbatch. Batch=5 so we can submit
# under the SLURM MaxSubmitJobs=20 cap even when spec06 is holding ~14
# slots.
#
# For each λ in the sweep (0.0, 0.25, 0.5, 0.75, 1.0):
#   sbatch the 5-fold array for that λ, wait for completion (run_lambda.py
#   is idempotent so any already-done cell is skipped instantly).
#
# Entry: sbatch --export=ALL,MEMBER=<m> member_launcher.sh
# Chains to MEMBER=m+1; when m > 4, submits final aggregation.
#
# Special: MEMBER=0 does a "recovery pass" — original spec07 workflow
# (jid 763538) failed to submit --array=13-19%5 for member=0, so we
# still need to fill lam0p50/{fold3,4} + lam0p75/{fold0..4} for m=0.

#SBATCH --job-name=s7_ml
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_ml_%A.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec07_ml_%A.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

M=${MEMBER:-0}
echo "[ml m=$M] start @ $(date -Iseconds)  hostname=$(hostname)"

# --- helpers ---
wait_slots_free () {
  local threshold=$1
  while true; do
    local total=$(squeue -h -u "$USER" -r 2>/dev/null | wc -l)
    if [[ $total -le $threshold ]]; then
      echo "[ml m=$M] slots ok: total=$total ≤ $threshold @ $(date +%H:%M:%S)"
      break
    fi
    echo "[ml m=$M] waiting slots: total=$total > $threshold @ $(date +%H:%M:%S)"
    sleep 90
  done
}

wait_job () {
  local jid=$1
  [[ -z "$jid" ]] && return 0
  echo "[ml m=$M] waiting on job $jid ..."
  while squeue -h -u "$USER" -j "$jid" 2>/dev/null | grep -q .; do sleep 90; done
  echo "[ml m=$M] job $jid drained @ $(date +%H:%M:%S)"
}

submit_retry () {
  local label=$1; shift
  local jid=""
  for _ in $(seq 1 60); do
    jid=$("$@" 2>&1)
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
      echo "[ml m=$M] submitted $label -> jid=$jid"
      printf "%s" "$jid"
      return 0
    fi
    echo "[ml m=$M] submit $label FAILED: $jid  (retry in 90s)" >&2
    sleep 90
  done
  echo ""; return 1
}

# --- process each λ for this member ---
# batch=5 cells, threshold=14 → 14+5=19 ≤ 20 cap.
# submit_delta.sh maps task = LAM_IDX*5 + fold (LAMBDAS_TRAIN=[0.0, 0.25, 0.5, 0.75])
# so we submit --array=<LI*5>-<LI*5+4>%5 for LI in 0..3.
# For λ=1.0 (base) we use submit_base.sh with --array=0-4%5.

echo "[ml m=$M] === λ=1.0 base ==="
wait_slots_free 14
JID_B=$(submit_retry "base m=$M" sbatch --parsable --array=0-4%5 \
  --export=ALL,MEMBER=$M spec/spec07_lambda_contribution/code/submit_base.sh)
wait_job "$JID_B"

for LI in 0 1 2 3; do
  START=$((LI * 5)); END=$((START + 4))
  case $LI in
    0) LAM="0.0" ;;
    1) LAM="0.25" ;;
    2) LAM="0.5" ;;
    3) LAM="0.75" ;;
  esac
  echo "[ml m=$M] === λ=$LAM delta (array=${START}-${END}) ==="
  wait_slots_free 14
  JID=$(submit_retry "delta λ=$LAM m=$M" sbatch --parsable \
    --array=${START}-${END}%5 --export=ALL,MEMBER=$M \
    spec/spec07_lambda_contribution/code/submit_delta.sh)
  wait_job "$JID"
done

# --- chain to next member or run final aggregation ---
NEXT=$((M + 1))
if [[ $NEXT -le 4 ]]; then
  wait_slots_free 18
  JID_NEXT=$(submit_retry "member=$NEXT launcher" sbatch --parsable \
    --export=ALL,MEMBER=$NEXT \
    spec/spec07_lambda_contribution/code/member_launcher.sh)
  echo "[ml m=$M] chained to member=$NEXT jid=$JID_NEXT"
else
  echo "[ml m=$M] members 0..4 done — submitting final aggregation"
  wait_slots_free 18
  JID_AGG=$(submit_retry "final agg" sbatch --parsable \
    spec/spec07_lambda_contribution/code/submit_agg.sh)
  echo "[ml m=$M] final agg jid=$JID_AGG"
fi

echo "[ml m=$M] end @ $(date -Iseconds)"
