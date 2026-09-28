#!/bin/bash
# submit_pilot.sh — ONE-SHOT submission from the login node (no python, no loops, returns at once):
#   S0 preflight  ->  job array (afterok:S0, max 10 running)  ->  report (afterany:array)
set -euo pipefail
cd "$(dirname "$0")"
PILOT_DIR=$PWD
N=$(( $(wc -l < pilot_manifest.csv) - 1 ))
NEED=$(( N + 2 ))
Q=$(squeue -u "$USER" -h | wc -l)
if [ $(( Q + NEED )) -gt 20 ]; then
    echo "STOP: $Q tasks already queued; this submission needs $NEED (MaxSubmit=20 counts array tasks)."; exit 1
fi
P=$(sinfo -h -p cpu1,cpu2 -o "%P %C" | tr -d '*' | awk '{split($2,a,"/"); if (a[2]>=m) {m=a[2]; p=$1}} END {print p}')
SCR=$(grep -E '^scratch:' config.yaml | sed -E 's/^scratch:[[:space:]]*//; s/[[:space:]]+#.*$//')
mkdir -p "$SCR/logs"
S0=$(sbatch --parsable -p "$P" -o "$SCR/logs/s0_%j.log" --export=ALL,PILOT_DIR="$PILOT_DIR" s0_preflight.sh)
ARR=$(sbatch --parsable -p "$P" --array=0-$((N - 1))%10 --dependency=afterok:"$S0" --kill-on-invalid-dep=yes \
      -o "$SCR/logs/job_%A_%a.log" --export=ALL,PILOT_DIR="$PILOT_DIR" run_job.sh)
REP=$(sbatch --parsable -p "$P" --dependency=afterany:"$ARR" \
      -o "$SCR/logs/report_%j.log" --export=ALL,PILOT_DIR="$PILOT_DIR" run_report.sh)
echo "partition=$P  S0=$S0  ARRAY=$ARR (0-$((N - 1))%10)  REPORT=$REP  tasks=$NEED  queue_before=$Q"
