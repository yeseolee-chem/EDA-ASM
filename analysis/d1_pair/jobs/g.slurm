#!/bin/bash
#SBATCH --job-name=d1g
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/logs/%x_%j.out
# g.slurm — one D1PAIR G / F python command with the G / F environment (jobs/g_env.sh):
#   sbatch [--export=ALL,PYENV=venv5] jobs/g.slurm <script.py> <args...>      (' ::: ' separates commands)
# Array use: an argument '@I' is replaced by $SLURM_ARRAY_TASK_ID.
set -uo pipefail
source "$SLURM_SUBMIT_DIR/jobs/env.sh"
source "$SLURM_SUBMIT_DIR/jobs/g_env.sh"
cmd=()
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then run || exit $?; cmd=(); else cmd+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}"); fi
done
run; exit $?
