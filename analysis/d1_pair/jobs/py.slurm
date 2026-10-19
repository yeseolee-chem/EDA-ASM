#!/bin/bash
#SBATCH --job-name=d1p
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/logs/%x_%j.out
# py.slurm — run one D1PAIR python command on a compute node:  sbatch [--export=ALL,PYENV=...] jobs/py.slurm <script.py> <args...>
# Several commands: separate them with ' ::: ' (each runs only if the previous one exited 0).
set -uo pipefail
source "$SLURM_SUBMIT_DIR/jobs/env.sh"
cmd=()
for a in "$@"; do
  if [ "$a" = ":::" ]; then
    echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc"; [ $rc -eq 0 ] || exit $rc; cmd=()
  else
    cmd+=("$a")
  fi
done
echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"; exit $rc
