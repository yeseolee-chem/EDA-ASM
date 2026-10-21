#!/bin/bash
#SBATCH --job-name=d1hf
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/logs/%x_%j.out
# usage (submit from analysis/d1_hf/): sbatch [-p cpu1,cpu2 --cpus-per-task=N --mem=M -J name] jobs/hf.slurm \
#        <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]
#   ' ::: ' = run the next command only if this one exited 0;  ' ;;; ' = run the next command regardless.
#   The job exit code is the first non-zero exit code.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source jobs/hf_env.sh
export D1HF_JOB_T0=$(date +%s)
echo "[d1hf] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} git=$(git rev-parse --short HEAD 2>/dev/null) $(python -V 2>&1) $(date '+%F %T')"
cmd=()
rc=0
for a in "$@" ":::"; do
  if [ "$a" = ":::" ] || [ "$a" = ";;;" ]; then
    if [ ${#cmd[@]} -gt 0 ]; then
      echo "=== ${cmd[*]}  $(date '+%F %T')"
      python "${cmd[@]}"; r=$?
      echo "=== exit $r $(date '+%F %T')"
      [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
      [ $r -ne 0 ] && [ "$a" = ":::" ] && exit $rc
    fi
    cmd=()
  else
    cmd+=("$a")
  fi
done
exit $rc
