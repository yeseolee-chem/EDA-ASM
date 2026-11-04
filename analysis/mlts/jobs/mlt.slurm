#!/bin/bash
#SBATCH --job-name=mlt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/%x_%j.out
# usage (submit from analysis/mlts/): sbatch [-p ... -c N --mem=M -J name [--gres=gpu:1] --export=ALL,MLTENV=<reactot|mlts|d1>] \
#        jobs/mlt.slurm <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]
#   environments: jobs/env.sh.  ' ::: ' = next only if exit 0;  ' ;;; ' = next regardless.
#   ORCA 6.1.1 is on PATH in every env; serial ORCA (nprocs 1) needs no MPI settings.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source jobs/env.sh || exit 2
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[mlt] host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} env=${MLTENV:-reactot} gpu=${GPU:-none} git=$(git rev-parse --short HEAD 2>/dev/null) $($PY -V 2>&1) $(date '+%F %T')"
cmd=(); rc=0
for a in "$@" ":::"; do
  if [ "$a" = ":::" ] || [ "$a" = ";;;" ]; then
    if [ ${#cmd[@]} -gt 0 ]; then
      echo "=== ${cmd[*]}  $(date '+%F %T')"
      "$PY" "${cmd[@]}"; r=$?
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
