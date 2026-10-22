#!/bin/bash
#SBATCH --job-name=d1g2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g2/logs/%x_%j.out
# usage (submit from analysis/d1_g2/): sbatch [-p ... --cpus-per-task=N --mem=M -J name --export=ALL,G2ENV=<d1|reactot>] \
#        jobs/g2.slurm <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]
#   G2ENV=d1 (default): the D1 run environment (d1_run/nurion/env.sh: venv python 3.9 + autodE 1.4.5 + rdkit, xtb 6.7.1,
#   ORCA 6.1.1, OpenMPI 4.1.8f); reactot: conda reactot.  ' ::: ' = next only if exit 0;  ' ;;; ' = next regardless.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
case "${G2ENV:-d1}" in
  d1) source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh; PY="${D1_PY:?}" ;;
  reactot) source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot; PY=python ;;
  *) echo "unknown G2ENV ${G2ENV}"; exit 2 ;;
esac
export D1G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_g2 PYTHONUNBUFFERED=1 D1G2_JOB_T0=$(date +%s)
echo "[d1g2] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} env=${G2ENV:-d1} git=$(git rev-parse --short HEAD 2>/dev/null) $($PY -V 2>&1) $(date '+%F %T')"
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
