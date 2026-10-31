#!/bin/bash
#SBATCH --job-name=geo2b_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs/%x_%j.out
# analysis/geo2/jobs/m3_cpu.slurm — GEO2b LADDER-training CPU commands with the GEO2b scratch roots fixed. Submit from
# analysis/geo2; the scripts are relative to analysis/d1_ladder (../geo2/<script> for the analysis/geo2 ones).
#   M3ENV=espley  env_espley (xgboost 3.2.0, scikit-learn 1.3.1; = analysis/d1_pair/jobs/env.sh PYENV=espley):
#                 train_xgb.py run / check / smoke / diag / l6, tests/run_tests.py geo cpu, ../geo2/lad_selftest.py xgb
#   M3ENV=d1fu2   env_d1fu2 on CPU (torch, no GPU): train_atoms.py collect --stage 3 [--with-xgb], list,
#                 ../geo2/lad_selftest.py atoms
#   M3ENV=reactot conda reactot (pandas, git): ../geo2/d0d1_splits.py, ../geo2/prereg_geo2b.py, tests/run_tests.py geo
#                 merge, ../geo2/lad_selftest.py splits / lists
#   sbatch -p cpu1,cpu2 --time=48:00:00 --cpus-per-task=32 --mem=64G --export=ALL,M3ENV=espley -J geo2b_x3 \
#          jobs/m3_cpu.slurm train_xgb.py run 0 1 --stage 3
# ' ::: ' = next command only if the previous exited 0; ' ;;; ' = next regardless; "@I" -> $SLURM_ARRAY_TASK_ID.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}/../d1_ladder" || exit 2
[ $# -gt 0 ] || { echo "usage: M3ENV=<espley|d1fu2|reactot> jobs/m3_cpu.slurm <script.py> <args...>"; exit 2; }
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
export GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2b D1LADDER_JOB_T0=$(date +%s)
case "${M3ENV:-reactot}" in
  espley)  export PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python PYTHONNOUSERSITE=1 \
             TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 TMPDIR=${SLURM_TMPDIR:-/tmp} ;;   # = d1_pair jobs/env.sh
  d1fu2)   ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
           [ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built"; exit 2; }
           export PY=$ENV/bin/python PYTHONNOUSERSITE=1 ;;
  reactot) export PY=$(command -v python) ;;
  *) echo "unknown M3ENV ${M3ENV}"; exit 2 ;;
esac
echo "[geo2b_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "env=${M3ENV:-reactot} git=$(git rev-parse --short HEAD 2>/dev/null) PY=$PY $($PY --version 2>&1) $(date '+%F %T')"
cmd=(); first=0
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"
        [ $first -eq 0 ] && [ $rc -ne 0 ] && first=$rc; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then
    run || exit $first; cmd=()
  elif [ "$a" = ";;;" ]; then
    run; cmd=()
  else
    cmd+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}")
  fi
done
[ ${#cmd[@]} -gt 0 ] && run
exit $first
