#!/bin/bash
#SBATCH --job-name=ch3m_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/ch3m/logs/%x_%j.out
# analysis/ch3m/jobs/cpu_m.slurm — CH3M CPU python commands (= analysis/ch3/jobs/cpu_ch3.slurm with the CH3M logs; the
# regenerated lf inputs of ch3 stay DEP_SCRATCH=$CH3/lf, DEP_RESULTS=<worktree>/analysis/ch3/results/lf_inputs).
# Submit from analysis/ch3m/; CH3MENV selects the interpreter:
#   CH3MENV=d1fu2   env_d1fu2 on CPU (torch, pandas 2.1.1): s0_m, xgb_pack, tests_m cpu / merge, train_m collect
#   CH3MENV=espley  env_espley (xgboost 3.2.0, sklearn 1.3.1, numpy 1.26.0, pandas 2.1.1): xgb_m, eval_m
#                   (XGB: --cpus-per-task=32; LAD_XGB_NJOBS=$SLURM_CPUS_PER_TASK, worker count does not change results)
#   CH3MENV=reactot conda reactot (git): prereg_m write / check
# ' ::: ' = next command only if the previous exited 0; ' ;;; ' = next regardless; "@I" -> $SLURM_ARRAY_TASK_ID.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: CH3MENV=<d1fu2|espley|reactot> jobs/cpu_m.slurm <script.py> <args...>"; exit 2; }
CH3=/gpfs/tmp_cpu2/yeseo1ee/ch3
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
export GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2b DEP_JOB_T0=$(date +%s)
export DEP_SCRATCH=$CH3/lf DEP_RESULTS=$(git rev-parse --show-toplevel)/analysis/ch3/results/lf_inputs
export LAD_XGB_NJOBS=${SLURM_CPUS_PER_TASK:-4}
case "${CH3MENV:-reactot}" in
  espley)  export PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python PYTHONNOUSERSITE=1 \
             TMPDIR=${SLURM_TMPDIR:-/tmp} ;;
  d1fu2)   ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
           [ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built"; exit 2; }
           export PY=$ENV/bin/python PYTHONNOUSERSITE=1 ;;
  reactot) export PY=$(command -v python) ;;
  *) echo "unknown CH3MENV ${CH3MENV}"; exit 2 ;;
esac
echo "[ch3m_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "env=${CH3MENV:-reactot} git=$(git rev-parse --short HEAD 2>/dev/null) PY=$PY $($PY --version 2>&1) $(date '+%F %T')"
cmd=(); first=0
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"
        [ $first -eq 0 ] && [ $rc -ne 0 ] && first=$rc; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then
    run || exit $first; cmd=()
  elif [ "$a" = ";;;" ]; then
    run; cmd=()
  else
    cmd+=("${a//@I/${SLURM_ARRAY_TASK_ID:-@I}}")
  fi
done
[ ${#cmd[@]} -gt 0 ] && run
exit $first
