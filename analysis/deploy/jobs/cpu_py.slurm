#!/bin/bash
#SBATCH --job-name=dep_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/deploy/logs/%x_%j.out
# analysis/deploy/jobs/cpu_py.slurm — DEPLOY-MODEL CPU python commands (= analysis/geo2/jobs/m3_cpu.slurm env selector).
# Submit from analysis/deploy/; DEPENV selects the interpreter:
#   DEPENV=d1fu2   env_d1fu2 on CPU (torch, pandas 2.1.1): tests.py cpu / merge, train_tf.py collect / list,
#                  data_lf.py summary
#   DEPENV=espley  env_espley (numpy 1.26.0, pandas 2.1.1, scipy; = analysis/d1_pair/jobs/env.sh PYENV=espley): eval_tf.py
#                  selftest / run
#   DEPENV=reactot conda reactot (git, pandas): prereg_deploy.py write / check, tests.py merge
#   sbatch -p cpu1,cpu2 --time=48:00:00 --export=ALL,DEPENV=espley -J dep_e jobs/cpu_py.slurm eval_tf.py run
# ' ::: ' = next command only if the previous exited 0; ' ;;; ' = next regardless; "@I" -> $SLURM_ARRAY_TASK_ID.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: DEPENV=<d1fu2|espley|reactot> jobs/cpu_py.slurm <script.py> <args...>"; exit 2; }
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
export GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2b DEP_JOB_T0=$(date +%s)
case "${DEPENV:-reactot}" in
  espley)  export PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python PYTHONNOUSERSITE=1 \
             TMPDIR=${SLURM_TMPDIR:-/tmp} ;;
  d1fu2)   ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
           [ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built"; exit 2; }
           export PY=$ENV/bin/python PYTHONNOUSERSITE=1 ;;
  reactot) export PY=$(command -v python) ;;
  *) echo "unknown DEPENV ${DEPENV}"; exit 2 ;;
esac
echo "[dep_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "env=${DEPENV:-reactot} git=$(git rev-parse --short HEAD 2>/dev/null) PY=$PY $($PY --version 2>&1) $(date '+%F %T')"
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
