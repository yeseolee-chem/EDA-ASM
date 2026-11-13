#!/bin/bash
#SBATCH --job-name=g2r_py
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/g2r/logs/%x_%j.out
# analysis/g2r/jobs/py_g2r.slurm — D1-G2R python steps that need pandas / pyarrow but no QM engine (start conditions,
# targets, usable list, checks), in env_d1fu2 (python 3.11, pandas 2.1, pyarrow). Submit from the repository root:
#   sbatch -p cpu2 --time=48:00:00 analysis/g2r/jobs/py_g2r.slurm analysis/g2r/<script>.py [args] [ ::: <script> ...]
# No DEP_* variable is set here (the scripts take their paths from analysis/g2r/g2r_common.py).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: analysis/g2r/jobs/py_g2r.slurm <script.py> <args...> [ ::: <script.py> ... ]"; exit 2; }
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[g2r_py] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD 2>/dev/null) PY=$PY $($PY --version 2>&1) $(date '+%F %T')"
cmd=(); first=0
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"
        [ $first -eq 0 ] && [ $rc -ne 0 ] && first=$rc; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then run || exit $first; cmd=(); else cmd+=("$a"); fi
done
[ ${#cmd[@]} -gt 0 ] && run
exit $first
