#!/bin/bash
#SBATCH --job-name=gnn3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/gnn3/logs/%x_%j.out
# gnn3/jobs/run.slurm — run one or more python commands of the gnn3 package in env_d1fu2 (torch 2.2.1+cu118).
#
# GPU (training, equivalence checks E-2 / E-3), RTX 3090 only:
#   sbatch -p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3_e3 gnn3/jobs/run.slurm gnn3/tests/equivalence.py e3
# CPU (data, metrics, unit tests):
#   sbatch -p cpu1,cpu2 --time=48:00:00 -J gnn3_e1 gnn3/jobs/run.slurm gnn3/tests/equivalence.py e1
# Submit from the repository root. ' ::: ' runs the next command only if the previous one exited 0, ' ;;; ' always.
# The conda env 'reactot' is activated only for git (provenance); python is always env_d1fu2.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: gnn3/jobs/run.slurm <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]"; exit 2; }
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# the original pipeline reads these (fu2 / d1pair / deploy paths); the new package reads the same files
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
export DEP_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/ch3/lf DEP_RESULTS=$(git rev-parse --show-toplevel)/analysis/ch3/results/lf_inputs
gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "[gnn3] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} git=$(git rev-parse --short HEAD 2>/dev/null)" \
     "PY=$PY $($PY --version 2>&1) gpu=${gpu:-none} $(date '+%F %T')"
cmd=(); first=0
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"
        [ $first -eq 0 ] && [ $rc -ne 0 ] && first=$rc; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then
    run || exit $first; cmd=()
  elif [ "$a" = ";;;" ]; then
    run; cmd=()
  else
    cmd+=("$a")
  fi
done
[ ${#cmd[@]} -gt 0 ] && run
exit $first
