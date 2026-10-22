#!/bin/bash
#SBATCH --job-name=lad_gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs/%x_%j.out
# analysis/d1_ladder/jobs/gpu.slurm — D1-REF·LADDER atom-model commands (train_atoms.py) in env_d1fu2 (torch 2.2.1+cu118),
# modelled on analysis/d1_pair/fu2/jobs/fu2.slurm (its PYENV=fu2 branch). Submit from analysis/d1_ladder/:
#   GPU (smoke / run):  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 --time=48:00:00 [-J lad_m1
#                         --array=0-3%K --output=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs/%x_%A_%a.out] \
#                         jobs/gpu.slurm train_atoms.py run @I 4
#   CPU (collect / estimate / list): sbatch -p cpu1,cpu2 --time=48:00:00 jobs/gpu.slurm train_atoms.py collect --stage 1
# GPU type not pinned (user decision 2026-10-21, as FU2). The unit fingerprint records the GPU name: a resubmitted array
# task must go to the GPU type its finished units ran on (this log's '[lad_gpu] ... gpu=<name>' line), otherwise
# train_atoms.py run exits 3 at its first cached unit. ' ::: ' = next command only if the previous exited 0; ' ;;; ' =
# next regardless (job exit = the first non-zero exit); "@I" -> $SLURM_ARRAY_TASK_ID (= fu2.slurm).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: jobs/gpu.slurm <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]"; exit 2; }
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot                 # = analysis/d1_pair/jobs/env.sh, which fu2.slurm sources before switching PY
[ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1          # = d1_pair/jobs/env.sh
export D1PAIR_SCRATCH=${D1PAIR_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/d1_pair}
export D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder D1LADDER_JOB_T0=$(date +%s)
gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "[lad_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) PY=$PY $($PY --version 2>&1) cuda_visible=${CUDA_VISIBLE_DEVICES:-none}" \
     "gpu=${gpu:-none} $(date '+%F %T')"
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
