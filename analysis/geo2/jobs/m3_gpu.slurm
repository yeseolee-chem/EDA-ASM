#!/bin/bash
#SBATCH --job-name=geo2b_gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs/%x_%j.out
# analysis/geo2/jobs/m3_gpu.slurm — GEO2b LADDER stage-3 atom-model commands (train_atoms.py, tests/run_tests.py geo gpu)
# in env_d1fu2 (torch 2.2.1+cu118) = analysis/d1_ladder/jobs/gpu.slurm with the GEO2b scratch roots fixed (the trainers
# and d0d1_splits.py compare absolute paths built from them). Submit from analysis/geo2; the scripts are relative to
# analysis/d1_ladder (e.g. `train_atoms.py run @I 9 --stage 3`, `tests/run_tests.py geo gpu`).
#   sbatch -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 --time=48:00:00 [-J geo2b_m3 --array=0-8 \
#          --output=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs/%x_%A_%a.out] jobs/m3_gpu.slurm train_atoms.py run @I 9 --stage 3
# GPU type not pinned (LADDER / FU2 rule); the unit fingerprint records the GPU name, so a resubmitted array task must
# go to the GPU type its finished units ran on (this log's '[geo2b_gpu] ... gpu=<name>' line). ' ::: ' = next command
# only if the previous exited 0; ' ;;; ' = next regardless (job exit = the first non-zero exit); "@I" ->
# $SLURM_ARRAY_TASK_ID. Writes only through the python commands (atomic, idempotent units).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}/../d1_ladder" || exit 2
[ $# -gt 0 ] || { echo "usage: jobs/m3_gpu.slurm <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]"; exit 2; }
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot                 # = analysis/d1_ladder/jobs/gpu.slurm
[ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
export GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2b D1LADDER_JOB_T0=$(date +%s)
gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "[geo2b_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
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
