#!/bin/bash
#SBATCH --job-name=abs3_gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/abs3/logs/%x_%j.out
# analysis/abs3/jobs/gpu_abs.slurm — ABS3 GPU commands (train_abs.py smoke / overfit / run, tests_abs.py gpu) in
# env_d1fu2 (torch 2.2.1+cu118) = analysis/ch3m/jobs/gpu_m.slurm with the ABS3 logs; DEP_SCRATCH=$CH3/lf and
# DEP_RESULTS=<worktree>/analysis/ch3/results/lf_inputs unchanged. Submit from analysis/abs3/ on RTX 3090 (gpu1):
#   sbatch -p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 [-J name --array=0-4%5 \
#          --output=/gpfs/tmp_cpu2/yeseo1ee/abs3/logs/%x_%A_%a.out] jobs/gpu_abs.slurm train_abs.py run GNN @I 5
# ' ::: ' = next command only if the previous exited 0; ' ;;; ' = next regardless; "@I" -> $SLURM_ARRAY_TASK_ID.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: jobs/gpu_abs.slurm <script.py> <args...> [ ::: | ;;; <script.py> <args...> ]"; exit 2; }
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
CH3=/gpfs/tmp_cpu2/yeseo1ee/ch3
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot                 # git (PREREG gate)
[ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
export GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2b DEP_JOB_T0=$(date +%s)
export DEP_SCRATCH=$CH3/lf DEP_RESULTS=$(git rev-parse --show-toplevel)/analysis/ch3/results/lf_inputs
gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "[abs3_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
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
    cmd+=("${a//@I/${SLURM_ARRAY_TASK_ID:-@I}}")
  fi
done
[ ${#cmd[@]} -gt 0 ] && run
exit $first
