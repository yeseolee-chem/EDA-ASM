#!/bin/bash
#SBATCH --job-name=d1fu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/logs/%x_%j.out
# fu2/jobs/fu2.slurm — run FU2 python commands on a compute node (submit from analysis/d1_pair/):
#   sbatch [-p ...] [--gres=gpu:1] --export=ALL,PYENV=<reactot|venv5|espley|fu2> fu2/jobs/fu2.slurm <script.py> <args...>
# PYENV=fu2 -> env_d1fu2 (torch 2.2.1+cu118; fu2/jobs/env_build.slurm); the others = jobs/env.sh (rev 9 envs).
# Phase M training GPU jobs: any GPU  -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1  (ADD2 2 / 8.5, user decision 2026-10-21;
# was A6000 only). The unit fingerprint records the GPU name (spec M-3): a resubmitted slice must go to the GPU type its
# finished units ran on (gpu2 = gpu6 = A10, gpu4 = gpu5 = RTX A6000), otherwise run_unit exits 3 on the first cached unit.
# Phase C tests / smoke are hardware-independent (float64 checks; the GPU name is recorded): any GPU partition.
# ' ::: ' separates commands (each runs only if the previous one exited 0); ' ;;; ' runs the next command regardless
# (job exit = the first non-zero exit); "@I" -> $SLURM_ARRAY_TASK_ID (= jobs/py.slurm).
set -uo pipefail
want=${PYENV:-reactot}
[ "$want" = fu2 ] && export PYENV=reactot
source "$SLURM_SUBMIT_DIR/jobs/env.sh"
if [ "$want" = fu2 ]; then
  export PY=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/bin/python PYTHONNOUSERSITE=1 PYENV=fu2
  [ -f /gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/.d1fu2_env_ok ] || { echo "STOP: env_d1fu2 not built"; exit 2; }
  echo "[d1fu2] PY=$PY $($PY --version 2>&1) gpu=${CUDA_VISIBLE_DEVICES:-none} $(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
fi
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
run; exit $first
