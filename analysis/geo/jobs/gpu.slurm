#!/bin/bash
#SBATCH --job-name=geo_gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo/logs/%x_%j.out
# submit from analysis/geo/:  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 jobs/gpu.slurm <script.py> <args...>
# CPU-only use (select / evaluate):  sbatch -p <cpu partition> jobs/gpu.slurm <script.py> <args...>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[geo_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD 2>/dev/null)" \
     "$($PY --version 2>&1) gpu=${gpu:-none} $(date '+%F %T')"
"$PY" "$@"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
