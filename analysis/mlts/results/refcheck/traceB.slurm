#!/bin/bash
#SBATCH --job-name=refB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck/traceB_%x_%j.log
# usage: sbatch -J refB_<mode> traceB.slurm <stats|nurion>
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export MLT_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/mlts PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck
echo "[refB] host=$(hostname) job=${SLURM_JOB_ID} mode=$1 $(date '+%F %T')"
python /gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck/traceB.py "$1" > /gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck/traceB_$1.stdout 2>&1
echo "[refB] exit $? $(date '+%F %T')"
