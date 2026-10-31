#!/bin/bash
#SBATCH --job-name=gb_d0lab
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/%x_%j.out
# GEO1b D0D1 CPU steps in conda reactot (python 3.10, rdkit, networkx, pyarrow). jobs/cpu.slurm runs the D1 venv, which
# has no pyarrow (refs_q.py decision D1), so d0_labels.py (Q_keys_d0.parquet) needs this wrapper.
# submit from analysis/geo1b/:  sbatch -p <cpu1|cpu2> jobs/d0lab.slurm <script.py> <args...>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot || { echo "STOP: conda activate reactot failed"; exit 2; }
export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo1b PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[gb_d0lab] host=$(hostname) job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) $(python -V 2>&1) $(date '+%F %T')"
python "$@"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
