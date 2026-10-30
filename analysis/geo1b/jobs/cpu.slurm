#!/bin/bash
#SBATCH --job-name=gb_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/%x_%j.out
# submit from analysis/geo1b/:  sbatch [-p ... --cpus-per-task=N --mem=M -J name --array=...] jobs/cpu.slurm <script.py> <args...>
# "@I" -> $SLURM_ARRAY_TASK_ID.  D1G2_SCRATCH: $D0G2 unless GEO_D1G2_SCRATCH is exported (only for check Z-1).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo1b D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
export D1G2_SCRATCH="${GEO_D1G2_SCRATCH:-$D0G2_SCRATCH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 D1G2_JOB_T0=$(date +%s)
echo "[gb_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) D1G2_SCRATCH=$D1G2_SCRATCH $($PY -V 2>&1) $(date '+%F %T')"
args=(); for a in "$@"; do args+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" "${args[@]}"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
