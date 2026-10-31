#!/bin/bash
#SBATCH --job-name=gb_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/%x_%j.out
# submit from analysis/geo1b/:  sbatch [-p ... --cpus-per-task=N --mem=M -J name --array=...] jobs/cpu.slurm <script.py> <args...>
# "@I" -> $SLURM_ARRAY_TASK_ID.  G2 inputs = G2c (decision A-G2c): D1 $G2C_SCRATCH/d1/g2_final (geo_common),
# D0 $D0G2_SCRATCH/g2_final = $G2C_SCRATCH/d0/g2_final.  D1G2_SCRATCH stays the old D0-G2 root unless GEO_D1G2_SCRATCH
# is exported (no GEO1b code reads it; it is never pointed at a G2c build root).
# Test-only overrides (GB-negtest; distinct names, so nothing inherits them by accident): GEO_D0G2_SCRATCH replaces
# D0G2_SCRATCH, GEO_TEST_SCRATCH replaces GEO_SCRATCH (a throwaway dir holding a symlink d0 -> $GEO1B/d0), e.g. the
# data.py negative test:  sbatch --export=ALL,GEO_D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2,GEO_TEST_SCRATCH=<dir> ...
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
export GEO_SCRATCH="${GEO_TEST_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/geo1b}" G2C_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/g2c
export D0G2_SCRATCH="${GEO_D0G2_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/g2c/d0}"
export D1G2_SCRATCH="${GEO_D1G2_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/d0_g2}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 D1G2_JOB_T0=$(date +%s)
echo "[gb_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) GEO_SCRATCH=$GEO_SCRATCH G2C_SCRATCH=$G2C_SCRATCH" \
     "D0G2_SCRATCH=$D0G2_SCRATCH D1G2_SCRATCH=$D1G2_SCRATCH $($PY -V 2>&1) $(date '+%F %T')"
args=(); for a in "$@"; do args+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" "${args[@]}"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
