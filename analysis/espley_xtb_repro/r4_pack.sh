#!/bin/bash
#SBATCH --job-name=r4_pack
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1
# Run several elements of an r4 array script inside ONE allocation (one queue slot instead of one per element, so
# the shared MaxSubmit=20 is not exhausted): PACK_PAR elements at a time, each with PACK_PER cores, each element's
# log in $R4_SCRATCH/logs/<PACK_NAME>_<index>.<jobid>.out. The array scripts are idempotent, so a clipped pack is
# simply resubmitted. Exit 1 if any element failed.
#   sbatch --cpus-per-task=40 --mem=80G --partition=cpu1,cpu2 \
#          --export=ALL,PACK_SCRIPT=r4_ml_array.sh,PACK_INDICES="0 1 2 3 4 5 6 7 8",PACK_PAR=5,PACK_PER=8,PACK_NAME=ml_g0 \
#          --output=$R4_SCRATCH/logs/pack_ml_g0.%j.out r4_pack.sh
set -uo pipefail
: "${PACK_SCRIPT:?}" "${PACK_INDICES:?}" "${PACK_PAR:?}" "${PACK_PER:?}" "${PACK_NAME:?}"
LOGS=${R4_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev4}/logs
echo "[pack] $PACK_NAME: $PACK_SCRIPT indices [$PACK_INDICES] par $PACK_PAR x $PACK_PER cores on $(hostname)"
printf '%s\n' $PACK_INDICES | xargs -P "$PACK_PAR" -I{} bash -c \
    'env SLURM_ARRAY_TASK_ID={} SLURM_CPUS_PER_TASK='"$PACK_PER"' bash "$SLURM_SUBMIT_DIR/'"$PACK_SCRIPT"'" \
         > "'"$LOGS/${PACK_NAME}"'_{}.'"$SLURM_JOB_ID"'.out" 2>&1; rc=$?; echo "[pack] element {} exit $rc"; exit $rc'
rc=$?
echo "[pack] $PACK_NAME done, xargs exit $rc"
exit $rc
