#!/bin/bash
#SBATCH --job-name=g2c
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/g2c/logs/%x_%j.out
# G2c reruns / selection (docs/specs/GEO1B.md 6). D1 run env. Submit from the folder of the script you run:
#   D1 rerun : cd analysis/g2c; sbatch -J g2c_d1_r1 -p <P> --cpus-per-task=<C> --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d1_r1 \
#              jobs/g2c.slurm g2c_d1.py pool /gpfs/tmp_cpu2/yeseo1ee/g2c/lists/d1_targets.csv <n_par> <cores> <i> <n>
#   D0 rerun : cd analysis/geo;  sbatch -J g2c_d0_r1 -p <P> --cpus-per-task=<C> --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1 \
#              ../g2c/jobs/g2c.slurm d0g2.py pool /gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1/d0_manifest.csv <n_par> <cores> <i> <n>
#   select   : cd analysis/g2c; sbatch -J g2c_sel -p <P> --cpus-per-task=16 --export=ALL,G2C_ROOT=none jobs/g2c.slurm select_g2c.py report 16
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
: "${G2C_ROOT:?G2C_ROOT must be exported (a $G2C/<src>_r<k> root, or none for select)}"
export G2C_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/g2c PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 G2C_STOP_AFTER_H=44 D0G2_STOP_AFTER_H=44
if [ "$G2C_ROOT" != none ]; then
  case "$G2C_ROOT" in /gpfs/tmp_cpu2/yeseo1ee/g2c/d1_r[12]|/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r[12]) ;; *) echo "STOP: bad G2C_ROOT $G2C_ROOT"; exit 3;; esac
  export D1G2_SCRATCH="$G2C_ROOT" D0G2_SCRATCH="$G2C_ROOT" GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D1G2_JOB_T0=$(date +%s)
fi
echo "[g2c] host=$(hostname) job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?} root=$G2C_ROOT git=$(git rev-parse --short HEAD 2>/dev/null)" \
     "$($PY -V 2>&1) $(date '+%F %T')"
"$PY" "$@"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
