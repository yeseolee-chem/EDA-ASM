# analysis/mlts/jobs/env.sh — environment for DEPLOY-V2 (MLTS) jobs; sourced by jobs/mlt.slurm and jobs/sbx.sh (compute nodes only).
#   MLTENV=reactot (default): conda reactot;  mlts: conda mlts (MLIPs);  d1: the D1 run environment (venv python 3.9 +
#   autodE 1.4.5, xtb 6.7.1, ORCA 6.1.1, OpenMPI 4.1.8f; d1_run/nurion/env.sh). Sets PY.
case "${MLTENV:-reactot}" in
  d1) source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh; PY="${D1_PY:?}" ;;
  reactot) source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot; PY=python ;;
  mlts) source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate mlts; PY=python ;;
  *) echo "unknown MLTENV ${MLTENV}"; return 2 2>/dev/null || exit 2 ;;
esac
export ORCA_DIR=/home1/yeseo1ee/orca_6_1_1_avx2
export PATH=$ORCA_DIR:$PATH LD_LIBRARY_PATH=$ORCA_DIR:${LD_LIBRARY_PATH:-}
export MLT_SCRATCH=${MLT_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/mlts} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MLT_JOB_T0=${MLT_JOB_T0:-$(date +%s)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
