# analysis/d1_pair/jobs/env.sh — sourced by every D1PAIR sbatch script (compute nodes only).
# Submit from analysis/d1_pair/ (CODE = $SLURM_SUBMIT_DIR). PYENV selects the interpreter:
#   reactot  conda env reactot (Phase P, G1: same env as rev 4 g1_geom / rev 6 r6_recover)
#   venv5    rev 5 venv = reactot + tblite 0.7.0 (Phase F: same interpreter as rev 6 r6_F.sh)
#   espley   env_espley (TF 2.16.1 / keras 3.3.3 / keras_tuner 1.4.7 / sklearn 1.3.1 / xgboost 3.2.0; F0, M, E: as r6_train)
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export D1PAIR_SCRATCH=${D1PAIR_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/d1_pair}
export CODE=${D1PAIR_CODE:-$SLURM_SUBMIT_DIR}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
case "${PYENV:-reactot}" in
  reactot) export PY=$(command -v python) ;;
  venv5)   export PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python ;;
  espley)  export PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python PYTHONNOUSERSITE=1 \
             TF_NUM_INTRAOP_THREADS=${D1PAIR_TF_THREADS:-2} TF_NUM_INTEROP_THREADS=1 TMPDIR=${SLURM_TMPDIR:-/tmp} ;;  # = r6_run.sh
  *) echo "unknown PYENV $PYENV"; exit 2 ;;
esac
cd "$CODE" || exit 2
echo "[d1pair env] host=$(hostname) code=$CODE git=$(git -C "$CODE" rev-parse --short HEAD) py=$PY $($PY --version 2>&1) cpus=${SLURM_CPUS_PER_TASK:-?} $(date '+%F %T')"
