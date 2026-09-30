#!/bin/bash
#SBATCH --job-name=r4_ablation
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-9
# Phase 4-3: geometry-information ablation on Espley ds3 (espley_fairness_ablation.py): 68 (arm, target, model) jobs,
# arms A, B, B_role, C, C0, O (G0), B_g1, O_g1 (G1) x KRR_rbf / SVR_rbf; `python espley_fairness_ablation.py list`.
# Element i runs jobs i, i+10, i+20, ... (N_SLICES = 10 in the script: keep --array=0-9; 10 elements count against
# MaxSubmit 20). Needs $ESPLEY_OUT/xtb_features_g0.parquet + xtb_features_g1.parquet (r4_feat_aggregate.sh GEOM=dft,
# g1), or $ESPLEY_FEAT_G0 / $ESPLEY_FEAT_G1.
# -> $R4_SCRATCH/ablation/<arm>__<target>__<model>.json. An existing JSON is skipped: resubmit after a 48 h clip.
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/ablation_%a.%A.out r4_ablation.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
# GridSearchCV n_jobs = the allocation, 1 BLAS thread per worker; PYTHONWARNINGS also silences the joblib workers
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK:-8} OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
python -c "import sklearn, pyarrow; print('py env ok: sklearn', sklearn.__version__)"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
echo "element $SLURM_ARRAY_TASK_ID: feat ${ESPLEY_FEAT_G0:-$ROOT/xtb_features_g0.parquet}" \
     "${ESPLEY_FEAT_G1:-$ROOT/xtb_features_g1.parquet} labels $ESPLEY_LABELS -> $R4_SCRATCH/ablation"
python espley_fairness_ablation.py run --slice "$SLURM_ARRAY_TASK_ID"
