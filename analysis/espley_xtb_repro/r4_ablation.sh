#!/bin/bash
#SBATCH --job-name=r4_ablation
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-8
# Phase 4-3: geometry-information ablation on Espley ds3 (espley_fairness_ablation.py): 42 (arm, target, model) jobs
# per row rule (ABL_ROWS=common default | own), arms A, C, C0 (Espley AM1 only), B_g1, O_g1 (G1) x KRR_rbf / SVR_rbf;
# `python espley_fairness_ablation.py list`. Element i runs jobs i, i+9, i+18, ... (N_SLICES = 9 in the script: keep
# --array=0-8; 9 elements count against MaxSubmit 20). Needs $ESPLEY_OUT/xtb_features_g1.parquet (or $ESPLEY_FEAT_G1).
# -> $R5_SCRATCH/phaseA/ablation[_ownrows]/<arm>__<target>__<model>.json. An existing JSON is skipped: resubmit after
# a 48 h clip. r5_phaseA.sh runs both row rules in one allocation instead.
#   sbatch --partition=cpu1,cpu2 [--export=ALL,ABL_ROWS=own] --output=$R4_SCRATCH/logs/ablation_%a.%A.out r4_ablation.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
# GridSearchCV n_jobs = the allocation, 1 BLAS thread per worker; PYTHONWARNINGS also silences the joblib workers
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
python -c "import sklearn, pyarrow; print('py env ok: sklearn', sklearn.__version__)"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
echo "element $SLURM_ARRAY_TASK_ID: rows ${ABL_ROWS:-common} feat ${ESPLEY_FEAT_G1:-$ROOT/xtb_features_g1.parquet}" \
     "labels $ESPLEY_LABELS -> $R5_SCRATCH/phaseA"
python espley_fairness_ablation.py run --slice "$SLURM_ARRAY_TASK_ID"
