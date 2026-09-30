#!/bin/bash
#SBATCH --job-name=r4_ml
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-8
# Phase 3-2: Protocol A only (Ridge / KRR_rbf / SVR_rbf / XGB, nested per-seed GridSearchCV 5-fold) on ESPLEY46/54/73
# for one G1 target: element i -> target TARGET_SETS["rev4"][i], trained on the pre-registered
# results_rev4/rows_rev4.csv. -> $ESPLEY_OUT/rev4/g1/ml_targets/. A target whose JSON + preds exist is skipped:
# resubmit after a 48 h clip. 9 elements count against MaxSubmit 20 (or run them in one allocation with r4_pack.sh).
# Refuses to train unless PREREG_REV4.md and rows_rev4.csv are committed and unchanged (REV4 §1: pre-registration
# before Phase 3).
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/ml_%a.%A.out r4_ml_array.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
i=${SLURM_ARRAY_TASK_ID:?array job only}
case "$i" in [0-8]) ;; *) echo "array index $i out of range 0-8"; exit 2 ;; esac
GEOM=g1
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_GEOM=$GEOM ESPLEY_TARGET_SET=rev4 ESPLEY_PROTOCOLS=A
export ESPLEY_FEAT=$ROOT/xtb_features_$GEOM.parquet ESPLEY_ML_OUT=$ROOT/rev4/$GEOM
export ESPLEY_ROWS=$CODE/results_rev4/rows_rev4.csv
unset ESPLEY_ARMS ESPLEY_EXT          # rev 5 switches: rev 4 = default arms ESPLEY46/54/73, no blocks
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK:-8}
# one BLAS thread per GridSearchCV worker (unset, every worker spawns one thread per allocated core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for f in results_rev4/PREREG_REV4.md results_rev4/rows_rev4.csv; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
        echo "PREREG gate: $f is not committed or differs from HEAD — commit the pre-registration first"; exit 2
    fi
done
python -c "import sklearn, xgboost; print('py env ok: sklearn', sklearn.__version__, 'xgboost', xgboost.__version__)"
echo "element $i: GEOM=$GEOM target_idx=$i feat=$ESPLEY_FEAT rows=$ESPLEY_ROWS -> $ESPLEY_ML_OUT"
python train_ml_single.py "$i"
