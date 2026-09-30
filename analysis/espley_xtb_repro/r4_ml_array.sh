#!/bin/bash
#SBATCH --job-name=r4_ml
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-17%10
# Phase 3-2: Protocol A only (Ridge / KRR_rbf / SVR_rbf / XGB, nested per-seed GridSearchCV 5-fold) on ESPLEY46/54/73
# for one (geometry, target): element i -> geom g0 if i < 9 else g1, target TARGET_SETS["rev4"][i % 9].
# Both geometries train on the pre-registered results_rev4/rows_rev4.csv (same rows, same splits).
# -> $ESPLEY_OUT/rev4/<geom>/ml_targets/. A target whose JSON + preds exist is skipped: resubmit after a 48 h clip.
# 18 elements count against MaxSubmit 20. Refuses to train unless PREREG_REV4.md and rows_rev4.csv are committed
# and unchanged (REV4 §1: pre-registration before Phase 3).
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/ml_%a.%A.out r4_ml_array.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 9 ]; then GEOM=g0; else GEOM=g1; fi
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_GEOM=$GEOM ESPLEY_TARGET_SET=rev4 ESPLEY_PROTOCOLS=A
export ESPLEY_FEAT=$ROOT/xtb_features_$GEOM.parquet ESPLEY_ML_OUT=$ROOT/rev4/$GEOM
export ESPLEY_ROWS=$CODE/results_rev4/rows_rev4.csv
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK:-8}
for f in results_rev4/PREREG_REV4.md results_rev4/rows_rev4.csv; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
        echo "PREREG gate: $f is not committed or differs from HEAD — commit the pre-registration first"; exit 2
    fi
done
python -c "import sklearn, xgboost; print('py env ok: sklearn', sklearn.__version__, 'xgboost', xgboost.__version__)"
echo "element $i: GEOM=$GEOM target_idx=$((i % 9)) feat=$ESPLEY_FEAT rows=$ESPLEY_ROWS -> $ESPLEY_ML_OUT"
python train_ml_single.py $((i % 9))
