#!/bin/bash
#SBATCH --job-name=r4_cmp_train
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-8
# Phase 4-1/4-2 training on Espley's ds3 rows and splits (compare_espley.py). Array index:
#    0-6   G1 (xTB geometry) train k=index    k: 0-4 Espley targets, 5-6 role d1/d2 (ESPLEY46 + ESPLEY73 x
#          Ridge/KRR/SVR/XGB)
#    7     Espley 46 AM1 feat., role d1/d2, their protocol; first re-runs their index d1/d2 SVR/KRR, which must give
#          their hps.pkl params (fixes the tuning columns, 47 then 46; neither -> 47, flagged, see role_theirs.json)
#    8     Espley 46 AM1 feat., role d1/d2, our pipeline
# -> $R4_SCRATCH/compare/{g1,espley}/. 9 array tasks count against MaxSubmit 20. Existing outputs are skipped,
# so a task clipped by the 48 h wall is simply resubmitted.
#   sbatch --dependency=afterok:<PREP_JID> --partition=cpu1,cpu2 \
#          --output=$R4_SCRATCH/logs/compare_train_%a.%A.out r4_compare_train.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK:-8}
# one BLAS thread per GridSearchCV worker (unset, every worker spawns one thread per allocated core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
i=${SLURM_ARRAY_TASK_ID:?array job only}
case $i in
    [0-6])
        # derived here, so a stale ESPLEY_GEOM / ESPLEY_FEAT in the submit shell cannot point elsewhere
        export ESPLEY_GEOM=g1 ESPLEY_FEAT=$ROOT/xtb_features_g1.parquet
        echo "GEOM=$ESPLEY_GEOM feat=$ESPLEY_FEAT k=$i"
        python compare_espley.py train "$i" ;;
    7) python compare_espley.py espley_role theirs ;;
    8) python compare_espley.py espley_role ours ;;
    *) echo "array index $i out of range 0-8"; exit 2 ;;
esac
