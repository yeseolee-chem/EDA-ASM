#!/bin/bash
#SBATCH --job-name=r4_cmp_train
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-15%10
# Phase 4-1/4-2 training on Espley's ds3 rows and splits (compare_espley.py). Array index:
#    0-6   G0 (DFT oracle geometry, upper bound)  train k=index      k: 0-4 Espley targets, 5-6 role d1/d2
#    7-13  G1 (xTB geometry)                      train k=index-7    (ESPLEY46 + ESPLEY73 x Ridge/KRR/SVR/XGB)
#    14    Espley 46 AM1 feat., role d1/d2, their protocol; first re-runs their index d1/d2 SVR/KRR, which must give
#          their hps.pkl params (fixes the tuning columns, 47 then 46; neither -> 47, flagged, see role_theirs.json)
#    15    Espley 46 AM1 feat., role d1/d2, our pipeline
# -> $R4_SCRATCH/compare/{g0,g1,espley}/. 16 array tasks count against MaxSubmit 20. Existing outputs are skipped,
# so a task clipped by the 48 h wall is simply resubmitted.
#   sbatch --dependency=afterok:<PREP_JID> --partition=cpu1,cpu2 \
#          --output=$R4_SCRATCH/logs/compare_train_%a.%A.out r4_compare_train.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK:-8}
i=${SLURM_ARRAY_TASK_ID:?array job only}
case $i in
    [0-9]|1[0-3])
        g=$([ "$i" -lt 7 ] && echo g0 || echo g1)
        # derived from the index only, so a stale ESPLEY_GEOM / ESPLEY_FEAT in the submit shell cannot mix geometries
        export ESPLEY_GEOM=$g ESPLEY_FEAT=$ROOT/xtb_features_$g.parquet
        echo "GEOM=$g feat=$ESPLEY_FEAT k=$((i % 7))"
        python compare_espley.py train $((i % 7)) ;;
    14) python compare_espley.py espley_role theirs ;;
    15) python compare_espley.py espley_role ours ;;
    *) echo "array index $i out of range 0-15"; exit 2 ;;
esac
