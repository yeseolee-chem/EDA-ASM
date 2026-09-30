#!/bin/bash
#SBATCH --job-name=r4_downstream
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
# Phase 3-3: downstream analyses of one geometry (GEOM=g1 default, or g2) on the pre-registered rows
# (results_rev4/rows_rev4.csv)
#   analyze_extra.py  -> charge_breakdown_<geom>.csv, group_split_<geom>.csv   (reads the ML outputs of GEOM)
#   evaluate_pairs.py -> split_metrics_<geom>.csv, margin_calibration_<geom>.csv, mmp_pairs_v2_<geom>.csv
# into results_rev4/downstream_<geom>/. Existing outputs are skipped. Run after r4_ml_aggregate.sh, afterany: the
# scripts gate their own inputs (ml_report.json present, same geom + rows_sha256), so a failing rev4_tables.py at the
# end of that job does not block this one:
#   sbatch --partition=cpu1,cpu2 --dependency=afterany:<ML_AGG_JID> --export=ALL,GEOM=g1 --output=$R4_SCRATCH/logs/downstream_g1.%j.out r4_downstream.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
GEOM=${GEOM:-g1}
case "$GEOM" in g1|g2) ;; *) echo "GEOM must be g1|g2 (--export=ALL,GEOM=...), got '$GEOM'"; exit 2 ;; esac
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
# derived from GEOM / CODE only (as r4_ml_array.sh), so a stale ESPLEY_FEAT / ESPLEY_ML_OUT / ESPLEY_ROWS in the
# submit shell cannot mix geometries or row sets
export ESPLEY_GEOM=$GEOM ESPLEY_FEAT=$ROOT/xtb_features_$GEOM.parquet ESPLEY_ML_OUT=$ROOT/rev4/$GEOM
export ESPLEY_ROWS=$CODE/results_rev4/rows_rev4.csv
# rev 5 switches (r5_downstream.sh sets them in its own job): a stale ESPLEY_RES5 / ESPLEY_DOWNSTREAM_ARM in the submit
# shell would send this rev 4 run to results_rev5/downstream_<arm>/
unset ESPLEY_DOWNSTREAM_ARM ESPLEY_RES5 ESPLEY_ARMS ESPLEY_EXT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "GEOM=$GEOM feat=$ESPLEY_FEAT ml=$ESPLEY_ML_OUT rows=$ESPLEY_ROWS"
rc=0
python analyze_extra.py || rc=1
python evaluate_pairs.py || rc=1
ls -la "results_rev4/downstream_$GEOM"
exit $rc
