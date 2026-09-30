#!/bin/bash
#SBATCH --job-name=r5_ext_merge
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase B-7 (docs/specs/REV5_FEATURES_FIGURES.md): r5_ext_merge.py — merge the 18 slices of r5_ext_array.sh into
# $ESPLEY_OUT/xtb_features_ext_g1.parquet, apply the B-7 gates (failure share of rows_rev4 > MAX_EXT_FAIL; NaN = any
# rows_rev4 rxn with a nonfinite:<cols> block failure, or NaN in an ok row; any B5 scan_gate_fail) and, only if all
# pass, write results_rev5/rows_rev5.csv (rows_rev4 ∩ ext ok). Always writes results_rev5/B7_report.json and
# B7_feature_quantiles.csv, also on a STOP. Exit 2 = slices missing / inconsistent, 3 = STOP
# (a gate failed, or an existing parquet / rows_rev5.csv differs and is kept: MERGE_FORCE=1 via
# --export=ALL,MERGE_FORCE=1 replaces them). A rerun with unchanged slices rewrites nothing but the report.
#   sbatch --dependency=afterany:<ARRAY_JID> --partition=cpu1,cpu2 --cpus-per-task=1 --mem=8G --output=$R5_SCRATCH/logs/ext_merge.%j.out r5_ext_merge.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=${PY:-$R5_SCRATCH/venv/bin/python}
[ -x "$PY" ] || { echo "STOP: $PY missing (r5_probe.sh creates the rev 5 venv)"; exit 2; }
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$CODE/results_rev5" || exit 2
FORCE_ARG=""
[ "${MERGE_FORCE:-0}" = 1 ] && FORCE_ARG="--force"
ls "$R5_SCRATCH"/ext/slices/ 2>/dev/null | head -20
"$PY" r5_ext_merge.py $FORCE_ARG
rc=$?
echo "r5_ext_merge.py exit $rc (0 = B-7 gates passed and rows_rev5.csv written, 2 = inputs, 3 = STOP)"
ls -la results_rev5/B7_report.json results_rev5/B7_feature_quantiles.csv results_rev5/rows_rev5.csv "$ESPLEY_OUT/xtb_features_ext_g1.parquet" 2>&1
exit $rc
