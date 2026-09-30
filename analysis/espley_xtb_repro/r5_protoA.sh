#!/bin/bash
#SBATCH --job-name=r5_protoA
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=64G
# Phase D-2 (docs/specs/REV5_FEATURES_FIGURES.md): Protocol A as rev 4 (80/10/10, seeds 22/23/14/1/2, nested per-seed
# GridSearchCV 5-fold; train_ml_single.protocol_A unchanged) on all of results_rev5/rows_rev5.csv, 9 targets x arms
# ESPLEY46, ESPLEY73, EXT_SEL, EXT_ALL x Ridge / KRR_rbf / SVR_rbf / XGB, in ONE allocation: the 9 targets run
# R5_PROTOA_PAR (default 3) at a time (xargs -P), each GridSearchCV with SLURM_CPUS_PER_TASK / R5_PROTOA_PAR workers
# and one BLAS thread per worker -> $R5_SCRATCH/protoA/ml_targets/. A finished target is skipped (and one made from
# other inputs stops it), so a run clipped by the 48 h wall is simply resubmitted.
# Then aggregate_ml.py -> $R5_SCRATCH/protoA/{ml_report.json, ml_table_espley.csv, predictions.parquet} and
# results_rev5/D2_protocolA_{table.csv, headline.csv, headline.md, meta.json} (EXT_SEL carries the note that it was
# selected on dev rows overlapping these test rows; the unbiased numbers are D-1), and plot_results.py ->
# $R5_SCRATCH/protoA/figures/, of which the arm comparison (png / pdf / csv), the per-arm KRR_rbf scatter and the per-arm
# model bars are copied to results_rev5/D2_protocolA_*.
# Refuses to run unless results_rev5/PREREG_REV5b.md, prereg_rev5b.json and rows_rev5.csv are committed and unchanged.
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=48 --mem=64G --output=$R5_SCRATCH/logs/protoA.%j.out r5_protoA.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
for f in results_rev5/PREREG_REV5b.md results_rev5/prereg_rev5b.json results_rev5/rows_rev5.csv; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
        echo "PREREG gate: $f is not committed or differs from HEAD — commit PREREG_REV5b (C-3) first"; exit 2
    fi
done
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
# set here only, so a stale ESPLEY_* in the submit shell cannot change arms, rows or inputs
export ESPLEY_OUT=$ROOT ESPLEY_GEOM=g1 ESPLEY_TARGET_SET=rev4 ESPLEY_PROTOCOLS=A ESPLEY_EXT=1
export ESPLEY_ARMS=ESPLEY46,ESPLEY73,EXT_SEL,EXT_ALL
export ESPLEY_FEAT=$ROOT/xtb_features_g1.parquet ESPLEY_ML_OUT=$R5_SCRATCH/protoA
export ESPLEY_ROWS=$CODE/results_rev5/rows_rev5.csv
unset ESPLEY_PLOT_ARMS ESPLEY_PLOT_FS ESPLEY_PLOT_MODEL
# one BLAS thread per GridSearchCV worker (unset, every worker spawns one thread per allocated core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NCPU=${SLURM_CPUS_PER_TASK:-48}
PAR=${R5_PROTOA_PAR:-3}
export ESPLEY_NJOBS=$(( NCPU / PAR ))
[ "$ESPLEY_NJOBS" -ge 1 ] || { echo "SLURM_CPUS_PER_TASK=$NCPU < R5_PROTOA_PAR=$PAR"; exit 2; }
export PY LOGS=$R5_SCRATCH/logs JOB=${SLURM_JOB_ID:-local}
mkdir -p "$LOGS" "$ESPLEY_ML_OUT"
"$PY" -c "import sklearn, xgboost; print('py env ok: sklearn', sklearn.__version__, 'xgboost', xgboost.__version__)" || exit 2
echo "D-2: 9 targets x arms $ESPLEY_ARMS, $PAR targets at a time x $ESPLEY_NJOBS workers"
echo "     rows $ESPLEY_ROWS, features $ESPLEY_FEAT (+ blocks $ROOT/xtb_features_ext_g1.parquet) -> $ESPLEY_ML_OUT"

run_target() {
    local log=$LOGS/protoA_t$1.$JOB.log
    "$PY" train_ml_single.py "$1" > "$log" 2>&1
    local rc=$?
    echo "target $1 exit $rc ($log)"
    return $rc
}
export -f run_target
seq 0 8 | xargs -P "$PAR" -I{} bash -c 'run_target {}'
rc=$?
n=$(ls "$ESPLEY_ML_OUT"/ml_targets/target_*.json 2>/dev/null | wc -l)
if [ "$rc" -ne 0 ] || [ "$n" -ne 9 ]; then
    echo "training incomplete: xargs exit $rc, $n / 9 target JSONs"
    tail -n 5 "$LOGS"/protoA_t*."$JOB".log
    exit 1
fi

export ESPLEY_REPORT_PREFIX=$CODE/results_rev5/D2_protocolA_ ESPLEY_REPORT_TITLE="rev 5 D-2 Protocol A"
mkdir -p results_rev5
"$PY" aggregate_ml.py || exit $?
"$PY" plot_results.py || exit $?
FIG=$ESPLEY_ML_OUT/figures
IFS=, read -ra ARMS <<< "$ESPLEY_ARMS"
src=("$FIG"/mae_bar_arms_KRR_rbf_g1.png "$FIG"/mae_bar_arms_KRR_rbf_g1.pdf "$FIG"/mae_bar_arms_KRR_rbf_g1.csv)
for a in "${ARMS[@]}"; do
    src+=("$FIG/scatter_KRR_rbf_${a}_g1.png" "$FIG/mae_bar_${a,,}_g1.png")
done
for f in "${src[@]}"; do
    [ -f "$f" ] || { echo "figure missing: $f"; exit 1; }
    dst=results_rev5/D2_protocolA_$(basename "$f")
    cp -f "$f" "$dst.tmp" && mv -f "$dst.tmp" "$dst" || exit 1
done
ls -la results_rev5/D2_protocolA_*
