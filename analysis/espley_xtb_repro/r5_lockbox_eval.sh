#!/bin/bash
#SBATCH --job-name=r5_lockbox_eval
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=64G
# Phase D-1 (docs/specs/REV5_FEATURES_FIGURES.md): lockbox evaluation, final_eval.py, in ONE allocation.
# Train on results_rev5/dev_ids.csv, score on results_rev5/lockbox_ids.csv; arms BASE (= ESPLEY73), EXT_SEL
# (prereg_rev5b.json), EXT_ALL x KRR_rbf (headline) / Ridge / SVR_rbf / XGB x 9 targets; GridSearchCV on all dev rows
# (KFold(N_INNER, shuffle, SEED)) with SLURM_CPUS_PER_TASK workers and one BLAS thread each; reaction-level bootstrap
# (N_BOOT, one shared index matrix) CIs of MAE / NMAE and of the paired differences EXT_SEL - BASE, EXT_ALL - BASE.
#   -> $R5_SCRATCH/lockbox/units/ (one cache per arm x model x target; finished units are reused on resubmit)
#      $R5_SCRATCH/lockbox/lockbox_predictions.parquet
#      results_rev5/D1_lockbox.csv, D1_lockbox_paired.csv, D1_lockbox.json
# Refuses to run unless PREREG_REV5b.md, prereg_rev5b.json, rows_rev5.csv, dev_ids.csv and lockbox_ids.csv are
# committed and unchanged (final_eval.py repeats this gate and also requires the lockbox sha256 in a committed
# PREREG_REV5*.md / prereg_rev5*.json), and unless no Phase C cache under $R5_SCRATCH/select holds a lockbox id
# (first use of the lockbox; exit 3 on a leak).
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=48 --mem=64G \
#          --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/logs/lockbox_eval.%j.out r5_lockbox_eval.sh
#   gates only (no fit, no write): sbatch --partition=cpu1,cpu2 --cpus-per-task=4 --mem=8G \
#          --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/logs/lockbox_eval_gates.%j.out r5_lockbox_eval.sh --gates-only
#   (--output is the literal default R5_SCRATCH path: SLURM opens it before this script runs, so an unexported
#   $R5_SCRATCH in the submitting shell would give /logs/... and a job without a log; that logs dir must exist)
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
for f in results_rev5/PREREG_REV5b.md results_rev5/prereg_rev5b.json results_rev5/rows_rev5.csv \
         results_rev5/dev_ids.csv results_rev5/lockbox_ids.csv; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
        echo "PREREG gate: $f is not committed or differs from HEAD — commit PREREG_REV5b (C-3) first"; exit 2
    fi
done
# final_eval.py passes rows / features / arms explicitly; only the rev5_common paths come from the environment
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
unset ESPLEY_FEAT ESPLEY_ROWS ESPLEY_ARMS ESPLEY_EXT ESPLEY_GEOM ESPLEY_ML_OUT ESPLEY_TARGET_SET ESPLEY_PROTOCOLS
# one BLAS thread per GridSearchCV worker (unset, every worker spawns one thread per allocated core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export R5_NJOBS=${SLURM_CPUS_PER_TASK:-48}
mkdir -p "$R5_SCRATCH/lockbox" "$R5_SCRATCH/logs"
"$PY" -c "import sklearn, xgboost; print('py env ok: sklearn', sklearn.__version__, 'xgboost', xgboost.__version__)" || exit 2
echo "D-1: features $ESPLEY_OUT/xtb_features_g1.parquet + xtb_features_ext_g1.parquet, $R5_NJOBS GridSearchCV workers"
"$PY" final_eval.py "$@"
rc=$?
[ "$rc" -eq 0 ] || { echo "final_eval.py exit $rc"; exit "$rc"; }
ls -la results_rev5/D1_lockbox* 2>/dev/null
exit 0
