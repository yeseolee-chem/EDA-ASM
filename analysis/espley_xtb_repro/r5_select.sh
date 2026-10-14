#!/bin/bash
#SBATCH --job-name=r5_select
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=64G
# Phase C-2 / C-3 (docs/specs/REV5_FEATURES_FIGURES.md): dev-CV block evaluation and forward block selection
# (select_blocks.py select) in ONE allocation, dev rows only (results_rev5/dev_ids.csv; no lockbox row is loaded).
# (arm, target, outer fold) units run R5_SEL_PAR at a time in a spawn ProcessPoolExecutor (default
# SLURM_CPUS_PER_TASK / R5_SEL_NJOBS), each a KRR(RBF) nested GridSearchCV with R5_SEL_NJOBS (default 1) workers and
# one BLAS thread. Units are cached in $R5_SCRATCH/select/units/, so a run clipped by the 48 h wall is simply
# resubmitted. Outputs: results_rev5/C2_block_cv.csv, C2_block_cv_folds.csv, C2_selection_path.{csv,json},
# C2_feature_target_r.csv and prereg_rev5b.json (C-3; written last, once the selection path is complete; exit 3, with
# nothing written, if an existing prereg_rev5b.json differs). Ends with select_blocks.py status.
# R5_SEL_MODE (via --export=ALL,R5_SEL_MODE=...): select (default) | check (validate inputs and cached units, fit
# nothing) | status (progress only). Mode select refuses to run (exit 2) unless results_rev5/PREREG_REV5a.md,
# lockbox_ids.csv, dev_ids.csv and lockbox.json are committed (in HEAD) and unchanged, and PREREG_REV5a.md contains
# the sha256 of lockbox_ids.csv (spec C-1: pre-registration, lockbox ids + sha256, before any selection).
# 48 cpus fit a cpu1 node (48 cores) as well as cpu2; ~0.5 GB per unit process.
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=48 --mem=64G --output=$R5_SCRATCH/logs/select.%j.out r5_select.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
MODE=${R5_SEL_MODE:-select}
case "$MODE" in
    select|check|status) ;;
    *) echo "R5_SEL_MODE must be select, check or status, got '$MODE'"; exit 2 ;;
esac
if [ "$MODE" = select ]; then
    for f in results_rev5/PREREG_REV5a.md results_rev5/lockbox_ids.csv results_rev5/dev_ids.csv \
             results_rev5/lockbox.json; do
        if [ ! -f "$f" ] || ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 \
           || ! git cat-file -e "HEAD:./$f" 2>/dev/null || ! git diff --quiet HEAD -- "$f"; then
            echo "PREREG gate: $f is missing, not committed or differs from HEAD — commit PREREG_REV5a (C-1) first"
            exit 2
        fi
    done
    # spec C-1: PREREG_REV5a records lockbox_ids.csv AND its sha256 (final_eval.py re-checks it before D-1)
    lock_sha=$(sha256sum results_rev5/lockbox_ids.csv | cut -d' ' -f1)
    [ ${#lock_sha} -eq 64 ] || { echo "PREREG gate: sha256sum of results_rev5/lockbox_ids.csv failed"; exit 2; }
    if ! grep -qiF -- "$lock_sha" results_rev5/PREREG_REV5a.md; then
        echo "PREREG gate: the lockbox_ids.csv sha256 $lock_sha is not written in results_rev5/PREREG_REV5a.md (spec C-1)"
        exit 2
    fi
    echo "PREREG gate ok: PREREG_REV5a.md, lockbox_ids.csv, dev_ids.csv, lockbox.json committed; lockbox sha256 ${lock_sha:0:12} recorded"
fi
# feature parquets come from ESPLEY_OUT (rev5_common.FEAT_G1 / FEAT_EXT); nothing else is read from a submit shell
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
# one BLAS thread per process (unset, every unit process spawns one thread per allocated core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
export R5_SEL_NJOBS=${R5_SEL_NJOBS:-1}
mkdir -p "$R5_SCRATCH/logs" "$R5_SCRATCH/select" || exit 2
"$PY" -c "import sklearn, numpy, pandas; print('py env ok: sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)" || exit 2
echo "mode $MODE, cpus ${SLURM_CPUS_PER_TASK:-?}, R5_SEL_NJOBS $R5_SEL_NJOBS, R5_SEL_PAR ${R5_SEL_PAR:-auto}, ESPLEY_OUT $ESPLEY_OUT, R5_SCRATCH $R5_SCRATCH"
rc=0
case "$MODE" in
    select) "$PY" select_blocks.py select || rc=$? ;;
    check)  "$PY" select_blocks.py select --check || rc=$? ;;
esac
echo
"$PY" select_blocks.py status
if [ "$rc" -eq 0 ] && [ "$MODE" = select ]; then
    ls -la results_rev5/C2_* results_rev5/prereg_rev5b.json
    cat results_rev5/prereg_rev5b.json; echo
fi
[ "$rc" -eq 0 ] || echo "select_blocks.py exit $rc (1 = a unit failed, 2 = input, 3 = an existing file differs)"
exit $rc
