#!/bin/bash
#SBATCH --job-name=r5_espley
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=64G
# Phase D-3(a), D-3(b), D-4 (docs/specs/REV5_FEATURES_FIGURES.md): comparison with Espley et al. 2024 on their ds3
# rows, espley_rev5.py, in ONE allocation. Sub-command (first argument, default all):
#   d3a  Espley split, seeds 22/23/14/1/2, rows = labelled ∩ rows_rev5: ours ESPLEY46 / ESPLEY73 / EXT_SEL x KRR vs
#        Espley stored predictions, their-protocol role SVR / KRR (rev 4 files reused when they cover the rows) and
#        Espley 46 features x our pipeline (Ridge / KRR / SVR / XGB); Nadeau-Bengio tests vs Espley SVR and vs the
#        strongest Espley-side model          -> results_rev5/D3a_*.csv, D3a.json
#   d3b  lockbox head-to-head: train = dev ∩ Espley rows, test = lockbox ∩ Espley rows; Espley SVR / KRR with their
#        protocol vs EXT_SEL KRR (nested); paired reaction-level bootstrap -> results_rev5/D3b_*.csv, D3b.json
#   d4   learning curves (needs d3a): Espley SVR vs EXT_SEL KRR on identical subsets, fixed hyperparameters
#                                             -> results_rev5/D4_learning_curves*.csv / .json
#   all  d3a, d3b, d4 in this order
# GridSearchCV runs one unit at a time with SLURM_CPUS_PER_TASK workers, the D-4 fits in a spawn process pool of that
# size; one BLAS thread per process. Units are cached in $R5_SCRATCH/espley/{d3a,d3b,d4}/units (a finished unit is
# reused, one made from other inputs stops the run with exit 3), so a run clipped by the 48 h wall is resubmitted
# unchanged. Refuses to run unless PREREG_REV5b.md, prereg_rev5b.json, rows_rev5.csv, dev_ids.csv and lockbox_ids.csv
# are committed and unchanged (espley_rev5.py repeats this gate via final_eval.gate_prereg), then, for every stage (all
# use lockbox rows), checks the dev / lockbox partition and audits the Phase C caches in $R5_SCRATCH/select for lockbox
# ids (final_eval.audit_select, as D-1). Exit codes: 1 a fit / check failed, 2 bad input or gate, 3 a unit cache made
# from other inputs (move $R5_SCRATCH/espley/<stage>/units away), 4 LOCKBOX LEAK in the Phase C caches.
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=48 --mem=64G --output=$R5_SCRATCH/logs/espley.%j.out r5_espley.sh all
#   (a second argument --gates-only runs the gates and the data load without fitting or writing)
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
STAGE=${1:-all}
case $STAGE in
    d3a|d3b|d4|all) ;;
    *) echo "usage: sbatch ... r5_espley.sh d3a|d3b|d4|all [--gates-only]; got '$STAGE'"; exit 2 ;;
esac
shift || true
for f in results_rev5/PREREG_REV5b.md results_rev5/prereg_rev5b.json results_rev5/rows_rev5.csv \
         results_rev5/dev_ids.csv results_rev5/lockbox_ids.csv; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
        echo "PREREG gate: $f is not committed or differs from HEAD — commit PREREG_REV5b (C-3) first"; exit 2
    fi
done
# espley_rev5.py passes rows / features / arms explicitly; only the paths come from the environment
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
unset ESPLEY_FEAT ESPLEY_ROWS ESPLEY_ARMS ESPLEY_EXT ESPLEY_GEOM ESPLEY_ML_OUT ESPLEY_TARGET_SET ESPLEY_PROTOCOLS
# one BLAS thread per worker (unset, every GridSearchCV / pool worker spawns one thread per allocated core)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export R5_NJOBS=${SLURM_CPUS_PER_TASK:-48}
export ESPLEY_NJOBS=$R5_NJOBS
# spawned pool workers import espley_rev5 / compare_espley / train_ml_single from the code directory
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$R5_SCRATCH/espley" "$R5_SCRATCH/logs"
"$PY" -c "import sklearn, xgboost, scipy; print('py env ok: sklearn', sklearn.__version__, 'xgboost', xgboost.__version__)" \
    || exit 2
echo "espley_rev5 $STAGE $*: $R5_NJOBS workers, R4_SCRATCH=$R4_SCRATCH (rev 4 Espley side), ESPLEY_REPO_DATA=$ESPLEY_REPO_DATA"
"$PY" espley_rev5.py "$STAGE" "$@"
rc=$?
[ "$rc" -eq 0 ] || { echo "espley_rev5.py $STAGE exit $rc"; exit "$rc"; }
ls -la results_rev5/D3a* results_rev5/D3b* results_rev5/D4_* 2>/dev/null
exit 0
