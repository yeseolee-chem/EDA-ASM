#!/bin/bash
#SBATCH --job-name=r5_phaseA
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
# REV5 Phase A (docs/specs/REV5_FEATURES_FIGURES.md A-1 .. A-4) in one allocation; submit from this directory:
#   1  r5_phase_a.py delete: record every scratch file listed in r5_phaseA_discard.txt (path, size, sha256)
#      -> results_rev5/A_deleted.txt, then delete them (a rerun re-verifies the rest against that record): the
#      'delete' paths whole, and in the kept rev 4 ablation directories only the JSONs of the dropped arms
#      (B, B_role, O; selected by their our_geom field)
#   2  make_rows_rev4.py --check-g1-only -> results_rev5/A_rows_check.json (exit 3 = rows differ from rows_rev4.csv)
#   3  regenerate results_rev4/ from G1 + Espley only: rev4_tables.py ($ESPLEY_OUT/rev4/g1), compare_espley.py plot,
#      espley_fairness_ablation.py run for ABL_ROWS=common and own at once (arms A, C, C0, B_g1, O_g1; KRR + SVR;
#      $SLURM_CPUS_PER_TASK/2 GridSearchCV workers each; JSONs in $R5_SCRATCH/phaseA/ablation[_ownrows]/, existing
#      ones skipped) + aggregate, then r5_phase_a.py regen-check (every regenerated row vs git 740ba895)
#      -> results_rev5/A_regen_check.json
#   4  r5_phase_a.py grep-check (spec A-4) -> results_rev5/A_grep_check.txt; exit 3 = a hit other than PREREG_REV4.md
#      and the discard sentence, or a file listed as 'moved' out of this directory that is not an unedited move
#      (each moved file is disclosed as a '# note:' header line). results_rev4/SUMMARY.md must already be rewritten,
#      so step 4 is NOT in the default and is submitted on its own (PHASEA_STEPS=4) after that rewrite.
# PHASEA_STEPS picks steps (default 123; PHASEA_STEPS=4 after SUMMARY.md is rewritten). The first failing step ends
# the job with its exit code; the last line PHASE_A_OK marks success. Every step is idempotent: 1 re-verifies what is
# left against A_deleted.txt and deletes nothing twice, 2 / 4 are cheap re-checks, 3 rewrites the tables atomically
# and skips every ablation JSON that already exists.
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=8 --mem=16G --output=$R5_SCRATCH/logs/phaseA.%j.out r5_phaseA.sh
#   (after SUMMARY.md is rewritten)
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=1 --mem=4G --export=ALL,PHASEA_STEPS=4 \
#          --output=$R5_SCRATCH/logs/phaseA_grep.%j.out r5_phaseA.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=${PY:-$R5_SCRATCH/venv/bin/python}
[ -x "$PY" ] || { echo "STOP: $PY missing (r5_probe.sh creates the rev 5 venv)"; exit 2; }
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
# one BLAS thread per worker; the scripts derive their inputs from the defaults, not from a stale submit shell
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
unset ESPLEY_GEOM ESPLEY_FEAT ESPLEY_ML_OUT ESPLEY_ROWS ESPLEY_FEAT_G1 ABL_ROWS ABL_SCRATCH
NCPU=${SLURM_CPUS_PER_TASK:-8}
HALF=$(( NCPU / 2 )); [ "$HALF" -ge 1 ] || HALF=1
STEPS=${PHASEA_STEPS:-123}                     # single-digit step ids, any separator ("123", "1 2 3", "4")
case "$STEPS" in *[!1-4\ ]*|"") echo "PHASEA_STEPS must list steps 1-4 (e.g. 123 or 4), got '$STEPS'"; exit 2 ;; esac
JOB=${SLURM_JOB_ID:-local}
LOGS=$R5_SCRATCH/logs
mkdir -p "$LOGS" "$CODE/results_rev5" || exit 2
has() { case "$STEPS" in *"$1"*) return 0 ;; *) return 1 ;; esac; }
step() { echo; echo "=== step $1: $2   ($(date '+%F %T'))"; }
"$PY" -c "import sklearn, pandas, pyarrow, matplotlib; print('py env ok:', '$PY', 'sklearn', sklearn.__version__)" || exit 2
echo "steps [$STEPS] cpus $NCPU code $CODE R4_SCRATCH $R4_SCRATCH R5_SCRATCH $R5_SCRATCH ESPLEY_OUT $ESPLEY_OUT"

if has 1; then
    step 1 "record and delete the discarded scratch files (A-3)"
    "$PY" r5_phase_a.py delete || exit $?
fi

if has 2; then
    step 2 "rows rebuilt from G1 alone == rows_rev4.csv (A-4)"
    "$PY" make_rows_rev4.py --check-g1-only || exit $?
fi

if has 3; then
    step 3 "regenerate results_rev4 (A-1)"
    "$PY" rev4_tables.py --g1 "$ESPLEY_OUT/rev4/g1" || exit $?
    "$PY" compare_espley.py plot || exit $?
    pids=()
    for m in common own; do
        log=$LOGS/phaseA_ablation_$m.$JOB.out
        echo "ablation ABL_ROWS=$m, $HALF workers -> $log"
        ABL_ROWS=$m ESPLEY_NJOBS=$HALF "$PY" espley_fairness_ablation.py run > "$log" 2>&1 &
        pids+=("$!")
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    for m in common own; do echo "--- tail ablation $m"; tail -3 "$LOGS/phaseA_ablation_$m.$JOB.out"; done
    [ "$rc" -eq 0 ] || { echo "### ablation run failed (logs $LOGS/phaseA_ablation_*.$JOB.out)"; exit 1; }
    for m in common own; do ABL_ROWS=$m "$PY" espley_fairness_ablation.py aggregate || exit $?; done
    "$PY" r5_phase_a.py regen-check || exit $?
fi

if has 4; then
    step 4 "grep check (A-4)"
    "$PY" r5_phase_a.py grep-check || exit $?
fi

echo
echo "PHASE_A_OK steps [$STEPS] $(date '+%F %T')"
