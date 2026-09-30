#!/bin/bash
#SBATCH --job-name=r5_ext
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G
# Phase B (docs/specs/REV5_FEATURES_FIGURES.md B-1..B-6): ext_features.py blocks B1..B6 for the G1-ok rxns in 18
# interleaved slices. Array element i = slices i and i+9, one after the other (--array=0-8: 9 queue slots, leaving room
# under the shared MaxSubmit=20); each slice runs EXT_WORKERS (default $SLURM_CPUS_PER_TASK) single-threaded worker
# processes (xtb 1 thread, tblite / BLAS 1 thread). Idempotent: an existing slice parquet is skipped, and every rxn's
# blocks are cached in $R5_SCRATCH/ext/<rid>/, so a task clipped by the 48 h wall is simply resubmitted. A slice with
# > 20 % failed rows (a systematic problem) exits 4 without writing its parquet; after the fix, resubmit with
# --export=ALL,EXT_RETRY_FAILED=1 so the cached failed blocks are recomputed (a fix that changes values of ok blocks
# must bump ext_features.CODE_VERSION instead).
# Refuses to start unless results_rev5/B0_smoke.json says passed (spec B-0 STOP rule); EXT_SKIP_B0_CHECK=1 overrides.
# Submit from this directory:
#   sbatch --array=0-8 --partition=cpu1,cpu2 --cpus-per-task=16 --mem=32G --output=$R5_SCRATCH/logs/ext.%A_%a.out r5_ext_array.sh
#   then: sbatch --dependency=afterany:<JID> ... r5_ext_merge.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=${PY:-$R5_SCRATCH/venv/bin/python}
[ -x "$PY" ] || { echo "STOP: $PY missing (r5_probe.sh creates the rev 5 venv)"; exit 2; }
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XTB_THREADS=1 OMP_STACKSIZE=1G PYTHONWARNINGS=ignore
N=18
TASK=${SLURM_ARRAY_TASK_ID:?submit as an array: --array=0-8}
case "$TASK" in [0-8]) ;; *) echo "array index must be 0-8 (element i = slices i and i+9), got $TASK"; exit 2 ;; esac
W=${EXT_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
RETRY_ARG=""
[ "${EXT_RETRY_FAILED:-0}" = 1 ] && RETRY_ARG="--retry-failed"
"$PY" -c "import tblite.interface, pyarrow, networkx, dftd3, morfeus; print('py env ok')" || { echo "STOP: venv imports"; exit 2; }
if [ "${EXT_SKIP_B0_CHECK:-0}" != 1 ]; then
    "$PY" -c "import json, sys; sys.exit(0 if json.load(open('results_rev5/B0_smoke.json')).get('passed') is True else 3)" \
        || { echo "STOP: results_rev5/B0_smoke.json missing or not passed (run r5_ext_smoke.sh first)"; exit 3; }
fi
"$XTB_BIN" --version 2>&1 | grep -i "xtb version" | head -1
rc=0
for s in "$TASK" $((TASK + 9)); do
    OUT=$R5_SCRATCH/ext/slices/slice_$(printf '%02d' "$s").parquet
    if [ -e "$OUT" ]; then echo "skip: $OUT exists"; continue; fi
    echo "=== slice $s/$N -> $OUT, $W workers $RETRY_ARG ($(date '+%F %T'))"
    "$PY" ext_features.py slice "$s" "$N" "$OUT" --workers "$W" $RETRY_ARG
    r=$?
    if [ "$r" -ne 0 ]; then echo "### slice $s exit $r"; rc=$r; fi
done
echo "done $(date '+%F %T') rc=$rc"
exit $rc
