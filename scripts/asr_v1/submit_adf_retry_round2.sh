#!/bin/bash
# Round-2 dipolar retry — covers the 46 reactions that timed out in 662082.
#
# Reads: outputs/asr_v1/retry/recoverable_dirs_dipolar_round2.txt (46 dirs)
# Sizing: 2-element array × 23 reactions/task → ~19h per-task wall (worst case
#         using gate1 dipolar mean 69min), --time=24h budget.

#SBATCH --job-name=asr_v1_adf_retry_r2
#SBATCH --array=0-1
#SBATCH --partition=cpu1,cpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/asr_v1/logs/retry_r2-%A_%a.out
#SBATCH --error=outputs/asr_v1/logs/retry_r2-%A_%a.err

set -o pipefail
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
LIST="$REPO/outputs/asr_v1/retry/recoverable_dirs_dipolar_round2.txt"
REACTIONS_PER_TASK=${REACTIONS_PER_TASK:-23}

mkdir -p "$REPO/outputs/asr_v1/logs"
mapfile -t DIRS < "$LIST"
N_TOTAL=${#DIRS[@]}

START_IDX=$(( SLURM_ARRAY_TASK_ID * REACTIONS_PER_TASK ))
END_IDX=$(( START_IDX + REACTIONS_PER_TASK - 1 ))
[ "$END_IDX" -ge "$N_TOTAL" ] && END_IDX=$(( N_TOTAL - 1 ))
[ "$START_IDX" -ge "$N_TOTAL" ] && { echo "[skip] task has no work"; exit 0; }

echo "=== round-2 task $SLURM_ARRAY_TASK_ID : reactions [$START_IDX..$END_IDX] of $N_TOTAL ==="
echo "=== host=$(hostname) cpus=$SLURM_CPUS_PER_TASK ==="

source /home1/yeseo1ee/ams2026.103/amsbashrc.sh
[ -z "${SCMLICENSE:-}" ] && { echo "[fatal] SCMLICENSE unset"; exit 3; }
ulimit -s unlimited 2>/dev/null || true

is_already_ok() {
    [ -f "$1" ] || return 1
    python3 - "$1" <<'PY' 2>/dev/null
import json, os, sys
status_p = sys.argv[1]; s = json.load(open(status_p)); rd = os.path.dirname(status_p)
ok = (s.get('exit_code')==0
      and all(v in ('converged','n/a_single_atom') for v in s.get('calc_status',{}).values())
      and all(os.path.isfile(os.path.join(rd,f)) for f in s.get('output_files',{}).values()))
sys.exit(0 if ok else 1)
PY
}

N_OK=0; N_FAIL=0
for IDX in $(seq "$START_IDX" "$END_IDX"); do
    RXN_DIR="${DIRS[$IDX]}"; RID="$(basename "$RXN_DIR")"
    if is_already_ok "$RXN_DIR/status.json"; then echo "[skip] $RID — already converged"; continue; fi
    echo
    echo "--- $(date -Iseconds) START $RID"
    pushd "$RXN_DIR" >/dev/null; bash run_reaction.sh; RC=$?; popd >/dev/null
    echo "--- $(date -Iseconds) END   $RID rc=$RC"
    [ "$RC" -eq 0 ] && N_OK=$((N_OK+1)) || N_FAIL=$((N_FAIL+1))
done

echo "=== round-2 task $SLURM_ARRAY_TASK_ID COMPLETE  ok=$N_OK fail=$N_FAIL ==="
