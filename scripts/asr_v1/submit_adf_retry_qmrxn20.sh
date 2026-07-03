#!/bin/bash
# SLURM-array retry of recoverable qmrxn20 (E2 + SN2) failures.
#
# Reads:   outputs/asr_v1/retry/recoverable_dirs_qmrxn20.txt
#          (499 reactions: 249 e2 + 250 sn2)
# Targets: SLURM compute partitions cpu1/cpu2.
#
# Sizing (per dipolar retry's measured baseline):
#   - qmrxn20 wallclock ~5 min/reaction (vs dipolar 16-69 min)
#   - 10 tasks × 50 reactions = 500 slots (last task has 49)
#   - per-task wall ≈ 50 × 5 min = 4.2 h, --time=10h gives 2.4x buffer
#
# License: 2026-06-02 license with Linux* hostid pattern. Already installed
# and verified working on compute nodes.

#SBATCH --job-name=asr_v1_adf_retry_qmrxn20
#SBATCH --array=0-9
#SBATCH --partition=cpu1,cpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/asr_v1/logs/retry_qmrxn20-%A_%a.out
#SBATCH --error=outputs/asr_v1/logs/retry_qmrxn20-%A_%a.err

set -o pipefail
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
LIST="$REPO/outputs/asr_v1/retry/recoverable_dirs_qmrxn20.txt"
REACTIONS_PER_TASK=${REACTIONS_PER_TASK:-50}

mkdir -p "$REPO/outputs/asr_v1/logs"
mapfile -t DIRS < "$LIST"
N_TOTAL=${#DIRS[@]}

START_IDX=$(( SLURM_ARRAY_TASK_ID * REACTIONS_PER_TASK ))
END_IDX=$(( START_IDX + REACTIONS_PER_TASK - 1 ))
if [ "$END_IDX" -ge "$N_TOTAL" ]; then
    END_IDX=$(( N_TOTAL - 1 ))
fi
if [ "$START_IDX" -ge "$N_TOTAL" ]; then
    echo "[skip] task $SLURM_ARRAY_TASK_ID has no work ($START_IDX >= $N_TOTAL)"
    exit 0
fi

echo "============================================================"
echo "=== qmrxn20 retry array task $SLURM_ARRAY_TASK_ID :"
echo "===   processing reactions [$START_IDX..$END_IDX] of $N_TOTAL"
echo "===   host=$(hostname)  cpus=$SLURM_CPUS_PER_TASK"
echo "============================================================"

source /home1/yeseo1ee/ams2026.103/amsbashrc.sh
if [ -z "${SCMLICENSE:-}" ]; then
    echo "[fatal] SCMLICENSE unset"; exit 3
fi
ulimit -s unlimited 2>/dev/null || true

is_already_ok() {
    local status_json="$1"
    [ -f "$status_json" ] || return 1
    python3 - "$status_json" <<'PY' 2>/dev/null
import json, os, sys
status_p = sys.argv[1]
s = json.load(open(status_p))
rxn_dir = os.path.dirname(status_p)
ok_states = all(v in ("converged","n/a_single_atom") for v in s.get("calc_status",{}).values())
ok_exit = s.get("exit_code") == 0
ok_files = all(os.path.isfile(os.path.join(rxn_dir,f)) for f in s.get("output_files",{}).values())
sys.exit(0 if (ok_exit and ok_states and ok_files) else 1)
PY
}

process_one() {
    local RXN_DIR="$1"
    local RID
    RID="$(basename "$RXN_DIR")"
    if is_already_ok "$RXN_DIR/status.json"; then
        echo "[skip] $RID — already converged"
        return 0
    fi
    if [ ! -f "$RXN_DIR/run_reaction.sh" ]; then
        echo "[err] $RID — no run_reaction.sh"
        return 1
    fi
    echo
    echo "--- $(date -Iseconds) START $RID  (cwd→$RXN_DIR)"
    pushd "$RXN_DIR" >/dev/null
    bash run_reaction.sh
    local RC=$?
    popd >/dev/null
    echo "--- $(date -Iseconds) END   $RID  rc=$RC"
    return $RC
}

N_OK=0; N_FAIL=0
for IDX in $(seq "$START_IDX" "$END_IDX"); do
    RXN_DIR="${DIRS[$IDX]}"
    process_one "$RXN_DIR"
    case $? in
        0) N_OK=$((N_OK+1));;
        *) N_FAIL=$((N_FAIL+1));;
    esac
done

echo
echo "============================================================"
echo "=== qmrxn20 task $SLURM_ARRAY_TASK_ID COMPLETE"
echo "===   ok = $N_OK,  fail = $N_FAIL"
echo "============================================================"
exit 0
