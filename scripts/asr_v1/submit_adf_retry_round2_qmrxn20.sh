#!/bin/bash
# Round-2 qmrxn20 retry — 238 reactions remaining (round-1 hit 10h walltime).
# Sizing: 5 tasks × 48 reactions/task, --time=20h (fast E2/SN2 systems).

#SBATCH --job-name=asr_v1_r2_qmrxn20
#SBATCH --array=0-4
#SBATCH --partition=cpu1,cpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=outputs/asr_v1/logs/r2_qmrxn20-%A_%a.out
#SBATCH --error=outputs/asr_v1/logs/r2_qmrxn20-%A_%a.err

set -o pipefail
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
LIST="$REPO/outputs/asr_v1/retry/recoverable_dirs_qmrxn20_round2.txt"
RPT=${REACTIONS_PER_TASK:-48}

mkdir -p "$REPO/outputs/asr_v1/logs"
mapfile -t DIRS < "$LIST"
N=${#DIRS[@]}
S=$(( SLURM_ARRAY_TASK_ID * RPT ))
E=$(( S + RPT - 1 )); [ "$E" -ge "$N" ] && E=$(( N - 1 ))
[ "$S" -ge "$N" ] && { echo "[skip] no work"; exit 0; }

echo "=== r2_qmrxn20 task $SLURM_ARRAY_TASK_ID : reactions [$S..$E] of $N  host=$(hostname) ==="
source /home1/yeseo1ee/ams2026.103/amsbashrc.sh
[ -z "${SCMLICENSE:-}" ] && { echo "[fatal] SCMLICENSE unset"; exit 3; }
ulimit -s unlimited 2>/dev/null || true

is_ok() {
    [ -f "$1" ] || return 1
    python3 - "$1" <<'PY' 2>/dev/null
import json, os, sys
s=json.load(open(sys.argv[1])); rd=os.path.dirname(sys.argv[1])
ok=(s.get('exit_code')==0 and all(v in ('converged','n/a_single_atom') for v in s.get('calc_status',{}).values()) and all(os.path.isfile(os.path.join(rd,f)) for f in s.get('output_files',{}).values()))
sys.exit(0 if ok else 1)
PY
}

N_OK=0; N_F=0
for IDX in $(seq "$S" "$E"); do
    RXN_DIR="${DIRS[$IDX]}"; RID="$(basename "$RXN_DIR")"
    if is_ok "$RXN_DIR/status.json"; then echo "[skip] $RID"; continue; fi
    echo
    echo "--- $(date -Iseconds) START $RID"
    pushd "$RXN_DIR" >/dev/null; bash run_reaction.sh; RC=$?; popd >/dev/null
    echo "--- $(date -Iseconds) END   $RID rc=$RC"
    [ "$RC" -eq 0 ] && N_OK=$((N_OK+1)) || N_F=$((N_F+1))
done
echo "=== r2_qmrxn20 task $SLURM_ARRAY_TASK_ID COMPLETE  ok=$N_OK fail=$N_F ==="
