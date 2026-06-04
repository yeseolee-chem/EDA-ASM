#!/bin/bash
# Submit every pending (no result yet) candidate to SLURM.
# Use this for any future retry / new candidate set instead of running on gate1.
#
# Usage:
#   ./Validate/refrag/submit_pending_to_slurm.sh [max_jobs]
# max_jobs defaults to 50 (caps how many we submit per invocation).

set -eo pipefail
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
MAX_JOBS="${1:-50}"

cd "$ROOT"
PENDING=()
for rid in $(ls Validate/refrag/candidates_stage5a/per_reaction/); do
    out="Validate/refrag/candidate_results/${rid}.json"
    if [ -f "$out" ]; then
        status=$(python3 -c "import json; print(json.load(open('$out')).get('status_at_queue','?'))" 2>/dev/null || echo "?")
        if [ "$status" != "FAILED" ] && [ "$status" != "?" ]; then
            continue
        fi
    fi
    PENDING+=("$rid")
    [ "${#PENDING[@]}" -ge "$MAX_JOBS" ] && break
done

echo "Pending: ${#PENDING[@]} (cap=$MAX_JOBS)"
for rid in "${PENDING[@]}"; do
    sbatch --export=RXN_ID="$rid" Validate/refrag/slurm_run_candidate.sbatch
done
