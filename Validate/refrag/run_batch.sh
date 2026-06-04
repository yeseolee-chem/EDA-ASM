#!/bin/bash
# Run all refrag reactions in parallel via xargs.
# Usage: run_batch.sh [parallelism]   (default 4)
set -eo pipefail
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
PAR="${1:-4}"

cd "$ROOT"
mapfile -t RIDS < <(ls Validate/refrag/stage5a/per_reaction/)
echo "Reactions to run: ${#RIDS[@]} (parallelism=$PAR)"

# Skip ones that already produced a non-FAILED result
TODO=()
for rid in "${RIDS[@]}"; do
    out="Validate/refrag/results/${rid}.json"
    if [ -f "$out" ]; then
        status=$(python3 -c "import json,sys; d=json.load(open('$out')); print(d.get('status_at_queue','?'))" 2>/dev/null || echo "?")
        if [ "$status" != "FAILED" ] && [ "$status" != "?" ]; then
            echo "[SKIP] $rid ($status)"
            continue
        fi
    fi
    TODO+=("$rid")
done
echo "Will run: ${#TODO[@]}"

printf '%s\n' "${TODO[@]}" | xargs -P "$PAR" -I {} bash -c './Validate/refrag/run_one.sh "$1"' _ {}
echo "=== batch done ==="
