#!/bin/bash
# Launch all candidates in parallel. Default parallelism 10.
set -eo pipefail
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
PAR="${1:-10}"

cd "$ROOT"
mapfile -t SYN_RIDS < <(ls "$ROOT/Validate/refrag/candidates_stage5a/per_reaction/")
echo "Total synthetic rids: ${#SYN_RIDS[@]} (parallelism=$PAR)"

# Skip ones that already produced a non-FAILED candidate result
TODO=()
for srid in "${SYN_RIDS[@]}"; do
    out="$ROOT/Validate/refrag/candidate_results/${srid}.json"
    if [ -f "$out" ]; then
        status=$(python3 -c "import json; print(json.load(open('$out')).get('status_at_queue','?'))" 2>/dev/null || echo "?")
        if [ "$status" != "FAILED" ] && [ "$status" != "?" ]; then
            continue
        fi
    fi
    TODO+=("$srid")
done
echo "Will run: ${#TODO[@]}"
echo "================================================================"

printf '%s\n' "${TODO[@]}" | xargs -P "$PAR" -I {} bash -c './Validate/refrag/run_candidate.sh "$1"' _ {}
echo "=== candidate batch done ==="
