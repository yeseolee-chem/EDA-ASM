#!/bin/bash
# Low-load runner: process the 17 paper-inspired candidates on gate1 with
# parallelism=2 to stay light on shared resources.
set -eo pipefail
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
PAR="${1:-2}"

cd "$ROOT"

# Build pending list (only __h* and __p* synth_rids without a non-FAILED result)
TODO=()
for srid in $(ls Validate/refrag/candidates_stage5a/per_reaction/); do
    case "$srid" in
        *__h[0-9]*|*__p[0-9]*) ;;
        *) continue;;
    esac
    out="Validate/refrag/candidate_results/${srid}.json"
    if [ -f "$out" ]; then
        status=$(python3 -c "import json; print(json.load(open('$out')).get('status_at_queue','?'))" 2>/dev/null || echo "?")
        if [ "$status" != "FAILED" ] && [ "$status" != "?" ]; then
            continue
        fi
    fi
    TODO+=("$srid")
done

echo "Will run: ${#TODO[@]} (parallelism=$PAR, on gate1, low-load)"
echo "================================================================"
printf '%s\n' "${TODO[@]}" | xargs -P "$PAR" -I {} bash -c './Validate/refrag/run_candidate.sh "$1"' _ {}
echo "=== low-load batch done ==="
