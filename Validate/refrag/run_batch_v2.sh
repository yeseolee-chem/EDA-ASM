#!/bin/bash
# Launch v2 alt-frag batch: skip any rid with an active run_one.sh, skip ones
# that already have a non-FAILED result. Default parallelism 6.
set -eo pipefail
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
PAR="${1:-6}"

cd "$ROOT"

# Active rids (currently being processed by an ongoing run_one.sh)
mapfile -t ACTIVE < <(ps -ef | awk '/run_one\.sh/ && !/awk/ {print $NF}' | grep -E '^(Halogen_|T1x_)' | sort -u)
echo "Active reruns to skip (${#ACTIVE[@]}): ${ACTIVE[*]:-none}"

mapfile -t RIDS < <(ls Validate/refrag/stage5a/per_reaction/)
echo "Alt stage5a entries: ${#RIDS[@]}"

# Build TODO list: skip active, skip completed-non-FAILED
TODO=()
for rid in "${RIDS[@]}"; do
    skip=""
    for a in "${ACTIVE[@]}"; do
        [ "$rid" = "$a" ] && skip="active" && break
    done
    if [ -z "$skip" ]; then
        out="Validate/refrag/results/${rid}.json"
        if [ -f "$out" ]; then
            status=$(python3 -c "import json; print(json.load(open('$out')).get('status_at_queue','?'))" 2>/dev/null || echo "?")
            if [ "$status" != "FAILED" ] && [ "$status" != "?" ]; then
                skip="done($status)"
            fi
        fi
    fi
    if [ -n "$skip" ]; then
        echo "[SKIP] $rid  ($skip)"
    else
        TODO+=("$rid")
    fi
done

echo
echo "Will run: ${#TODO[@]} (parallelism=$PAR)"
echo "================================================================"

printf '%s\n' "${TODO[@]}" | xargs -P "$PAR" -I {} bash -c './Validate/refrag/run_one.sh "$1"' _ {}
echo "=== batch_v2 done ==="
