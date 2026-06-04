#!/bin/bash
# Run all reactions in a single batch via xargs -P N (gate1, NSCM=1, nice 10).
# Idempotent: reactions whose status.json shows exit_code:0 are skipped.
# NO `set -e` — individual reaction failures must not abort the batch.
set -o pipefail
BATCH_DIR="${1:?usage: $0 <batch_dir> [parallelism=4]}"
PAR="${2:-4}"

run_one() {
    local rxn_dir="$1"
    local rid="$(basename "$rxn_dir")"
    local status="$rxn_dir/status.json"
    if [[ -f "$status" ]] && python3 -c "
import json, sys
try:
    s = json.load(open('$status'))
    sys.exit(0 if s.get('exit_code', 1) == 0 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        return  # already done
    fi
    nice -n 10 bash "$rxn_dir/run_reaction.sh" 2>/dev/null || true
}
export -f run_one

find "$BATCH_DIR" -mindepth 1 -maxdepth 1 -type d -print | \
    xargs -P "$PAR" -I {} bash -c 'run_one "$@"' _ {} || true

echo "=== run_batch done at $(date -Iseconds) ==="
