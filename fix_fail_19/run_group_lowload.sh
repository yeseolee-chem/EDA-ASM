#!/bin/bash
# Low-load runner for Group A or B on gate1 (parallelism = 2).
# Usage: run_group_lowload.sh <A|B> [parallelism=2]
set -eo pipefail
GROUP="${1:?usage: $0 <A|B> [par]}"
PAR="${2:-2}"
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
LOGDIR="$ROOT/work_fix_fail_19/${GROUP}_logs"
mkdir -p "$LOGDIR"

source "$HOME/ams2026.103/amsbashrc.sh"
# amspython unsets PYTHONPATH on entry; use SCM_PYTHONPATH instead so the
# child Python can still import eda_asm (for frames_cache pickle) + fix_fail_19.
export SCM_PYTHONPATH="$ROOT/src:$ROOT"
cd "$ROOT"

QUEUE="$ROOT/work_fix_fail_19/queue_${GROUP}.json"
mapfile -t RIDS < <(python3 -c "
import json
for e in json.load(open('$QUEUE')):
    print(e['reaction_id'])
")
echo "Group $GROUP: ${#RIDS[@]} reactions, parallelism=$PAR"

run_one() {
    local rid="$1"
    NSCM=1 "$AMSBIN/amspython" -m fix_fail_19.run_under_amspython \
        --group "$GROUP" --rxn_id "$rid" \
        --halo8-dir "$ROOT/ADF_500/stage5a" \
        --out-dir "$ROOT/work_fix_fail_19" > "$LOGDIR/$rid.log" 2>&1
    echo "[$rid] done"
}
export -f run_one
export AMSBIN PYTHONPATH ROOT LOGDIR GROUP

printf '%s\n' "${RIDS[@]}" | xargs -P "$PAR" -I {} bash -c 'run_one "$@"' _ {}
echo "=== group $GROUP done ==="
