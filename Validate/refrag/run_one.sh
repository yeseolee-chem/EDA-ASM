#!/bin/bash
# Run one reaction through the refrag pipeline. Usage: run_one.sh <rxn_id>
set -eo pipefail
RXN_ID="${1:?usage: run_one.sh <rxn_id>}"
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
LOGDIR="$ROOT/Validate/refrag/logs"
mkdir -p "$LOGDIR"

# AMS environment (its bashrc references some unset vars — keep -u OFF here)
source "$HOME/ams2026.103/amsbashrc.sh"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"

cd "$ROOT"
NSCM=1 "$AMSBIN/amspython" Validate/refrag/run_wrapper.py --rxn_id "$RXN_ID" \
    > "$LOGDIR/$RXN_ID.log" 2>&1
ec=$?
echo "[$RXN_ID] exit=$ec"
exit $ec
