#!/bin/bash
# Run one candidate (synthetic rid) through ADF.
set -eo pipefail
RXN_ID="${1:?usage: run_candidate.sh <synthetic_rid>}"
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
LOGDIR="$ROOT/Validate/refrag/candidate_logs"
mkdir -p "$LOGDIR"

source "$HOME/ams2026.103/amsbashrc.sh"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

NSCM=1 "$AMSBIN/amspython" Validate/refrag/run_candidate_wrapper.py \
    --rxn_id "$RXN_ID" > "$LOGDIR/$RXN_ID.log" 2>&1
ec=$?
echo "[$RXN_ID] exit=$ec"
exit $ec
