#!/bin/bash
# worker.sh — validation worker pool (VALIDATION_SPEC §6.2). One array element = one worker.
#
#   sbatch -p <idle> --array=0-9%10 -o scratch/logs/w_%A_%a.log --export=ALL,VAL_DIR=$PWD worker.sh
#
# Each worker walks tasks.csv in priority order, claims the first task that is neither done, failed,
# tried by this job, nor held by a live claim, runs it with run_task.sh, and rescans.
#   claim / heartbeat / takeover : claim_is_stale, try_claim, heartbeat, mine are taken verbatim from
#       label_true/scripts/smd_relabel_worker.sh v5 (verified on this cluster); only the paths differ.
#       Not taken over: successor chaining and bad-node lineage (§6.2: resubmit worker.sh by hand).
#   $Q/done/<id>    run_task.sh exit 0 (finished, including a scientific failure recorded as .fail_*)
#   $Q/failed/<id>  any other exit (infrastructure); never retried automatically
#   $Q/STOP         checked before every claim; also written by the interim check (A3/A6/A7 FAIL)
#   interim         once every priority 0-2 task is done or failed, one worker runs
#                   analysis/validation_report.py --interim (interim_report.md, may write $Q/STOP)
#SBATCH --job-name=d1v_wrk
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
set -uo pipefail
VAL_DIR=${VAL_DIR:?VAL_DIR must point at D1_build/validation}
export VAL_DIR
export PILOT_DIR=${PILOT_DIR:-$(cd "$VAL_DIR/../d1_autode_pilot" && pwd)}
export D1P_CONFIG=$VAL_DIR/config_val.yaml
export D1P_MANIFEST=$VAL_DIR/val_manifest.csv
source "$PILOT_DIR/pilot_env.sh"
export Q=$SCRATCH/queue
export VAL_QUEUE=$Q                               # orca_direct.stop() writes $Q/STOP on a §6.3 mismatch
mkdir -p "$Q/claims" "$Q/done" "$Q/failed" "$Q/log"

IDX=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID:-manual}
ME=${USER:-$(id -un)}
NODE=$(hostname -s)
JOB_START=$(date +%s)
WALL_S=$((48*3600))
SAFETY_H=$(awk '/^  worker_safety_h:/{print $2}' "$D1P_CONFIG"); SAFETY_S=$(( ${SAFETY_H:-8} * 3600 ))
HEARTBEAT_S=${VAL_HEARTBEAT_S:-120}   # owner file refresh interval while a task runs
STALE_AGE_S=900                       # owner file older than this + owner job gone = stale
EMPTY_OWNER_AGE_S=$((6*3600))         # a claim with no readable owner is stale only after this
MARKER_AGE_S=7200                     # a takeover marker this old was left by a dead takeover

# ---------------------------------------------------------------- from smd_relabel_worker.sh v5
# $1 = claim dir, $2 = owner job id read from it.
# Stale when the owner file is old enough and the owner job is not in squeue at
# all (no -t filter: RUNNING, COMPLETING, CONFIGURING, SUSPENDED, ... are all
# alive). squeue must list this very job; if it does not, squeue is not trusted.
claim_is_stale() {
    local c=$1 own=$2 ref age alive
    ref="$c/owner"; [ -f "$ref" ] || ref="$c"
    age=$(( $(date +%s) - $(stat -c %Y "$ref" 2>/dev/null || date +%s) ))
    [ "$age" -ge "$STALE_AGE_S" ] || return 1
    [ -n "$own" ] || [ "$age" -ge "$EMPTY_OWNER_AGE_S" ] || return 1
    alive=$(squeue -h -u "$ME" -o "%A" 2>/dev/null) || return 1
    echo "$alive" | grep -qx "$JOB_ID" || return 1
    if [ -n "$own" ] && echo "$alive" | grep -qx "$own"; then return 1; fi
    return 0
}

# mkdir wins a free claim. A stale claim is taken by whoever first creates the
# takeover marker for that exact (task, owner, claim mtime); nobody else can then
# touch that claim, so a fresh claim made after it is never removed by mistake.
try_claim() {
    local c=$1 own mt m
    mkdir "$c" 2>/dev/null && return 0
    [ -d "$c" ] || return 1
    own=$(awk '{print $1}' "$c/owner" 2>/dev/null)
    mt=$(stat -c %Y "$c" 2>/dev/null) || return 1
    claim_is_stale "$c" "$own" || return 1
    m="$c.takeover.${own:-none}.$mt"
    if ! mkdir "$m" 2>/dev/null; then
        # left by a takeover that died half way: clear it so a later scan retries
        if [ $(( $(date +%s) - $(stat -c %Y "$m" 2>/dev/null || date +%s) )) -ge $MARKER_AGE_S ]; then
            rmdir "$m" 2>/dev/null
        fi
        return 1
    fi
    # the claim must still be the one judged stale
    if [ "$(awk '{print $1}' "$c/owner" 2>/dev/null)" != "$own" ] || \
       [ "$(stat -c %Y "$c" 2>/dev/null)" != "$mt" ]; then
        rmdir "$m" 2>/dev/null; return 1
    fi
    echo "  taking over stale claim $(basename "$c") (owner ${own:-none})"
    if ! rm -rf "$c" || ! mkdir "$c" 2>/dev/null; then
        rmdir "$m" 2>/dev/null; return 1
    fi
}

mine() { [ "$(awk '{print $1}' "$CLAIM_DIR/owner" 2>/dev/null)" = "$JOB_ID" ]; }

heartbeat() {   # $1 = claim dir; keeps its owner file fresh while the task runs
    local sp=
    trap '[ -n "$sp" ] && kill $sp 2>/dev/null; exit 0' TERM
    while :; do
        sleep "$HEARTBEAT_S" & sp=$!
        wait $sp
        touch -c "$1/owner" 2>/dev/null
    done
}
# ---------------------------------------------------------------- end of v5 functions

maybe_interim() {
    [ -e "$Q/interim.done" ] && return
    local t left=0
    while read -r t; do
        [ -e "$Q/done/$t" ] || [ -e "$Q/failed/$t" ] || left=$((left+1))
    done < <(awk -F, 'NR>1 && $3<=2 {print $1}' "$VAL_DIR/tasks.csv")
    [ "$left" -eq 0 ] || return
    mkdir "$Q/interim.lock" 2>/dev/null || return
    echo "  priority 0-2 finished: interim check"
    ( cd "$VAL_DIR" && python analysis/validation_report.py --interim ) > "$Q/log/interim.log" 2>&1
    touch "$Q/interim.done"
    [ -f "$Q/STOP" ] && echo "  interim check wrote STOP: $(head -3 "$Q/STOP")"
}

echo "=== validation worker IDX=$IDX JOB=$JOB_ID NODE=$NODE start=$(date -Is) ==="
declare -A TRIED=()
processed=0; infra=0
while :; do
    if [ -f "$Q/STOP" ]; then echo "  STOP present — not claiming: $(head -1 "$Q/STOP")"; break; fi
    REMAIN=$(( WALL_S - ($(date +%s) - JOB_START) ))
    if [ $REMAIN -lt $SAFETY_S ]; then
        echo "  wall guard: ${REMAIN}s left < ${SAFETY_S}s — resubmit worker.sh to continue"; break
    fi
    TID=""; busy=0
    while read -r T; do
        [ -e "$Q/done/$T" ] && continue
        [ -e "$Q/failed/$T" ] && continue
        [ -n "${TRIED[$T]:-}" ] && continue
        if ! try_claim "$Q/claims/$T"; then busy=$((busy+1)); continue; fi
        TID=$T; break
    done < <(awk -F, 'NR>1 {print $1}' "$VAL_DIR/tasks.csv")
    [ -z "$TID" ] && { echo "  nothing left to claim (held by live workers: $busy)"; break; }
    TRIED[$TID]=1
    CLAIM_DIR=$Q/claims/$TID
    if ! echo "$JOB_ID $IDX $NODE $(date -Is)" > "$CLAIM_DIR/owner"; then
        echo "  [SKIP] $TID cannot write the claim owner file"; rm -rf "$CLAIM_DIR"; continue
    fi
    # another worker may have finished it between the scan and the claim
    if [ -e "$Q/done/$TID" ] || [ -e "$Q/failed/$TID" ]; then rm -rf "$CLAIM_DIR"; continue; fi
    heartbeat "$CLAIM_DIR" & HB=$!
    T0=$(date +%s)
    echo "  [RUN] $TID $(date -Is)"
    bash "$VAL_DIR/run_task.sh" "$TID" >> "$Q/log/$TID.log" 2>&1
    RC=$?
    EL=$(( $(date +%s) - T0 ))
    kill $HB 2>/dev/null; wait $HB 2>/dev/null
    if ! mine; then
        echo "  [LOST] $TID claim was taken over while it ran (rc=$RC ${EL}s) — left to the new owner"; continue
    fi
    if [ $RC -eq 0 ]; then
        echo "$JOB_ID $NODE $EL $(date -Is)" > "$Q/done/$TID"; processed=$((processed+1))
        echo "  [DONE] $TID ${EL}s"
    else
        echo "$JOB_ID $NODE $EL rc=$RC $(date -Is)" > "$Q/failed/$TID"; infra=$((infra+1))
        echo "  [FAILED] $TID rc=$RC ${EL}s — see $Q/log/$TID.log (not retried automatically)"
    fi
    rm -rf "$CLAIM_DIR"
    maybe_interim
done
echo "=== worker IDX=$IDX done processed=$processed infra_failed=$infra end=$(date -Is) ==="
