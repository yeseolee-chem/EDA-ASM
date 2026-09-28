#!/bin/bash
# SMD-keyword eda.inp re-runner. One job = one worker.
#
# Dynamic work-claim via mkdir (atomic on POSIX). Each worker scans the list of
# inputs/rxn_* dirs starting from its own offset (IDX), runs ORCA on the first
# rxn that has no done flag and no live claim, then scans again. Idempotent skip
# via the done flag or ORCA TERMINATED NORMALLY in eda.out.
#
# Preserves ORCA auto-generated files (eda_frag1.out / eda_frag2.out / eda.out)
# — stage3_parse.py needs them for the BSSE-corrected fragment reference and the
# EDA table. Only .tmp/.gbw/.densities/etc are cleaned.
#
# v4 (2026-09-25):
#  - work list = actual inputs/rxn_* dirs (the numeric 0..TOTAL range missed
#    ids above 5264 in v2 and made v3 depend on a hand-set upper bound)
#  - wall guard 4 h (largest rxn so far 2.3 h): the worker stops taking new
#    rxns and exits before the 48 h wall instead of being killed mid-rxn
#  - claims are removed with rm -rf (they hold an owner file, so rmdir never
#    removed them)
#  - each rxn starts clean: everything except eda.inp is deleted before ORCA
#    runs, so ORCA never restarts from a half-written .gbw
#  - successor without dependency: a plain job that waits on AssocMaxJobsLimit
#    and starts as soon as any slot frees. It is submitted after this worker's
#    first successful rxn, so a worker with no work does not spawn one and the
#    chain stops by itself. A worker that found only rxns held by live workers
#    leaves a successor delayed by 30 min to recheck.
#
# v5 (2026-09-25, review fixes):
#  - rescan from the start after every rxn (each rxn is tried at most once per
#    job), so claims orphaned by a killed job are picked up by the running
#    generation, and rxns merged in from returned bundles are seen
#  - failures: a slow failure (>= 5 min) counts toward MAX_FAIL=2 for the rxn.
#    A fast failure (< 5 min, node/environment suspected) goes to
#    fail/<rxn>.fast instead; the rxn is skipped only after fast failures on 3
#    distinct nodes. Two fast failures in a row flag the node
#    (state/badnode/<node>) and stop the worker; successors are submitted with
#    --exclude=<flagged nodes>, a job that lands on a flagged node hands off,
#    and a lineage stopped by a bad node is replaced while < 3 nodes are flagged
#  - claim liveness: while ORCA runs a heartbeat touches claim/<rxn>/owner every
#    2 min. A claim is stale only when the owner file is 15+ min old AND the
#    owner job is in no non-final squeue state (R/CG/CF/S/ST/PD all count as
#    alive), so a COMPLETING owner at the 48 h wall is never taken over
#  - takeover re-checks that the claim is unchanged after winning the marker,
#    releases the marker if the takeover cannot finish, and clears a marker
#    left 2+ h by a takeover that died half way. A claim with no readable owner
#    is stale only after 6 h. The owner write is checked
#  - after ORCA returns, the worker touches state only if it still owns the claim
#
#SBATCH --job-name=smd_wrk
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --array=0-9%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/wrk_%A_%a.out
set -uo pipefail

# SMD_ORCA_BIN / SMD_ROOT / SMD_HEARTBEAT_S exist only for dry tests
export ORCA_BIN=${SMD_ORCA_BIN:-/home1/yeseo1ee/orca_6_1_1_avx2/orca}
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1
export PATH=$MPI_ROOT/bin:$(dirname $ORCA_BIN):$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib64:$(dirname $ORCA_BIN):${LD_LIBRARY_PATH:-}
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp,self,sm

ROOT=${SMD_ROOT:-/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel}
INPUTS=$ROOT/inputs
STATE=$ROOT/state
BADNODE=$STATE/badnode
SCRIPT=/home1/yeseo1ee/projects/eda-asm-prediction/label_true/scripts/smd_relabel_worker.sh
mkdir -p $STATE/done $STATE/claim $STATE/timing $STATE/fail $BADNODE
IDX=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID:-manual}
ME=${USER:-$(id -un)}
JOB_START=$(date +%s)
WALL_S=$((48*3600))
SAFETY_S=$((4*3600))          # stop taking new rxns when < 4 h of wall is left
HEARTBEAT_S=${SMD_HEARTBEAT_S:-120}  # owner file refresh interval while ORCA runs
STALE_AGE_S=900               # owner file older than this + owner job gone = stale
EMPTY_OWNER_AGE_S=$((6*3600)) # a claim with no readable owner is stale only after this
MARKER_AGE_S=7200             # a takeover marker this old was left by a dead takeover
FAST_FAIL_S=300               # a failure faster than this is blamed on the node
MAX_FAIL=2                    # slow failures before a rxn is skipped
MAX_FAST_NODES=3              # distinct nodes with a fast failure before a rxn is skipped
MAX_BAD_NODES=3               # no lineage replacement once this many nodes are flagged
NODE=$(hostname -s)

n_bad() { ls "$BADNODE" 2>/dev/null | wc -l; }

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
# takeover marker for that exact (rxn, owner, claim mtime); nobody else can then
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

submit_successor() {    # $1 = extra sbatch options, e.g. --begin=now+1800
    [ -n "${NO_SUCCESSOR:-}" ] && { echo "  (NO_SUCCESSOR set: would submit ${1:-plain})"; return 1; }
    local s ex
    ex=$(ls "$BADNODE" 2>/dev/null | paste -sd,)
    s=$(sbatch --parsable ${ex:+--exclude=$ex} $1 --array="$IDX" --job-name=smd_wrk "$SCRIPT" 2>&1 | tail -1)
    if [[ "$s" =~ ^[0-9]+$ ]]; then
        echo "  successor queued: $s ${1:-(no dependency)} ${ex:+exclude=$ex}"; return 0
    fi
    echo "  successor submit FAILED: $s"; return 1
}

mine() { [ "$(awk '{print $1}' "$CLAIM_DIR/owner" 2>/dev/null)" = "$JOB_ID" ]; }

heartbeat() {   # $1 = claim dir; keeps its owner file fresh while ORCA runs
    local sp=
    trap '[ -n "$sp" ] && kill $sp 2>/dev/null; exit 0' TERM
    while :; do
        sleep "$HEARTBEAT_S" & sp=$!
        wait $sp
        touch -c "$1/owner" 2>/dev/null
    done
}

echo "=== worker IDX=$IDX JOB=$JOB_ID NODE=$NODE start=$(date -Is) ==="

if [ -e "$BADNODE/$NODE" ]; then
    echo "  node $NODE is flagged bad (state/badnode) — not claiming, handing off"
    [ "$(n_bad)" -lt $MAX_BAD_NODES ] && submit_successor "--begin=now+600"
    echo "=== worker IDX=$IDX handed off end=$(date -Is) ==="
    exit 0
fi

declare -A TRIED=()
processed=0; fails=0; fastrun=0; chained=0; stopped_bad=0
skipped=0; busy=0; failskip=0

while :; do
    REMAIN=$(( WALL_S - ($(date +%s) - JOB_START) ))
    if [ $REMAIN -lt $SAFETY_S ]; then
        echo "  wall guard: ${REMAIN}s left < ${SAFETY_S}s — exiting cleanly"
        [ $chained -eq 0 ] && submit_successor "" && chained=1
        break
    fi
    # Rescan the whole list every time: orphaned claims and rxns merged in from
    # returned bundles are picked up by this job too.
    mapfile -t RXNS < <(ls -1 "$INPUTS" | grep -E '^rxn_[0-9]+$' | sort)
    TOTAL=${#RXNS[@]}
    if [ "$TOTAL" -eq 0 ]; then echo "  no rxn dirs in $INPUTS"; break; fi
    RID=""; skipped=0; busy=0; failskip=0
    for OFFSET in $(seq 0 $((TOTAL - 1))); do
        R=${RXNS[$(( (IDX + OFFSET) % TOTAL ))]}
        [ -f "$STATE/done/$R" ] && { skipped=$((skipped+1)); continue; }
        [ -n "${TRIED[$R]:-}" ] && continue
        [ -f "$INPUTS/$R/eda.inp" ] || continue
        if [ -f "$INPUTS/$R/eda.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$INPUTS/$R/eda.out" 2>/dev/null; then
            touch "$STATE/done/$R"; skipped=$((skipped+1)); continue
        fi
        F=$STATE/fail/$R
        if { [ -f "$F" ] && [ "$(wc -l < "$F")" -ge $MAX_FAIL ]; } || \
           { [ -f "$F.fast" ] && [ "$(awk '{print $NF}' "$F.fast" | sort -u | wc -l)" -ge $MAX_FAST_NODES ]; }; then
            failskip=$((failskip+1)); continue
        fi
        if ! try_claim "$STATE/claim/$R"; then busy=$((busy+1)); continue; fi
        RID=$R; break
    done
    [ -z "$RID" ] && break
    TRIED[$RID]=1

    DONE_FLAG=$STATE/done/$RID
    CLAIM_DIR=$STATE/claim/$RID
    FAIL_LOG=$STATE/fail/$RID
    RUN_DIR=$INPUTS/$RID
    if ! echo "$JOB_ID $IDX $NODE $(date -Is)" > "$CLAIM_DIR/owner"; then
        echo "  [SKIP] $RID cannot write the claim owner file"; rm -rf "$CLAIM_DIR"; continue
    fi
    # Another worker may have finished this rxn between the check and the claim.
    if [ -f "$DONE_FLAG" ]; then rm -rf "$CLAIM_DIR"; continue; fi
    if ! cd "$RUN_DIR"; then
        echo "  [SKIP] $RID cannot cd into $RUN_DIR"; rm -rf "$CLAIM_DIR"; continue
    fi
    # Clean start: drop everything but the input (partial output of a killed run).
    find . -maxdepth 1 -type f ! -name "eda.inp" -delete
    heartbeat "$CLAIM_DIR" & HB=$!
    T0=$(date +%s)
    $ORCA_BIN eda.inp > eda.out 2> eda.err
    RC=$?
    ELAPSED_RXN=$(( $(date +%s) - T0 ))
    kill $HB 2>/dev/null; wait $HB 2>/dev/null
    if ! mine; then
        echo "  [LOST] $RID claim was taken over while ORCA ran (rc=$RC ${ELAPSED_RXN}s) — left to the new owner"
        cd "$ROOT"; continue
    fi
    if [ $RC -eq 0 ] && grep -q "ORCA TERMINATED NORMALLY" eda.out; then
        touch "$DONE_FLAG"
        echo "$RID $ELAPSED_RXN $NODE" >> $STATE/timing/w${IDX}.tsv
        processed=$((processed+1)); fastrun=0
        find . -maxdepth 1 -type f \
            ! -name "*.inp" ! -name "*.out" ! -name "*.err" \
            -delete
        echo "  [OK] $RID  ${ELAPSED_RXN}s  processed=$processed"
        cd "$ROOT"; rm -rf "$CLAIM_DIR"
        if [ $chained -eq 0 ] && submit_successor ""; then chained=1; fi
    else
        fails=$((fails+1))
        cd "$ROOT"; rm -rf "$CLAIM_DIR"
        if [ $ELAPSED_RXN -lt $FAST_FAIL_S ]; then
            echo "$JOB_ID $(date -Is) rc=$RC ${ELAPSED_RXN}s $NODE" >> "$FAIL_LOG.fast"
            fastrun=$((fastrun+1))
            echo "  [FASTFAIL] $RID rc=$RC ${ELAPSED_RXN}s on $NODE — not counted toward MAX_FAIL ($fastrun in a row)"
            if [ $fastrun -ge 2 ]; then
                touch "$BADNODE/$NODE"; stopped_bad=1
                echo "  two fast failures in a row: $NODE flagged bad ($(n_bad) flagged) — stopping this worker"
                if [ $chained -eq 0 ] && [ "$(n_bad)" -lt $MAX_BAD_NODES ]; then
                    submit_successor "--begin=now+600" && chained=1
                fi
                break
            fi
        else
            echo "$JOB_ID $(date -Is) rc=$RC ${ELAPSED_RXN}s $NODE" >> "$FAIL_LOG"
            echo "  [FAIL] $RID rc=$RC  ${ELAPSED_RXN}s — attempt $(wc -l < "$FAIL_LOG")/$MAX_FAIL, files kept"
            fastrun=0
        fi
    fi
done

# Only rxns held by other live workers were left: recheck in 30 min so an
# orphaned claim is still picked up after this chain would otherwise stop.
if [ $chained -eq 0 ] && [ $stopped_bad -eq 0 ] && [ $busy -gt 0 ]; then
    submit_successor "--begin=now+1800" && chained=1
fi

echo "=== worker IDX=$IDX done processed=$processed skipped=$skipped busy=$busy failskip=$failskip fails=$fails successor=$chained end=$(date -Is) ==="
