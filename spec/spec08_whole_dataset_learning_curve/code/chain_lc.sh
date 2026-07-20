#!/bin/bash
# SPEC_08 whole-dataset LC — auto-launcher, runs on a compute node
# (submitted by submit_chain_lc.sh onto cpu2). Iterates through
# (SIZE, FOLD, MEMBER) cells and submits each as running slots free up.
#
# Rules (CLAUDE.md):
#   - max 10 concurrent running (MAX_INFLIGHT) — running-only count
#   - stay under MaxSubmit=20 with a safety margin (MAX_SUBMIT=19)
#   - one-shot sbatch calls only, never a long-lived login-node process
#
# Env overrides:
#   SIZES="100 200 ..."
#   FOLDS="0 1 2 3 4"
#   MEMBERS="0"
#   MAX_INFLIGHT=10
#   MAX_SUBMIT=19
#   POLL_SECONDS=60
#   USER_TAG=yeseo1ee
#   DRY_RUN=1  (print sbatch, skip slot wait)

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

REPO="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
SPEC="${REPO}/spec/spec08_whole_dataset_learning_curve"
SUBMIT="${SPEC}/code/submit_lc_cell.sh"
OOF_ROOT="${SPEC}/oof"
LOG="${SPEC}/code/chain_lc.log"

SIZES="${SIZES:-100 200 300 400 500 600 700 786}"
FOLDS="${FOLDS:-0 1 2 3 4}"
MEMBERS="${MEMBERS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-10}"
MAX_SUBMIT="${MAX_SUBMIT:-19}"
POLL_SECONDS="${POLL_SECONDS:-30}"
USER_TAG="${USER_TAG:-${USER}}"
DRY_RUN="${DRY_RUN:-0}"

: > "${LOG}"

logmsg() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

n_running() {
    squeue -u "${USER_TAG}" -r -h -t R 2>/dev/null | wc -l
}
n_total() {
    squeue -u "${USER_TAG}" -r -h 2>/dev/null | wc -l
}

cell_done() {
    local size="$1" fold="$2" member="$3"
    # already-written output → definitely done
    if [ -f "${OOF_ROOT}/size${size}/fold${fold}/member${member}.json" ]; then
        return 0
    fi
    # in-flight (pending or running) → treat as "done" for the purposes of
    # skipping resubmission, so a launcher restart doesn't double-book a
    # (size, fold, member) triple that a prior launcher already sent to
    # the scheduler. Match the SIZE/FOLD tag inside each cell's stdout log
    # (identical marker across old/new launchers).
    for jid in $(squeue -u "${USER_TAG}" -h -n s08w_lc -o '%i' 2>/dev/null); do
        local log="/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_${jid}.out"
        if [ -f "${log}" ] \
           && grep -qE "SIZE\] ${size} +\[FOLD\] ${fold} +\[MEMBER\] ${member}" \
                  "${log}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

wait_for_slot() {
    # Gate on total (pending + running) only, not on running count. SLURM's
    # own MaxJobs=10 caps how many actually execute at once; MAX_INFLIGHT
    # here is retained for logging but not enforced, so we can pre-queue
    # pending cells up to MAX_SUBMIT. When a running cell finishes, SLURM
    # promotes a pending one instantly — no polling gap.
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    while true; do
        local r t
        r=$(n_running)
        t=$(n_total)
        if [ "${t}" -lt "${MAX_SUBMIT}" ]; then
            return 0
        fi
        logmsg "wait: running=${r}/${MAX_INFLIGHT} total=${t}/${MAX_SUBMIT}; sleep ${POLL_SECONDS}s"
        sleep "${POLL_SECONDS}"
    done
}

submit_cell() {
    local size="$1" fold="$2" member="$3"
    if [ "${DRY_RUN}" = "1" ]; then
        logmsg "DRY size=${size} fold=${fold} member=${member}"
        return 0
    fi
    local out rc jid
    while true; do
        out=$(sbatch --export=ALL,SIZE="${size}",FOLD="${fold}",MEMBER="${member}" \
                     "${SUBMIT}" 2>&1)
        rc=$?
        if [ "${rc}" -eq 0 ]; then
            logmsg "OK  size=${size} fold=${fold} member=${member}: ${out}"
            jid=$(echo "${out}" | grep -oE '[0-9]+' | tail -1)
            if [ -n "${jid}" ]; then
                CELL_JIDS+=("${jid}")
            fi
            return 0
        fi
        if echo "${out}" | grep -q "AssocMaxSubmitJobLimit"; then
            logmsg "SLURM cap on submit; sleep ${POLL_SECONDS}s"
            sleep "${POLL_SECONDS}"
            continue
        fi
        logmsg "ERR size=${size} fold=${fold} member=${member} rc=${rc}: ${out}"
        return "${rc}"
    done
}

logmsg "chain_lc start SIZES='${SIZES}' FOLDS='${FOLDS}' MEMBERS='${MEMBERS}'"
logmsg "               MAX_INFLIGHT=${MAX_INFLIGHT} MAX_SUBMIT=${MAX_SUBMIT} POLL=${POLL_SECONDS}s"

# Track every spec08 cell job we need the aggregator to wait for. Seed with
# any that are already in the queue (a prior launcher's leftovers) so a
# restart-scheduled aggregate depends on those too.
CELL_JIDS=()
for jid in $(squeue -u "${USER_TAG}" -h -n s08w_lc -o '%i' 2>/dev/null); do
    CELL_JIDS+=("${jid}")
done
logmsg "pre-existing s08w_lc jobs in queue: ${#CELL_JIDS[@]}"

n_planned=0; n_skipped=0; n_submitted=0
# Iterate size-major so early progress covers every size across folds first.
for M in ${MEMBERS}; do
    for S in ${SIZES}; do
        for F in ${FOLDS}; do
            n_planned=$((n_planned + 1))
            if cell_done "${S}" "${F}" "${M}"; then
                logmsg "SKIP done: size=${S} fold=${F} member=${M}"
                n_skipped=$((n_skipped + 1))
                continue
            fi
            wait_for_slot
            if submit_cell "${S}" "${F}" "${M}"; then
                n_submitted=$((n_submitted + 1))
            fi
        done
    done
done

logmsg "chain_lc done planned=${n_planned} skipped=${n_skipped} submitted=${n_submitted}"

# Schedule the aggregator on cpu2 with an afterany dependency on every
# spec08 cell we know about. `afterany` fires whether cells succeed or
# fail — no completed cells' JSONs are missing from disk by then, and
# the aggregator handles missing-cell reporting itself.
if [ "${DRY_RUN}" != "1" ] && [ "${#CELL_JIDS[@]}" -gt 0 ]; then
    dep_list=$(IFS=:; echo "${CELL_JIDS[*]}")
    logmsg "scheduling aggregator with dependency on ${#CELL_JIDS[@]} cell jobs"
    out=$(sbatch --dependency=afterany:"${dep_list}" \
                 --kill-on-invalid-dep=yes \
                 "${SPEC}/code/submit_aggregate.sh" 2>&1)
    logmsg "aggregator submit: ${out}"
else
    logmsg "no cells tracked — aggregator not scheduled"
fi
