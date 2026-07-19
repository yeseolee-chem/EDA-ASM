#!/bin/bash
# SPEC_10 — auto-launcher for the per-family learning curve. Runs on the
# login node under nohup; iterates through (FAMILY, SIZE, FOLD, MEMBER)
# cells and submits each as a queue slot becomes free.
#
# Queue policy (CLAUDE.md): no more than 10 concurrent jobs. Before every
# sbatch we count the caller's jobs and wait if we already hold 10+.
# The trainer is idempotent, so if a slot opens between check and submit
# and we race with a manual submission there is no correctness risk.
#
# Usage:
#   nohup bash spec/spec10_family_learning_curve/code/chain_lc_family.sh \
#         > spec/spec10_family_learning_curve/code/chain_lc_family.nohup 2>&1 &
#
# Environment overrides:
#   FAMILIES="dipolar qmrxn20_e2 ..."
#   SIZES="50 100 150 200"
#   FOLDS="0 1 2 3 4"
#   MEMBERS="0"
#   MAX_INFLIGHT=10   (default: 10, from CLAUDE.md)
#   POLL_SECONDS=180  (default: 3 min between queue checks)
#   USER_TAG=yeseo1ee (default: $USER)
#   DRY_RUN=1         (print sbatch commands without executing; skip slot wait)

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

REPO="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
SPEC="${REPO}/spec/spec10_family_learning_curve"
SUBMIT="${SPEC}/code/submit_lc_family_cell.sh"
OOF_ROOT="${SPEC}/oof"
LOG="${SPEC}/code/chain_lc_family.log"

FAMILIES="${FAMILIES:-dipolar qmrxn20_e2 qmrxn20_sn2 rgd1}"
SIZES="${SIZES:-50 100 150}"
FOLDS="${FOLDS:-0 1 2 3 4}"
MEMBERS="${MEMBERS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
USER_TAG="${USER_TAG:-${USER}}"
DRY_RUN="${DRY_RUN:-0}"

: > "${LOG}"

logmsg() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

n_running_jobs() {
    # Only count RUNNING jobs against the concurrency cap. Pending jobs of
    # ours (e.g. someone else's cpu2 job stuck in the queue) don't consume
    # a compute slot, so blocking on them wastes a running slot. SLURM's
    # own MaxJobs=10 caps running for us regardless, so we can't exceed it
    # even if our count races ahead of newly-started spec10 cells.
    squeue -u "${USER_TAG}" -r -h -t R 2>/dev/null | wc -l
}

n_total_submitted() {
    # Also track total (pending + running) so we stay under MaxSubmit=20.
    squeue -u "${USER_TAG}" -r -h 2>/dev/null | wc -l
}

cell_done() {
    local fam="$1" size="$2" fold="$3" member="$4"
    [ -f "${OOF_ROOT}/${fam}/size${size}/fold${fold}/member${member}.json" ]
}

wait_for_slot() {
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    # Guard against MaxSubmit=20 too: refuse to submit if the total
    # (pending + running) already sits at the SLURM submit cap.
    local max_submit="${MAX_SUBMIT:-19}"
    while true; do
        local n_run n_total
        n_run=$(n_running_jobs)
        n_total=$(n_total_submitted)
        if [ "${n_run}" -lt "${MAX_INFLIGHT}" ] && [ "${n_total}" -lt "${max_submit}" ]; then
            return 0
        fi
        logmsg "wait: running=${n_run}/${MAX_INFLIGHT} total=${n_total}/${max_submit}; sleeping ${POLL_SECONDS}s"
        sleep "${POLL_SECONDS}"
    done
}

submit_cell() {
    local fam="$1" size="$2" fold="$3" member="$4"
    local out
    if [ "${DRY_RUN}" = "1" ]; then
        logmsg "DRY fam=${fam} size=${size} fold=${fold} member=${member}"
        return 0
    fi
    while true; do
        out=$(sbatch --export=ALL,FAMILY="${fam}",SIZE="${size}",FOLD="${fold}",MEMBER="${member}" \
                     "${SUBMIT}" 2>&1)
        rc=$?
        if [ "${rc}" -eq 0 ]; then
            logmsg "OK  fam=${fam} size=${size} fold=${fold} member=${member}: ${out}"
            return 0
        fi
        if echo "${out}" | grep -q "AssocMaxSubmitJobLimit"; then
            n=$(n_running_jobs)
            logmsg "FULL slots=${n} on submit; sleeping ${POLL_SECONDS}s"
            sleep "${POLL_SECONDS}"
            continue
        fi
        logmsg "ERR fam=${fam} size=${size} fold=${fold} member=${member} rc=${rc}: ${out}"
        return "${rc}"
    done
}

logmsg "chain_lc_family start"
logmsg "  FAMILIES='${FAMILIES}'"
logmsg "  SIZES='${SIZES}'  FOLDS='${FOLDS}'  MEMBERS='${MEMBERS}'"
logmsg "  MAX_INFLIGHT=${MAX_INFLIGHT}  POLL_SECONDS=${POLL_SECONDS}"

n_planned=0
n_skipped=0
n_submitted=0
# Order (member outer, family, size, fold inner) so early progress covers
# every (family, size) combination — you can watch a learning-curve
# preview after one round of cells finishes instead of waiting for all
# folds of one family.
for M in ${MEMBERS}; do
    for FAM in ${FAMILIES}; do
        for S in ${SIZES}; do
            for F in ${FOLDS}; do
                n_planned=$((n_planned + 1))
                if cell_done "${FAM}" "${S}" "${F}" "${M}"; then
                    logmsg "SKIP already done: fam=${FAM} size=${S} fold=${F} member=${M}"
                    n_skipped=$((n_skipped + 1))
                    continue
                fi
                wait_for_slot
                if submit_cell "${FAM}" "${S}" "${F}" "${M}"; then
                    n_submitted=$((n_submitted + 1))
                fi
            done
        done
    done
done

logmsg "chain_lc_family done  planned=${n_planned} skipped=${n_skipped} submitted=${n_submitted}"
