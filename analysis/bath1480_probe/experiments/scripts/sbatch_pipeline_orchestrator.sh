#!/bin/bash
# Phase 0-5 pipeline orchestrator for the 3504-rxn cohort.
# Runs on compute node (CLAUDE.md: no login-node launchers).
#
# Waves (respect QoS: max 10 running / 12 submit):
#   1. Phase 0 (1 task)                         — build cohort/labels/features
#   2. Phase 1 (5) + Phase 2 (5) = 10           — sklearn Espley/ArmB tau=1e-10
#   3. Phase 3 (5) + Phase 4a (3) = 8           — sklearn ArmB tau=0.05 + MACE TS
#   4. Phase 4b (1)                             — physics_24
#   5. Phase 4c chunks: 0-9, 10-19, 20-24        — XGB+delta (each ≤10 tasks)
#   6. Phase 5 (1)                              — aggregate/figures
#
# Each wave uses sbatch --wait so the orchestrator blocks until wave finishes.
#
#SBATCH --job-name=pipe_orch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/orchestrator.%j.log

set -uo pipefail

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
COHORT_DIR=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1
RESULTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results
LABELS=$COHORT_DIR/labels_v1.pkl
FEAT_ESPLEY=$COHORT_DIR/features_espley_v1.pkl
FEAT_ARMB=$COHORT_DIR/features_armB_v1.pkl

echo "=== orchestrator start $(date -Is) host=$(hostname) ==="

wait_jobs() {
    # Poll every 20s until listed job IDs are all out of queue (tight for fast turnaround).
    local ids="$1"
    while true; do
        n=$(squeue -j "$ids" -h 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && break
        sleep 20
    done
}

# ============================================================
# Wave 1: Phase 0 (skip if labels_v1.pkl already exists; idempotent)
# ============================================================
if [ -f "$LABELS" ] && [ -f "$FEAT_ESPLEY" ] && [ -f "$FEAT_ARMB" ]; then
    echo "=== Wave 1: Phase 0 SKIP (labels/features pkls already present) $(date -Is) ==="
    JP0=""
else
    echo "=== Wave 1: Phase 0 (cohort/labels/features build) $(date -Is) ==="
    JP0=$(sbatch --parsable $SCRIPTS/sbatch_phase0.sh)
    echo "P0 jid=$JP0"
fi

# ============================================================
# Wave 2: Phase 1 (Espley) + Phase 2 (ArmB tau=1e-10) = 10 tasks
# Pre-chained via --dependency=afterok:$JP0 so they queue immediately.
# Column names use Phase-0-renamed schema (d1_dft_espley etc.).
# ============================================================
echo "=== Wave 2: Phase 1 + Phase 2 (pre-chained on P0) $(date -Is) ==="
DEP_P0=""
[ -n "$JP0" ] && DEP_P0="--dependency=afterok:$JP0"

J1=$(sbatch --parsable $DEP_P0 --export=ALL,\
PHASE_NAME=phase1_espley,\
LABELS_PKL=$LABELS,\
FEATURES_PKL=$FEAT_ESPLEY,\
"TARGET_COLS=d1_dft_espley d2_dft_espley sum_dist_espley interaction_espley barrier_e_espley barrier_q_espley",\
VAR_TAU=0.05 \
    $SCRIPTS/sbatch_phase123.sh)
J2=$(sbatch --parsable $DEP_P0 --export=ALL,\
PHASE_NAME=phase2_armB_tau1e-10,\
LABELS_PKL=$LABELS,\
FEATURES_PKL=$FEAT_ARMB,\
"TARGET_COLS=d1_own d2_own elst_dft pauli_dft oi_dft disp_dft cpcm_dft cds_dft",\
VAR_TAU=1e-10 \
    $SCRIPTS/sbatch_phase123.sh)
echo "P1 jid=$J1  P2 jid=$J2 (both queued with dep on P0)"
wait_jobs "$J1,$J2"
echo "P1+P2 done at $(date -Is)"

# ============================================================
# Wave 3: Phase 3 (ArmB tau=0.05, 5 CPU) + Phase 4a MACE TS (3 GPU)
# ============================================================
echo "=== Wave 3: Phase 3 + Phase 4a MACE $(date -Is) ==="
J3=$(sbatch --parsable --export=ALL,\
PHASE_NAME=phase3_armB_tau0.05,\
LABELS_PKL=$LABELS,\
FEATURES_PKL=$FEAT_ARMB,\
"TARGET_COLS=d1_own d2_own elst_dft pauli_dft oi_dft disp_dft cpcm_dft cds_dft",\
VAR_TAU=0.05 \
    $SCRIPTS/sbatch_phase123.sh)
J4A=$(sbatch --parsable $SCRIPTS/sbatch_phase4_mace.sh)
echo "P3 jid=$J3  P4a jid=$J4A"
wait_jobs "$J3,$J4A"
echo "P3+P4a done at $(date -Is)"

# ============================================================
# Wave 4: Phase 4b physics_24
# ============================================================
echo "=== Wave 4: Phase 4b (physics_24) $(date -Is) ==="
J4B=$(sbatch --parsable --job-name=exp_p4phys \
    --time=48:00:00 --partition=cpu1,cpu2 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --mem=8G \
    --output=$RESULTS/phase4_physics.%j.log \
    --wrap="source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; \
            conda activate reactot; \
            python3 -u $SCRIPTS/04_compute_physics_24.py")
echo "P4b jid=$J4B"
wait_jobs "$J4B"
echo "P4b done at $(date -Is)"

# ============================================================
# Wave 5: Phase 4c chunks 0-9, 10-19, 20-24 (each ≤10 array tasks)
# ============================================================
run_p4c_chunk() {
    local range=$1
    echo "=== P4c chunk $range starting $(date -Is) ==="
    local JID=$(sbatch --parsable --array=$range%3 $SCRIPTS/sbatch_phase4_train.sh)
    echo "P4c[$range] jid=$JID"
    wait_jobs "$JID"
    echo "P4c[$range] done at $(date -Is)"
}
run_p4c_chunk 0-9
run_p4c_chunk 10-19
run_p4c_chunk 20-24

# ============================================================
# Wave 6: Phase 5 aggregate + figures
# ============================================================
echo "=== Wave 6: Phase 5 aggregate $(date -Is) ==="
JP5=$(sbatch --parsable $SCRIPTS/sbatch_phase5.sh)
echo "P5 jid=$JP5"
wait_jobs "$JP5"
echo "P5 done at $(date -Is)"

echo "=== ALL PHASES COMPLETE $(date -Is) ==="
ls -la $RESULTS/comparison_v1/ 2>/dev/null || echo "WARN: no comparison_v1 output"
exit 0
