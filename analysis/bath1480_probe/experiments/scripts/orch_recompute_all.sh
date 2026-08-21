#!/bin/bash
# Continuous dispatcher: submit XGB + physics_24 + phase4c after sklearn drains.
# Keeps queue at ≤10 tasks. Idempotent skip via output existence.
#
#SBATCH --job-name=orch_recompute
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/orch_recompute.%j.log

set -o pipefail

EXP=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments
SCRIPTS=$EXP/scripts
MAX_QUEUE=${MAX_QUEUE:-10}   # keep our queue ≤ MAX_QUEUE tasks
POLL=30

wait_for_slot() {
    local need=$1
    while true; do
        n=$(squeue -u $USER -h -r | wc -l)
        avail=$((MAX_QUEUE - n))
        if [ "$avail" -ge "$need" ]; then break; fi
        sleep $POLL
    done
}

wait_jobs() {
    local ids="$1"
    while true; do
        n=$(squeue -j "$ids" -h 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && break
        sleep $POLL
    done
}

echo "=== orch_recompute start $(date -Is) ==="

# --- Step 0: build v2 datasets + hard gate on persistent artifacts ---
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
python3 $SCRIPTS/build_v2_datasets.py || { echo "FAIL: v2 builder"; exit 10; }
python3 - <<'PY' || { echo "FAIL: v2 gate"; exit 11; }
import pandas as pd, numpy as np
from pathlib import Path
EXP = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
ORACLE = ["disp_xtb", "dispd4_xtb", "eint_total_xtb"]
for tag in ["phase2","phase3","phase5"]:
    p = EXP/f"cohort_v1/{tag}_dataset_v2.pkl"
    assert p.exists(), f"MISSING {p}"
    df = pd.read_pickle(p)
    assert len(df) == 3504, f"{tag}: rows={len(df)} != 3504"
    for c in ORACLE:
        assert c not in df.columns, f"{tag}: oracle '{c}' still present"
    if "strain_1_xtb" in df.columns and "strain_2_xtb" in df.columns:
        r1 = np.corrcoef(df["strain_1_xtb"], df["d1_own_dft"])[0,1]
        r2 = np.corrcoef(df["strain_1_xtb"], df["d2_own_dft"])[0,1]
        assert abs(r1) > abs(r2), f"{tag}: strain swap NOT applied (|r(d1)|={abs(r1):.3f} < |r(d2)|={abs(r2):.3f})"
        print(f"  {tag}: gate OK  |r(strain_1,d1)|={abs(r1):.3f}  |r(strain_1,d2)|={abs(r2):.3f}")
    else:
        print(f"  {tag}: strain_1_xtb / strain_2_xtb absent (τ dropped) — no swap check")
    print(f"  {tag}: shape={df.shape}, oracle features absent ✓")
print("=== v2 gate passed ===")
PY

# --- Sklearn Phase 2/3/5 (v2 datasets) ---
wait_for_slot 1
JSKL=$(sbatch --parsable $SCRIPTS/sbatch_phase235_v2_sklearn.sh)
echo "Sklearn Phase 2/3/5 v2 jid=$JSKL $(date -Is)"

# --- XGB Phase 2/3/5 (single array of 15 tasks with %10 conc → counts as 1 submit slot) ---
wait_for_slot 1
JXGB=$(sbatch --parsable $SCRIPTS/sbatch_xgb_phase235.sh)
echo "XGB Phase 2/3/5 jid=$JXGB $(date -Is)"

# --- Phase 4b physics_v2 rebuild (oracle-clean + strain swap fix) ---
wait_for_slot 1
JPHYS=$(sbatch --parsable --job-name=exp_p4phys \
    --time=48:00:00 --partition=cpu1,cpu2 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --mem=8G \
    --output=$EXP/results/phase4_physics.%j.log \
    --wrap="source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; \
            conda activate reactot; \
            python3 -u $SCRIPTS/04_compute_physics_24_clean.py")
echo "Phase 4b physics jid=$JPHYS $(date -Is)"
wait_jobs "$JPHYS"

# --- Phase 4c 25 cells, dispatch continuously to versioned _v2 dir ---
# Idempotent skip against phase4_stacked_xgb_mace_v2 (fresh path — no stale skips).
echo "Phase 4c dispatch loop $(date -Is)"
declare -a SUBMITTED4C
for CELL in $(seq 0 24); do
    FOLD=$((CELL / 5)); SEED=$((CELL % 5))
    if [ -f "$EXP/results/phase4_stacked_xgb_mace_v2/fold_$FOLD/seed_$SEED/metrics.json" ]; then
        echo "SKIP cell $CELL (already done in v2)"
        continue
    fi
    wait_for_slot 1
    JID=$(sbatch --parsable --array=$CELL $SCRIPTS/sbatch_phase4_train.sh)
    SUBMITTED4C+=($JID)
    echo "$(date -Is) cell=$CELL jid=$JID"
done
# Wait for all P4c
if [ ${#SUBMITTED4C[@]} -gt 0 ]; then
    IDS=$(IFS=,; echo "${SUBMITTED4C[*]}")
    wait_jobs "$IDS"
fi

echo "=== orch_recompute DONE $(date -Is) ==="
exit 0
