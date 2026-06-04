#!/bin/bash
# ASR ADF run for reaction qmrxn20_sn2_A_E_A_D_B_B
# Per ASR_ADF_Computation_Spec_v1.0 §7.1 (single-job C1-C5 chain).
# C4/C5 backgrounded; C1, C2 sequential; C3 after C1+C2; wait for C4/C5.
#SBATCH --job-name=asr_qmrxn20_sn2_A_E_A_D_B_B
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm.log
#SBATCH --error=slurm.err

# NO `set -e` — we want all 5 calcs to attempt, then write_status to record
# what passed and what failed. Failures should not abort the chain.
set -o pipefail
source /home1/yeseo1ee/ams2026.103/amsbashrc.sh

cd "$(dirname "$0")"

export NSCM=${SLURM_NTASKS:-1}
ulimit -s unlimited 2>/dev/null || true

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== qmrxn20_sn2_A_E_A_D_B_B START $START_UTC host=$(hostname) NSCM=$NSCM ==="

# C4 + C5: fragment geometry optimization (independent, backgrounded)
# C4 skipped: fragA is a single atom (strain_a = 0 by definition)
AMS_JOBNAME=c5_fragB_opt "$AMSBIN/ams" <c5_fragB_opt.in > c5_fragB_opt.out 2>&1 &
PID_C5=$!

# C1, C2: fragment SP at TS geometry (sequential — share gate1 license slot)
AMS_JOBNAME=c1_fragA_ts "$AMSBIN/ams" <c1_fragA_ts.in > c1_fragA_ts.out 2>&1
C1_RC=$?
AMS_JOBNAME=c2_fragB_ts "$AMSBIN/ams" <c2_fragB_ts.in > c2_fragB_ts.out 2>&1
C2_RC=$?

# C3: EDA, depends on C1 + C2 rkfs
if [[ $C1_RC -eq 0 && $C2_RC -eq 0 ]]; then
    AMS_JOBNAME=c3_eda "$AMSBIN/ams" <c3_eda.in > c3_eda.out 2>&1
    C3_RC=$?
else
    echo "skipping C3 (C1 rc=$C1_RC, C2 rc=$C2_RC)" >&2
    C3_RC=2
fi

# Wait for the background optimizations
# (no C4 to wait on)
wait ${PID_C5:-1} 2>/dev/null || true

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== qmrxn20_sn2_A_E_A_D_B_B END $END_UTC ==="

# Generate status.json with calc statuses derived from output parsing
python /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/scripts/adf/write_status.py \
    --rid "qmrxn20_sn2_A_E_A_D_B_B" \
    --rxn-dir "$(pwd)" \
    --start "$START_UTC" \
    --end "$END_UTC" \
    --functional "BLYP-D3(BJ)" \
    --basis "TZ2P" \
    --frag-method "smarts" \
    --atoms-a '[2]' \
    --atoms-b '[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]' \
    --charge-a -1 \
    --charge-b 0 \
    --mult-a 1 \
    --mult-b 1 \
    --total-charge -1 \
    --dataset-delta-Ea 13.265342563524602 \
    --atom-permutation '[2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]' \
    --single-atom-a 1 \
    --single-atom-b 0 \
    || true   # write_status exit code is informational; don't abort the .sh

# Cleanup binary results dirs (kept .out and .in for audit + re-parse).
# Saves ~100MB per reaction; ~80GB across 794 reactions.
find . -maxdepth 1 -type d -name '*.results' -exec rm -rf {} + 2>/dev/null || true

echo "=== qmrxn20_sn2_A_E_A_D_B_B cleanup done ==="
