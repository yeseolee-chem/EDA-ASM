#!/bin/bash
# Orchestrator: waits for the final wave to finish, then runs 3 escalating
# retry waves against any rxns still missing strain.json.
#
# Runs on a compute node (NOT login), per CLAUDE.md rule that launchers
# must not live on gate1.hpc.
#
# Each wave uses `sbatch --wait` which blocks until the submitted array
# finishes; the orchestrator only proceeds to the next wave when the
# previous one is done and there are still missing rxns.
#
# Strategies (escalating):
#   1. slowconv2p   → nprocs 2 + SlowConv + MaxIter 500  (MPI race avoidance)
#   2. stable_scf   → above + %scf DirectResetFreq 1 DIISMaxEq 10
#   3. no_solv      → above + drop CPCM  (labels lose solvation; last-resort)
#
# NOTE: no_solv labels aren't directly comparable to the main set — they're
# the "at least some d1/d2 value" fallback for intractable rxns.
#
#SBATCH --job-name=strain_orchestrator
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/orchestrator.%j.log

set -uo pipefail

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts
WAVE=$SCRIPTS/sbatch_retry_wave.sh

count_missing() {
    python3 - <<'PY'
from pathlib import Path
w = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/work")
d = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/data")
n = sum(1 for p in d.iterdir()
        if not (w / p.name / "strain.json").exists())
print(n)
PY
}

echo "=== orchestrator start $(date -Is) host=$(hostname) ==="
echo "=== initial missing: $(count_missing) ==="

# Each wave targets ≤10 tasks (respects the QoS 10-running cap when the
# orchestrator has already used 1 slot).
# Split across cpu1/cpu2 to spread node load.
run_wave() {
    local STRAT=$1
    local NTASK=$2
    local MISSING=$(count_missing)
    if [ "$MISSING" -eq 0 ]; then
        echo "=== ALL DONE — no rxns missing before wave $STRAT ==="
        return 0
    fi
    echo "=== wave $STRAT starting: $MISSING rxns missing ==="

    # Split evenly: half to cpu1, half to cpu2.
    local HALF=$((NTASK / 2))
    export STRAIN_RETRY_STRATEGY=$STRAT
    export STRAIN_WAVE_NTASK=$NTASK

    local J1 J2
    J1=$(sbatch --parsable --export=ALL --partition=cpu1 \
                --array=0-$((HALF - 1)) "$WAVE")
    J2=$(sbatch --parsable --export=ALL --partition=cpu2 \
                --array=$HALF-$((NTASK - 1)) "$WAVE")
    echo "=== wave $STRAT: cpu1 jid=$J1  cpu2 jid=$J2 ==="

    # Wait for both waves. Poll every 5 min; safe on compute node.
    while true; do
        RUNNING=$(squeue -j "${J1},${J2}" -h 2>/dev/null | wc -l)
        if [ "$RUNNING" -eq 0 ]; then
            break
        fi
        sleep 300
    done
    echo "=== wave $STRAT done at $(date -Is); missing now: $(count_missing) ==="
}

# Escalating strategies. Adjust NTASK to fit under the 10-running SLURM cap
# (orchestrator itself uses 1 slot → up to 9 wave tasks concurrent).
# We use 8 tasks per wave (4 cpu1 + 4 cpu2), staying comfortably under.
run_wave slowconv2p 8
run_wave stable_scf 8
run_wave no_solv    8

echo "=== orchestrator end $(date -Is) ==="
echo "=== final missing: $(count_missing) ==="

exit 0
