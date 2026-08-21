#!/bin/bash
# One retry wave. Called by orchestrator with a specific strategy.
# Enumerates all missing rxns, distributes across --array slots.
#
# Env: STRAIN_RETRY_STRATEGY (slowconv2p / stable_scf / no_solv)
#
#SBATCH --job-name=strain_wave
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/wave.%A_%a.log

set -uo pipefail

module load openmpi/4.1.5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRATCH=/tmp/strain_wave_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts
RUNNER=$SCRIPTS/strain_one_rxn.sh

# STRAIN_RETRY_STRATEGY should be set by caller and inherited via --export=ALL
: "${STRAIN_RETRY_STRATEGY:?must set STRAIN_RETRY_STRATEGY}"
export STRAIN_RETRY_STRATEGY

# STRAIN_WAVE_NTASK = total tasks in this array (used to slice the missing rxns).
: "${STRAIN_WAVE_NTASK:?must set STRAIN_WAVE_NTASK}"

TASK=${SLURM_ARRAY_TASK_ID:-0}

# Enumerate rxns still missing strain.json (recompute at wave start — dynamic).
MISSING_IDS=$(python3 - <<PY
from pathlib import Path
work = Path("$STRAIN/work")
data = Path("$STRAIN/data")
for p in sorted(data.iterdir(), key=lambda x: int(x.name.split("_")[1])):
    rid = int(p.name.split("_")[1])
    if not (work / p.name / "strain.json").exists():
        print(rid)
PY
)

TOTAL_MISSING=$(echo "$MISSING_IDS" | wc -w)
echo "=== wave strategy=$STRAIN_RETRY_STRATEGY task=$TASK/$STRAIN_WAVE_NTASK  host=$(hostname)  missing=$TOTAL_MISSING  time=$(date -Is) ==="

# Round-robin slicing: this task processes index i where i % NTASK == TASK.
n_done=0
n_fail=0
idx=0
for RID in $MISSING_IDS; do
    if [ $((idx % STRAIN_WAVE_NTASK)) -eq "$TASK" ]; then
        RXN_TAG=$(printf "rxn_%04d" "$RID")
        W=$STRAIN/work/$RXN_TAG
        # Wipe partial state so strategy change actually re-runs.
        if [ ! -f "$W/strain.json" ]; then
            rmdir "$W/.lock" 2>/dev/null || true
            find "$W" -maxdepth 1 -type f -delete 2>/dev/null
        fi
        bash $RUNNER $RID
        RC=$?
        if [ $RC -eq 0 ]; then
            n_done=$((n_done+1))
        else
            n_fail=$((n_fail+1))
            echo "!! rxn_$RID wave failed rc=$RC"
        fi
    fi
    idx=$((idx+1))
done

echo "=== wave task=$TASK done=$n_done failed=$n_fail time=$(date -Is) ==="

cd / && rm -rf $SCRATCH
exit 0
