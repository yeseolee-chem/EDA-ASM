#!/bin/bash
# One-shot: retry rxn_1538 with 3 escalating strategies until success.
#
#SBATCH --job-name=strain_1538
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/rxn1538.%j.log

set -uo pipefail
module load openmpi/4.1.5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRATCH=/tmp/strain_1538_${SLURM_JOB_ID}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
RUNNER=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts/strain_one_rxn.sh
W=$STRAIN/work/rxn_1538
RID=1538

for STRAT in slowconv2p stable_scf no_solv; do
    if [ -f "$W/strain.json" ]; then
        echo "=== rxn_1538 SUCCESS before $STRAT — done ==="
        break
    fi
    echo "=== rxn_1538 trying $STRAT at $(date -Is) ==="
    # wipe partial state
    rmdir "$W/.lock" 2>/dev/null || true
    find "$W" -maxdepth 1 -type f -delete 2>/dev/null
    export STRAIN_RETRY_STRATEGY=$STRAT
    bash $RUNNER $RID
    RC=$?
    echo "=== rxn_1538 $STRAT rc=$RC at $(date -Is) ==="
done

if [ -f "$W/strain.json" ]; then
    echo "=== FINAL: rxn_1538 SUCCESS ==="
    cat "$W/strain.json"
else
    echo "=== FINAL: rxn_1538 all 3 strategies exhausted, giving up ==="
fi

cd / && rm -rf $SCRATCH
exit 0
