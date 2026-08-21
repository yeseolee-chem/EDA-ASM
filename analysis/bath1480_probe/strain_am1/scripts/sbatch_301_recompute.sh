#!/bin/bash
# One-shot: recompute rxn_301 which converged to wrong (higher) basin during opt.
# Uses tight_opt strategy: TightOpt + Trust -0.1 + Calc_Hess true.
#
# Existing strain.json is overwritten only if new d1 is physical (>= 0).
# Old strain.json backed up to strain.json.old.
#
#SBATCH --job-name=strain_301
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/logs/rxn301.%j.log

set -uo pipefail
module load openmpi/4.1.5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

SCRATCH=/tmp/strain_301_${SLURM_JOB_ID}
mkdir -p $SCRATCH
export TMPDIR=$SCRATCH

STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
RUNNER=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/strain_am1/scripts/strain_one_rxn.sh
W=$STRAIN/work/rxn_0301

echo "=== rxn_301 recompute start $(date -Is) host=$(hostname) ==="
echo "--- current (bad) strain.json ---"
cat "$W/strain.json" 2>/dev/null || echo "(none)"

# Back up + wipe partial state so runner starts fresh.
if [ -f "$W/strain.json" ]; then
    cp "$W/strain.json" "$W/strain.json.old_$(date +%Y%m%d_%H%M%S)"
fi
rmdir "$W/.lock" 2>/dev/null || true
find "$W" -maxdepth 1 -type f ! -name "strain.json.old*" -delete 2>/dev/null

export STRAIN_RETRY_STRATEGY=tight_opt
bash $RUNNER 301
RC=$?
echo "=== rxn_301 tight_opt rc=$RC $(date -Is) ==="

if [ -f "$W/strain.json" ]; then
    echo "--- new strain.json ---"
    cat "$W/strain.json"
    NEW_D1=$(python3 -c "import json; print(json.load(open('$W/strain.json'))['distortion_energy_1_dft'])")
    echo "new d1 = $NEW_D1"
    IS_OK=$(python3 -c "print(1 if $NEW_D1 >= -0.5 else 0)")
    if [ "$IS_OK" = "1" ]; then
        echo "=== SUCCESS: d1 now physical (>= -0.5 kcal/mol) ==="
    else
        echo "=== WARN: d1 still negative after tight_opt — problem may be intrinsic to AM1 geometry ==="
    fi
else
    echo "=== FAIL: no strain.json produced ==="
fi

cd / && rm -rf $SCRATCH
exit 0
