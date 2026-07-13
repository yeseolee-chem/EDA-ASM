#!/bin/bash
#SBATCH --job-name=orca_2784
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_2784.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_2784.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
if [ -f "$HOME/orca6/orca-env.sh" ]; then source "$HOME/orca6/orca-env.sh"; fi

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
DIR="$REPO/outputs/v8_review/orca_inputs/dipolar_002784"

cd "$REPO"
python -u scripts/v8/fix_002784.py

cd "$DIR"
echo "[$(date)] START dipolar_002784 (aggressive SCF+NOCV)"
"$HOME/orca_6_1_1_avx2/orca" eda.inp > eda.out 2> eda.err
if grep -q "ORCA TERMINATED NORMALLY" eda.out; then
  echo "[$(date)] OK"
  rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null
else
  echo "[$(date)] STILL FAIL"
  tail -30 eda.out
fi
