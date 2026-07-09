#!/bin/bash
#SBATCH --job-name=fix_ts
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/fix_dip_ts.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/fix_dip_ts.%j.err

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python scripts/fix_dipolar_TS.py
# Then regenerate inputs immediately
rm -rf outputs/orca_eda/inputs outputs/orca_eda/manifest.txt
mkdir -p outputs/orca_eda/inputs
python scripts/make_orca_eda_inputs.py --only-reviewed
# Refresh manifest
INPUT_ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/orca_eda/inputs
MANIFEST=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/orca_eda/manifest.txt
[ -d "$INPUT_ROOT" ] && ls -1 "$INPUT_ROOT" | sort > "$MANIFEST"
echo "manifest: $(wc -l < "$MANIFEST") entries"
