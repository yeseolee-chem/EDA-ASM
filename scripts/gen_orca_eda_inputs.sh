#!/bin/bash
#SBATCH --job-name=orca_eda_inp
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_eda_inp.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_eda_inp.%j.err

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python scripts/make_orca_eda_inputs.py --only-reviewed

# Refresh the manifest so the ORCA array picks up the new input tree.
INPUT_ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/orca_eda/inputs
MANIFEST=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/orca_eda/manifest.txt
mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/orca_eda
if [ -d "$INPUT_ROOT" ]; then
  ls -1 "$INPUT_ROOT" | sort > "$MANIFEST"
  echo "manifest: $(wc -l < "$MANIFEST") entries → $MANIFEST"
fi
