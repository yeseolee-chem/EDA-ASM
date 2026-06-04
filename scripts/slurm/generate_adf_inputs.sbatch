#!/bin/bash
#SBATCH --job-name=adf_gen
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/adf_gen_%j.out
#SBATCH --error=logs/adf_gen_%j.err

set -eo pipefail
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
cd "$REPO"

echo "=== generate ADF C1-C5 inputs on $(hostname) at $(date -Iseconds) ==="
python scripts/adf/generate_adf_inputs.py --config configs/adf_computation.yaml --force
echo "=== done at $(date -Iseconds) ==="
