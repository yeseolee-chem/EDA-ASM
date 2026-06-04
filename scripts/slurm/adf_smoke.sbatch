#!/bin/bash
#SBATCH --job-name=adf_smoke
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
cd "$REPO"

RXN_ID=${RXN_ID:-0}
REACTION_ID="smoke_dipolar_${RXN_ID}"
RUN_DIR="${REPO}/ADF_800/smoke_test/${REACTION_ID}"

echo "=== Smoke test for ${REACTION_ID} on $(hostname) at $(date -Iseconds) ==="
echo "Slurm job: ${SLURM_JOB_ID:-NOJOB}"
echo "CPUs     : ${SLURM_CPUS_PER_TASK}"

echo ""
echo "--- step 1: build inputs ---"
python scripts/adf/build_smoke_inputs.py --rxn-id "${RXN_ID}"

echo ""
echo "--- step 2: run ADF pipeline (5 SPs) ---"
cd "${RUN_DIR}"
bash run_eda.sh

echo ""
echo "--- step 3: parse EDA channels + strain ---"
cd "${REPO}"
python scripts/adf/parse_run.py "${RUN_DIR}"

echo ""
echo "=== Smoke test DONE at $(date -Iseconds) ==="
