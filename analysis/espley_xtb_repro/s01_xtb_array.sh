#!/bin/bash
# Array element = one slice of accepted rxns (5,260 / 20 = ~263 per slice).
# Runs 5 GFN2-xTB SPEs per rxn on Coley DFT geometries (no re-optimisation).
#SBATCH --job-name=xtb_espley
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-19%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/xtb_slice_%a.%j.out

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

CODE=/home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
OUT=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/slices/slice_$(printf '%02d' ${SLURM_ARRAY_TASK_ID}).parquet

python "$CODE/xtb_slice.py" ${SLURM_ARRAY_TASK_ID} 20 "$OUT"
