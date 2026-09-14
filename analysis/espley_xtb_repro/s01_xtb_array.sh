#!/bin/bash
# Array element = one slice of the 5,260 accepted rxns (18 slices x ~293).
# 5 GFN2-xTB/ALPB(water) SPEs per rxn on Coley DFT geometries (no re-optimisation).
# Before sbatch (login node, shell only):  mkdir -p /gpfs/tmp_cpu2/yeseo1ee/espley_xtb/{logs,slices}
#SBATCH --job-name=xtb_espley
#SBATCH --time=12:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --array=0-17%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/xtb_slice_%a.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate ${ESPLEY_ENV:-reactot}
python -c "import tblite, pyarrow, networkx, numpy, pandas; print('env ok: tblite', getattr(tblite,'__version__','?'))"
CODE=/home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
OUT=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/slices/slice_$(printf '%02d' ${SLURM_ARRAY_TASK_ID}).parquet
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python "$CODE/xtb_slice.py" ${SLURM_ARRAY_TASK_ID} 18 "$OUT"
