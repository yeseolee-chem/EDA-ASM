#!/bin/bash
# 55-feature xTB pipeline: distances(11) + Mulliken(15) + APT_surrogate(15) +
# xTB energies(6) + GFN2-xTB channels(8). Gas-phase tblite (no ALPB).
# Before sbatch: mkdir -p /gpfs/tmp_cpu2/yeseo1ee/espley_xtb/{logs,slices}
#SBATCH --job-name=xtb_espley
#SBATCH --time=48:00:00
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
