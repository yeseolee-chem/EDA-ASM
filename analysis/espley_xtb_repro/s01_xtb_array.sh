#!/bin/bash
# Array element = one slice of the 5,260 accepted rxns (18 slices x ~293).
# Per rxn: 5 xtb single points (GFN2 + ALPB water; energies + charges + Wiberg valences +
#          gap + dipole from one run each) + 3 D3(BJ) dispersion + 3 SASA.  ~0.25 s/rxn, 1 core.
# Before sbatch (login node, shell only):
#   mkdir -p /gpfs/tmp_cpu2/yeseo1ee/espley_xtb/{logs,slices}
#SBATCH --job-name=xtb_espley
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --array=0-17%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/xtb_slice_%a.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate ${ESPLEY_ENV:-reactot}

export XTB_BIN=${XTB_BIN:-/home1/yeseo1ee/xtb-dist/bin/xtb}
export XTBPATH=${XTBPATH:-/home1/yeseo1ee/xtb-dist/share/xtb}
export XTBHOME=${XTBHOME:-/home1/yeseo1ee/xtb-dist}
export XTB_THREADS=1
export OMP_NUM_THREADS=1

python -c "import pyarrow, networkx, numpy, pandas, dftd3, morfeus; print('py env ok')"
"$XTB_BIN" --version | head -2

CODE=/home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
OUT=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/slices/slice_$(printf '%02d' ${SLURM_ARRAY_TASK_ID}).parquet
python "$CODE/xtb_slice.py" ${SLURM_ARRAY_TASK_ID} 18 "$OUT"
