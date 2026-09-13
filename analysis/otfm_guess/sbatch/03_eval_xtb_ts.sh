#!/bin/bash
#SBATCH --job-name=guess03
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess/logs/03_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$BASE"
XTB_MODE="${XTB_MODE:-pilot}" python 03_eval_xtb_ts.py
