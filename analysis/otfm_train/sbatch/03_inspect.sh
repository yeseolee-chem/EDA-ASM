#!/bin/bash
#SBATCH --job-name=otfm03
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/03_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# torch.load unpickles reactot.* classes from the checkpoint — react-ot
# source tree must be on PYTHONPATH. Script also sys.path.inserts as
# defense-in-depth.
export PYTHONPATH=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/external/react-ot:${PYTHONPATH:-}

cd "$BASE"
python 03_inspect.py
