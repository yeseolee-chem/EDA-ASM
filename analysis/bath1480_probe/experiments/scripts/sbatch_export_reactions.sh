#!/bin/bash
#SBATCH --job-name=exp_reacts
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/export_reactions.%j.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
python3 -u /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts/export_reactions_visible.py
