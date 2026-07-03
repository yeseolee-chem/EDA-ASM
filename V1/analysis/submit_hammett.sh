#!/bin/bash
#SBATCH --job-name=v1_hammett
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=V1/analysis/logs/hammett_%j.out
#SBATCH --error=V1/analysis/logs/hammett_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p V1/analysis/logs V1/analysis/figures V1/analysis/results

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u V1/analysis/hammett_plot.py
