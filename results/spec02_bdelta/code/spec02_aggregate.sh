#!/bin/bash
#SBATCH --job-name=sp02_agg
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/sp02_agg_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/sp02_agg_%j.err
set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
python -u pipeline_rebuild/spec_v1/spec02_aggregate.py
