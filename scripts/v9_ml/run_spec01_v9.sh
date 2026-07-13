#!/bin/bash
#SBATCH --job-name=v9_spec01
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_spec01.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_spec01.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
export BUNDLE_PT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt
export SPLIT_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9
python -u spec/spec01_alpha/code/spec01_alpha.py
