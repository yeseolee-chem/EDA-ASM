#!/bin/bash
#SBATCH --job-name=v9_prereq
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_prereq.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_prereq.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
export BUNDLE_PT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt
export SPLIT_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9

# 1) v9 charges parquet
python -u scripts/v9_ml/build_v9_charges.py

# 2) make_folds for spec02
python -u spec/spec02_abc_ablation/code/make_folds.py
