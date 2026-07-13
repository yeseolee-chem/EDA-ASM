#!/bin/bash
#SBATCH --job-name=v9_spec05
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_spec05.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_spec05.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
export BUNDLE_PT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt
export SPLIT_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9
# main spec5
python -u spec/spec05_d25_sum/code/spec05_2x2.py 2>&1 || echo "spec05_2x2 failed but continuing"
# spec6 integration (from_spec06)
python -u spec/spec05_d25_sum/code/from_spec06/spec06_xgb.py 2>&1 || echo "spec06_xgb failed but continuing"
