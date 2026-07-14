#!/bin/bash
#SBATCH --job-name=replot_notitle
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/replot_notitle.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/replot_notitle.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

export BUNDLE_PT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt
export SPLIT_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9

echo "[$(date +%H:%M:%S)] === spec01 ==="
python -u spec/spec01_alpha/code/spec01_alpha.py 2>&1 || echo "  spec01 FAILED"

echo "[$(date +%H:%M:%S)] === spec02 aggregate ==="
python -u spec/spec02_abc_ablation/code/aggregate.py 2>&1 || echo "  spec02 FAILED"

echo "[$(date +%H:%M:%S)] === spec03 ==="
python -u spec/spec03_bmax/code/spec03_bmax.py 2>&1 || echo "  spec03 FAILED"

echo "[$(date +%H:%M:%S)] === spec04 ==="
python -u spec/spec04_descriptors/code/spec04_xgb_importance.py 2>&1 || echo "  spec04 FAILED"

echo "[$(date +%H:%M:%S)] === spec05 2x2 ==="
python -u spec/spec05_d25_sum/code/spec05_2x2.py 2>&1 || echo "  spec05_2x2 FAILED"

echo "[$(date +%H:%M:%S)] === spec05 from_spec06 xgb (channel proxies) ==="
python -u spec/spec05_d25_sum/code/from_spec06/spec06_xgb.py 2>&1 || echo "  spec06_xgb FAILED"

echo "[$(date +%H:%M:%S)] === spec05 compare_2way ==="
python -u scripts/v9_ml/spec5_compare_2way.py 2>&1 || echo "  compare_2way FAILED"

echo "[$(date +%H:%M:%S)] === m3 regen ==="
python -u scripts/v9_ml/regen_m3_figures_v9.py 2>&1 || echo "  m3 regen FAILED"

echo "[$(date +%H:%M:%S)] DONE"
