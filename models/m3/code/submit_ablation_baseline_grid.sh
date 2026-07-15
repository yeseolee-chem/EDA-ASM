#!/bin/bash
# m3 spec2 ablation: baseline_only ridge-alpha grid sweep.
# Sweeps alpha in {0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000} x 5 members, fold 0.
# All 40 fits complete in seconds on CPU.

#SBATCH --job-name=abl_m3_bgrid
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/abl_m3_bgrid_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/abl_m3_bgrid_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

mkdir -p outputs/asr_v1/phase3/subsamples
ln -sfn /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9 \
        outputs/asr_v1/phase3/subsamples/v9_all

export MODE=baseline_only
export BASELINE=xtb_geom6_plus_v2
export SUBSAMPLES_TAG=v9_all
export SIZE_FULL=626
export OUT_TAG=ablation
export FOLD=0

# 0 = OLS (no penalty); log10-spaced sweep otherwise.
for ALPHA in 0 0.001 0.01 0.1 1 10 100 1000; do
    export RIDGE_ALPHA="${ALPHA}"
    # Sanitize tag: 0.001 -> a0p001, 1 -> a1, 0 -> a0
    export RIDGE_ALPHA_TAG="a$(python -c "import sys; a=float(sys.argv[1]); s=('%g'%a).replace('.','p'); print(s)" "${ALPHA}")"
    echo "[$(date)] === baseline_only alpha=${ALPHA} tag=${RIDGE_ALPHA_TAG} ==="
    python -u models/m3/code/runner_ablation_m3.py --fold 0 --all-members --device cpu
done

echo "[$(date)] === m3 baseline_only grid done ==="
