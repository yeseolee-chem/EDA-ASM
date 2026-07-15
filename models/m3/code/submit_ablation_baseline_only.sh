#!/bin/bash
# m3 spec2 ablation: baseline_only (ridge only, no ML head).
# One CPU task loops members 0..4 (each run is seconds).

#SBATCH --job-name=abl_m3_bo
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/abl_m3_bo_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/abl_m3_bo_%j.err

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

echo "[$(date)] === m3 ablation baseline_only (5 members, fold 0) start ==="

python -u models/m3/code/runner_ablation_m3.py --fold 0 --all-members --device cpu

echo "[$(date)] === m3 ablation baseline_only done ==="
