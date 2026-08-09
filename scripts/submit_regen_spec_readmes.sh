#!/bin/bash
#SBATCH --job-name=regen_spec_readmes
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/regen_spec_readmes_%j.log
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/regen_spec_readmes_%j.log
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
echo "[submit] host=$(hostname) start=$(date -Is)"
python -u scripts/regen_spec_readmes.py
echo "[submit] end=$(date -Is)"
