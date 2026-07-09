#!/bin/bash
#SBATCH --job-name=organize_776
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/organize_776.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/organize_776.%j.err

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

# Build organized folder (outputs/final_776_v7/)
python scripts/organize_final_dataset.py

# Report uncompressed size
du -sh outputs/final_776_v7 2>&1 || true

# Create download-ready tarball in outputs/ (small enough to easily scp)
cd outputs
tar czf final_776_v7.tar.gz final_776_v7/
ls -lh final_776_v7.tar.gz
