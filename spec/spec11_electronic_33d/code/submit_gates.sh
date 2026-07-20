#!/bin/bash
# SPEC_11 Gate-A + Gate-B - run under sbatch (CLAUDE.md: no python on login).
# Gate-A is pure numpy; Gate-B needs tblite (loads on cpu2).
#SBATCH --job-name=s11_gates
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_gates_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_gates_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "==== Gate-A: multipole units ===="
python -u spec/spec11_electronic_33d/code/test_multipole_units.py
GA=$?

echo ""
echo "==== Gate-B: AO block structure + tblite probe ===="
python -u spec/spec11_electronic_33d/code/test_ao_blocks.py
GB=$?

echo ""
echo "Gate-A exit=$GA  Gate-B exit=$GB"
exit $(( GA + GB ))
