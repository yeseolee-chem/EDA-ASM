#!/bin/bash
# Phase 1: install xtb binary + dftd3 python (D3BJ oracle) into reactot env
#SBATCH --job-name=inst_xtb
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/install_xtb.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== Before install ==="
which xtb 2>&1 || echo "xtb: not found"
python -c "import dftd3" 2>&1 || echo "dftd3: not found"
echo ""
echo "=== conda install xtb (idempotent) ==="
conda install -y -c conda-forge xtb 2>&1 | tail -20
echo ""
echo "=== pip install dftd3 (D3BJ) ==="
pip install --no-cache-dir dftd3 2>&1 | tail -5
echo ""
echo "=== After install ==="
which xtb
xtb --version 2>&1 | head -3
python -c "from dftd3.interface import RationalDampingParam, DispersionModel; print('dftd3 OK')"
echo ""
echo "=== quick xtb sanity ==="
mkdir -p /tmp/xtb_test_$$
cat > /tmp/xtb_test_$$/h2o.xyz << EOF
3
water
O 0.0 0.0 0.0
H 0.96 0.0 0.0
H -0.24 0.93 0.0
EOF
xtb /tmp/xtb_test_$$/h2o.xyz --gfn 2 --sp 2>&1 | grep -E "TOTAL ENERGY|electronic energy" | head -3
rm -rf /tmp/xtb_test_$$
echo "OK"
