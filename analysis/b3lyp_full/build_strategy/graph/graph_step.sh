#!/bin/bash
#SBATCH --job-name=fraggraph
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/build_strategy/graph/step.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export EDA_BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full
export EDA_PROF=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles
export EDA_CSV=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv
export EDA_REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
G=$EDA_BASE/build_strategy/graph
cd "$G"

echo "=== ENV ==="
python --version
python -c "import numpy, pandas, networkx, rdkit; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('networkx', networkx.__version__); print('rdkit', rdkit.__version__)" 2>&1 || {
  echo ">> networkx or rdkit missing — installing networkx via pip"
  pip install --quiet "networkx>=3.3"
  python -c "import numpy, pandas, networkx, rdkit; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('networkx', networkx.__version__); print('rdkit', rdkit.__version__)"
}

echo
echo "=== STEP 1: test_fragmenter.py ==="
python test_fragmenter.py

echo
echo "=== STEP 2: build_inputs_graph.py --dry-run ==="
python build_inputs_graph.py --dry-run

echo
echo "=== STEP 3: audit_extra.py ==="
python audit_extra.py

echo
echo "=== ALL STEPS COMPLETE ==="
