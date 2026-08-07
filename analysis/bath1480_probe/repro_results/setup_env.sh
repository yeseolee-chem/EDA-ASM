#!/bin/bash
#SBATCH --job-name=distintml_env
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_repro/logs/env_setup.%j.log

set -euo pipefail

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh

ENV_NAME="distintml_repro"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "env $ENV_NAME exists, skipping create"
else
    conda create -y -n $ENV_NAME python=3.11
fi

conda run -n $ENV_NAME pip install \
    molml==0.9.0 \
    cclib==1.8 \
    numpy==1.26.0 \
    pandas==2.1.1 \
    scikit-learn==1.3.1 \
    matplotlib==3.8.4 \
    tensorflow==2.16.1 \
    keras==3.3.3 \
    keras_tuner==1.4.5 \
    xyz_py==5.9.3 \
    inquirer==3.2.4 \
    pyyaml==6.0.1 \
    morfeus-ml==0.7.2 \
    ipykernel==6.29.5

echo ""
echo "=== 확인 ==="
conda run -n $ENV_NAME python -c "
import tensorflow, keras, keras_tuner, sklearn, pandas, numpy, matplotlib
print(f'tensorflow {tensorflow.__version__}')
print(f'keras {keras.__version__}')
print(f'keras_tuner {keras_tuner.__version__}')
print(f'sklearn {sklearn.__version__}')
print(f'pandas {pandas.__version__}')
print(f'numpy {numpy.__version__}')
"
echo "setup done"
