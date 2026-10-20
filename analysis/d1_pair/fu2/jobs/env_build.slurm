#!/bin/bash
#SBATCH --job-name=d1fu2_env
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/logs/%x_%j.out
# fu2/jobs/env_build.slurm — FU2 M-3: build env_d1fu2 (pure PyTorch training env), same pattern as rev 6 r6_espley_env.sh.
# Pins: python 3.11, torch 2.2.1 + CUDA 11.8 wheels (= reactot's torch/CUDA, known to run on this cluster's A6000 nodes),
# numpy 1.26.0 / pandas 2.1.1 / scikit-learn 1.3.1 (= env_espley, so GroupKFold etc. behave identically), pyarrow < 18.
# Records the result in results/fu2/env.yaml (conda export) + env_pip_freeze.txt. Idempotent: skips the build when the
# version stamp exists. Submit from analysis/d1_pair/:  sbatch -p cpu1,cpu2 fu2/jobs/env_build.slurm
set -euo pipefail
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
STAMP=$ENV/.d1fu2_env_ok
RES=$SLURM_SUBMIT_DIR/results/fu2
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_NO_CACHE_DIR=1
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
echo "[d1fu2_env] host=$(hostname) start=$(date -Is)"
if [[ ! -f $STAMP ]]; then
  rm -rf "${ENV:?}"
  conda create -y -p "$ENV" python=3.11 pip
  "$ENV/bin/python" -m pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
  "$ENV/bin/python" -m pip install --no-cache-dir numpy==1.26.0 pandas==2.1.1 scikit-learn==1.3.1 "pyarrow<18"
fi
mkdir -p "$RES"
conda env export -p "$ENV" > "$RES/env.yaml.tmp" && mv "$RES/env.yaml.tmp" "$RES/env.yaml"
"$ENV/bin/python" -m pip freeze > "$RES/env_pip_freeze.txt.tmp" && mv "$RES/env_pip_freeze.txt.tmp" "$RES/env_pip_freeze.txt"
"$ENV/bin/python" - <<'EOF'
import sys
from importlib.metadata import version
import torch, numpy, pandas, sklearn, pyarrow
print("python", sys.version.split()[0])
for m in ("torch", "numpy", "pandas", "scikit-learn", "pyarrow", "scipy"):
    print(m, version(m))
print("torch.version.cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
want = {"torch": "2.2.1+cu118", "numpy": "1.26.0", "pandas": "2.1.1", "scikit-learn": "1.3.1"}
bad = {k: version(k) for k, v in want.items() if version(k) != v}
assert not bad, f"pin mismatch {bad}"
x = torch.randn(4, 3, dtype=torch.float64)
print("cpu torch ok", float((x @ x.T).trace()))
EOF
touch "$STAMP"                                    # only after the pin check passed (set -e)
echo "[d1fu2_env] done $(date -Is)"
