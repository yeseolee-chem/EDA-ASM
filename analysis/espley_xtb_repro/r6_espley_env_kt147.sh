#!/bin/bash
#SBATCH --job-name=r6_kt147
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%j.out
# r6_espley_env_kt147.sh — user decision 2026-10-01 (after the P-1 STOP): their pinned keras_tuner 1.4.5 cannot run
# their Hyperband NN tuning under keras 3.3.3 (checkpoint "filename must end in .weights.h5"); upgrade keras_tuner ONLY
# to 1.4.7 in env_espley (--no-deps: tensorflow / keras / sklearn / numpy / pandas stay as pinned), then smoke-test a
# tiny Hyperband search with their builder (r6_nn.make_builder). Idempotent.
set -euo pipefail
ENV=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0
cd "$SLURM_SUBMIT_DIR"
"$ENV/bin/python" -m pip install --no-cache-dir --no-deps keras_tuner==1.4.7
"$ENV/bin/python" -m pip freeze > /gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley_freeze_kt147.txt
"$ENV/bin/python" - <<'EOF'
from importlib.metadata import version
for m in ("tensorflow", "keras", "keras_tuner", "scikit-learn", "numpy", "pandas"):
    print(m, version(m))
assert version("keras_tuner") == "1.4.7" and version("keras") == "3.3.3" and version("tensorflow") == "2.16.1"
import tempfile, numpy as np, pandas as pd
import r6_nn as NN
NN.set_seeds(23)
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 8)), columns=[f"x{i}" for i in range(8)])
y = pd.Series(X.x0 - 2 * X.x1 + rng.normal(scale=0.1, size=300), name="t")
import keras_tuner as kt
with tempfile.TemporaryDirectory() as d:
    b = NN.make_builder(8, 2, "t")
    tuner = kt.Hyperband(b, project_name="smoke", directory=d, objective=["mean_absolute_error", "val_mean_absolute_error"],
                         max_epochs=3, factor=3, seed=23)
    tuner.search(x=X[:240], y=y[:240], validation_data=(X[240:], y[240:]), epochs=3, verbose=0)
    print("hyperband smoke ok:", tuner.get_best_hyperparameters(1)[0].values)
EOF
