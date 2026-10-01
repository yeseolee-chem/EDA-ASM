#!/bin/bash
#SBATCH --job-name=r6_espenv
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%j.out
# r6_espley_env.sh — rev 6 Phase 0/P: build the Espley-pinned environment for running THEIR code unchanged
# (ml_analysis.py, hyp_tuning.py incl. the TensorFlow/Keras NN and keras_tuner Hyperband).
# Pins = the-grayson-group/distortion-interaction_ML requirements.txt (keras 3.3.3, tensorflow 2.16.1, keras_tuner
# 1.4.5, scikit-learn 1.3.1, numpy 1.26.0, pandas 2.1.1, matplotlib 3.8.4, inquirer 3.2.4, pyyaml 6.0.1).
# Added (not in their file): xgboost 3.2.0 (= our pipeline's version, for the D4 XGB arm) and pyarrow (parquet I/O).
# Not installed (feature-extraction only, never imported by the ML scripts): molml, cclib, morfeus, xyz_py, ipykernel.
# Python 3.11 (their README states no Python version); PYTHONNOUSERSITE=1 keeps ~/.local (python3.10) out.
# Idempotent: skips the build when the env's version stamp already exists; the smoke test always reruns.
set -euo pipefail
ENV=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley
STAMP=$ENV/.r6_env_ok
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
echo "[r6_env] host=$(hostname) start=$(date -Is)"
if [[ ! -f $STAMP ]]; then
  rm -rf "${ENV:?}"
  conda create -y -p "$ENV" python=3.11 pip
  "$ENV/bin/python" -m pip install --no-cache-dir \
    tensorflow==2.16.1 keras==3.3.3 keras_tuner==1.4.5 scikit-learn==1.3.1 numpy==1.26.0 pandas==2.1.1 \
    matplotlib==3.8.4 inquirer==3.2.4 pyyaml==6.0.1 xgboost==3.2.0 "pyarrow<18"
  touch "$STAMP"
fi
"$ENV/bin/python" -m pip freeze > /gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley_freeze.txt
"$ENV/bin/python" - <<'EOF'
import sys, numpy, pandas, sklearn, scipy, tensorflow as tf, keras, keras_tuner, xgboost, pyarrow
from importlib.metadata import version
print("python", sys.version.split()[0])
for m in ("tensorflow", "keras", "keras_tuner", "scikit-learn", "numpy", "pandas", "scipy", "xgboost", "pyarrow",
          "matplotlib", "inquirer", "PyYAML"):
    print(m, version(m))
want = {"tensorflow": "2.16.1", "keras": "3.3.3", "keras_tuner": "1.4.5", "scikit-learn": "1.3.1",
        "numpy": "1.26.0", "pandas": "2.1.1"}
bad = {k: version(k) for k, v in want.items() if version(k) != v}
assert not bad, f"pin mismatch {bad}"
# their pickles load
import pickle
for p in ("feature_selection/_f_selection/tt/manual_tt_solvent.pkl", "machine_learning/tt/solvent/ml_results.pkl",
          "hyperparameter_tuning/tt/solvent/hps.pkl"):
    with open("/gpfs/tmp_cpu2/yeseo1ee/espley_compare/" + p, "rb") as f:
        obj = pickle.load(f)
    print("load ok", p, type(obj).__name__)
# NN determinism smoke: the same seed twice in one process (their seeding = tf.keras.utils.set_random_seed)
import numpy as np
X = np.random.default_rng(0).normal(size=(200, 10)).astype("float32"); y = X[:, 0] - 2 * X[:, 1]
def fit():
    tf.keras.utils.set_random_seed(23)
    m = keras.Sequential([keras.layers.Input((10,)), keras.layers.Dense(32, activation="elu"),
                          keras.layers.Dropout(0.2), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    m.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return m.predict(X, verbose=0).ravel()
a, b = fit(), fit()
print("nn same-seed max|diff|", float(np.abs(a - b).max()))
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
g = GridSearchCV(XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method="hist"),
                 {"n_estimators": [20]}, cv=3, scoring="neg_mean_absolute_error").fit(X, y)
print("xgb+sklearn GridSearchCV ok", g.best_score_)
EOF
echo "[r6_env] done=$(date -Is)"
