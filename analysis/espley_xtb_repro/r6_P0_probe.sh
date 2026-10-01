#!/bin/bash
#SBATCH --job-name=r6_P0probe
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%j.out
# r6_P0_probe.sh — rev 6 Phase P preparation (read-only): structure of Espley's stored hps.pkl / ml_results.pkl /
# ds3 dataset as their code reads them (pd.read_pickle / pickle.load) in the Espley-pinned env. No training.
set -euo pipefail
export PYTHONNOUSERSITE=1
ENV=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley
"$ENV/bin/python" - <<'EOF'
import pickle, numpy as np, pandas as pd
B = "/gpfs/tmp_cpu2/yeseo1ee/espley_compare/"
ds = pd.read_pickle(B + "feature_selection/_f_selection/tt/manual_tt_solvent.pkl")
print("ds shape", ds.shape, "index", type(ds.index).__name__, ds.index[:5].tolist(), "...", ds.index[-3:].tolist())
print("ds columns", list(ds.columns))
with open(B + "hyperparameter_tuning/tt/solvent/hps.pkl", "rb") as f:
    hps = pickle.load(f)
print("hps keys", list(hps.keys()))
for k, v in hps.items():
    print(" ", k, v)
ml = pd.read_pickle(B + "machine_learning/tt/solvent/ml_results.pkl")
print("ml_results shape", ml.shape)
print("ml columns", list(ml.columns))
for _, r in ml.iterrows():
    rs = r["random_state"]
    v = r["y_test_pred_values"]
    lens = [len(np.ravel(np.asarray(x[0] if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (list, np.ndarray)) else x))) for x in v]
    print(" ", r["model_target"], "rs", rs, "n_pred", lens, "y_test_pred", r.get("y_test_pred"))
EOF
