#!/bin/bash
#SBATCH --job-name=merge_ml
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/merge_ml.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results

echo "=== Merge sklearn (checkpoint + per-target) + NN per-target → hps_7ch.pkl ==="
python3 << 'PY'
import pickle
from pathlib import Path

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results")
CHK = BASE / "pipeline_7ch/sklearn/checkpoint.pkl"
SK_DIR = BASE / "pipeline_7ch/sklearn_parallel"
NN_DIR = BASE / "pipeline_7ch/nn_parallel"
OUT = BASE / "pipeline_7ch/hps_7ch.pkl"

hps = pickle.load(CHK.open("rb"))
print(f"initial checkpoint: {len(hps)} models")

for pkl in sorted(SK_DIR.glob("sk_hps_*.pkl")):
    entry = pickle.load(pkl.open("rb"))  # {'b_hps': {...}, 'train_mae': ...}
    # filename: sk_hps_MODEL_TARGET.pkl → key: MODEL_TARGET
    key = pkl.stem.replace("sk_hps_", "")
    hps[key] = entry
print(f"after sklearn per-target: {len(hps)} models")

for pkl in sorted(NN_DIR.glob("nn_hps_*.pkl")):
    e = pickle.load(pkl.open("rb"))
    hps[e["model_name"]] = e["best_hps"]
print(f"after NN per-target: {len(hps)} models")

from collections import Counter
c = Counter()
for k in hps.keys():
    if k.startswith('2_st_nn_'): c['2L_NN'] += 1
    elif k.startswith('4_st_nn_'): c['4L_NN'] += 1
    else: c[k.split('_')[0]] += 1
print("\nbreakdown:")
for k, v in sorted(c.items()): print(f"  {k}: {v}")

OUT.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(hps, OUT.open("wb"))
print(f"\nsaved: {OUT}")
PY
