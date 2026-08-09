#!/bin/bash
#SBATCH --job-name=A_merge
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_arm_a/merge.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro

python3 << 'PY'
import pickle
from pathlib import Path
BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_arm_a")
CHK = BASE / "sklearn/checkpoint.pkl"
SK_DIR = BASE / "sklearn_parallel"
NN_DIR = BASE / "nn_parallel"
OUT = BASE / "hps_arm_a.pkl"

hps = pickle.load(CHK.open("rb"))
print(f"initial: {len(hps)}")
for pkl in sorted(SK_DIR.glob("sk_hps_*.pkl")):
    key = pkl.stem.replace("sk_hps_", "")
    hps[key] = pickle.load(pkl.open("rb"))
print(f"after sklearn: {len(hps)}")
for pkl in sorted(NN_DIR.glob("nn_hps_*.pkl")):
    e = pickle.load(pkl.open("rb"))
    hps[e["model_name"]] = e["best_hps"]
print(f"after NN: {len(hps)}")
OUT.parent.mkdir(exist_ok=True)
pickle.dump(hps, OUT.open("wb"))
print(f"saved: {OUT}")
from collections import Counter
c = Counter()
for k in hps.keys():
    if k.startswith('2_st_nn_'): c['2L_NN'] += 1
    elif k.startswith('4_st_nn_'): c['4L_NN'] += 1
    else: c[k.split('_')[0]] += 1
for k,v in sorted(c.items()): print(f"  {k}: {v}")
PY
