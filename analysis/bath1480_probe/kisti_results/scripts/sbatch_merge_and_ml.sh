#!/bin/bash
# Phase D+E: merge sklearn+NN hps → final hps.pkl, then queue ml_analysis 5-seed
#SBATCH --job-name=merge_ml
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_merge.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results

echo "=== Merge sklearn (checkpoint.pkl) + NN (per-target pkls) → hps_7ch.pkl ==="
python3 << 'PY'
import pickle
from pathlib import Path

SK_CHK = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_7ch/sklearn/checkpoint.pkl")
NN_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_7ch/nn_parallel")
OUT    = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_7ch/hps_7ch.pkl")

hps = pickle.load(SK_CHK.open("rb"))
print(f"sklearn models: {len(hps)}")

for pkl in sorted(NN_DIR.glob("nn_hps_*.pkl")):
    e = pickle.load(pkl.open("rb"))
    hps[e["model_name"]] = e["best_hps"]  # Espley raw-dict format for NNs

print(f"total models after NN merge: {len(hps)}")
OUT.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(hps, OUT.open("wb"))
print(f"saved: {OUT}")
PY

echo ""
echo "=== Submit ml_analysis 5-seed array (dependency chain end) ==="
JID=$(sbatch --parsable /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/sbatch_ml_array.sh)
echo "ml_analysis array JID: $JID"
