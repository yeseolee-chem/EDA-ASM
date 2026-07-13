#!/bin/bash
# Assemble labels_v9_5channel.parquet from primary + helper SP outputs.
# Dependency-launched: fires once both SP arrays finish (afterany).

#SBATCH --job-name=v9_asm
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review/logs/assemble_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review/logs/assemble_%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
V9=$REPO/outputs/v9_review

echo "[$(date +%H:%M:%S)] === assemble v9 labels ==="
python $REPO/scripts/v9/assemble_labels_v9.py

echo
echo "[$(date +%H:%M:%S)] === completeness / failure summary ==="
python - <<'PY'
from pathlib import Path
import pandas as pd
V9 = Path('/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v9_review')
LBL = Path('/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v8_review/labels/labels_v9_5channel.parquet')

def done_out(p):
    if not p.exists(): return False
    return 'ORCA TERMINATED NORMALLY' in p.read_text(errors='ignore')

df = pd.read_parquet(LBL)
n_ok  = df.strain_kcal.notna().sum()
n_nan = df.strain_kcal.isna().sum()
print(f'labels rows: {len(df)}   complete: {n_ok}   NaN: {n_nan}')

# List rids still missing R-SPs after both dirs
missing = []
for _, r in df[df.strain_kcal.isna()].iterrows():
    rid = r.reaction_id
    a_prim = V9/'strain_sp_cp'/rid/'fragA_R.out'
    a_help = V9/'strain_sp_helper'/rid/'fragA_R.out'
    b_prim = V9/'strain_sp_cp'/rid/'fragB_R.out'
    b_help = V9/'strain_sp_helper'/rid/'fragB_R.out'
    a_done = done_out(a_prim) or done_out(a_help)
    b_done = done_out(b_prim) or done_out(b_help)
    missing.append((rid, a_done, b_done))
if missing:
    print()
    print(f'--- {len(missing)} rows still missing R-SPs after both dirs ---')
    for rid, a, b in missing[:20]:
        print(f'  {rid}  fragA={\"OK\" if a else \"MISSING\"}  fragB={\"OK\" if b else \"MISSING\"}')
PY

echo
echo "[$(date +%H:%M:%S)] === done ==="
