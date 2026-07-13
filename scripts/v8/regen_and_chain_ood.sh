#!/bin/bash
#SBATCH --job-name=chain_ood
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/chain_ood.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/chain_ood.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

# Regenerate inputs
python -u scripts/v8/regen_ood_tight.py

# Build combined manifest
python3 -c "
from pathlib import Path
V8 = Path('outputs/v8_review')
lines = []
for r in Path(V8/'orca_inputs/manifest_ood_retry_eda.txt').read_text().splitlines():
    r = r.strip()
    if r: lines.append(f'EDA {r}')
for r in Path(V8/'strain_sp/manifest_ood_retry_sp.txt').read_text().splitlines():
    r = r.strip()
    if r: lines.append(f'SP {r}')
(V8/'manifest_ood_parallel.txt').write_text('\n'.join(lines) + '\n')
print(f'combined manifest: {len(lines)} lines')
"

# Submit parallel array + trigger (submit-limit should be freed by now)
PAR=$(sbatch --parsable scripts/v8/rerun_ood_parallel.sh)
echo "[chain] parallel array submitted: $PAR"
TRIG=$(sbatch --dependency=afterany:$PAR --parsable scripts/v8/trigger_finalize.sh)
echo "[chain] trigger_finalize submitted: $TRIG (dep afterany:$PAR)"
