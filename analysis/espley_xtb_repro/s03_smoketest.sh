#!/bin/bash
#SBATCH --job-name=xtb_ml_smoke
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/ml_smoke.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
python -c "
from train_ml_single import GRIDS, FEATURE_SETS, TARGETS, _make_pipe
print('TARGETS:', len(TARGETS), '->', TARGETS[-1])
print('feature counts:', {k: len(v) for k,v in FEATURE_SETS.items()})
for name, (est, grid) in GRIDS.items():
    p = _make_pipe(est)
    params = p.get_params(deep=True)
    missing = [k for k in grid if k not in params]
    print(f'{name}: unknown grid keys = {missing}')
import pandas as pd
df = pd.read_parquet('/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features.parquet')
must = {'dft_barrier_eda','dft_bsse_gap','is_charged'}
missing_cols = must - set(df.columns)
print(f'derived-cols missing in parquet: {missing_cols}')
print(f'is_charged sum: {int(df.is_charged.sum())}')
print('SMOKE_OK')
"
