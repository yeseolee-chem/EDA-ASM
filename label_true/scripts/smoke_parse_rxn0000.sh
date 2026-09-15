#!/bin/bash
#SBATCH --job-name=lbl_smoke
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/label_true_logs/smoke.%j.out
set -euo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/label_true_logs
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/label_true
python - <<'PY'
import sys, pandas as pd
sys.path.insert(0, 'scripts')
from stage3_parse import parse_one_rxn
work = 'work'
row = pd.read_csv(f'{work}/input_meta.csv').set_index('rxn_id').loc[0]
row['rxn_id'] = 0
from pathlib import Path
d = parse_one_rxn(Path(f'{work}/inputs/rxn_0000'), row)
for k in ('status','d1_kcal','d2_kcal','eint_spe_kcal','barrier_kcal','e_bond_kcal',
         'eint_spe_minus_bond_kcal','identity_residual_kcal','bsse_shift_kcal',
         'elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft'):
    print(f'  {k:32s} = {d.get(k)}')
eight = d['d1_kcal']+d['d2_kcal']+d['elst_dft']+d['pauli_dft']+d['oi_dft']+d['disp_dft']+d['cpcm_dft']+d['cds_dft']
print(f'  8-channel sum                    = {eight:.6f}')
print(f'  8-channel sum - barrier          = {eight - d["barrier_kcal"]:.6f}   (should be near 0)')
PY
