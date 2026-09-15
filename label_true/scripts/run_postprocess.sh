#!/bin/bash
#SBATCH --job-name=lbl_post
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/label_true_logs/postprocess.%j.out
set -euo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/label_true_logs
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/label_true
python scripts/postprocess_labels_all.py
python - <<'PY'
import json, statistics
data = json.load(open('/home1/yeseo1ee/projects/eda-asm-prediction/labels_all.json'))
from collections import Counter
print('final status counts:', dict(Counter(r['status'] for r in data)))
ok = [r for r in data if r['status']=='ok']
def eight_sum_minus_barrier(r):
    return (r['d1_kcal']+r['d2_kcal']+r['elst_dft']+r['pauli_dft']+r['oi_dft']+r['disp_dft']+r['cpcm_dft']+r['cds_dft']) - r['barrier_kcal']
res = [abs(eight_sum_minus_barrier(r)) for r in ok]
gap = [abs(r['eint_spe_minus_bond_kcal']) for r in ok]
idr = [abs(r['identity_residual_kcal']) for r in ok]
print(f'ok={len(ok)}')
print(f'  max |8ch sum - barrier|     = {max(res):.6f} kcal/mol')
print(f'  max |eint_spe - e_bond|     = {max(gap):.6e} kcal/mol')
print(f'  max |ASM identity residual| = {max(idr):.6e} kcal/mol')
print(f'  n rxns with |8ch sum - barrier| > 0.02  = {sum(1 for r in res if r > 0.02)}')
print(f'  n rxns with |8ch sum - barrier| > 1.0   = {sum(1 for r in res if r > 1.0)}')
PY
