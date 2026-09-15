#!/bin/bash
#SBATCH --job-name=stage3
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/label_true_logs/stage3.%j.out
set -euo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/label_true_logs
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/label_true
python scripts/stage3_parse.py
python - <<'PY'
import json, statistics
data = json.load(open('work/labels_all.json'))
by_status = {}
for r in data:
    by_status.setdefault(r['status'], []).append(r)
for s, rs in sorted(by_status.items()):
    print(f'  {s:30s} {len(rs)}')
ok = by_status.get('ok', [])
if ok:
    gaps = [abs(r['eint_spe_minus_bond_kcal']) for r in ok]
    ids  = [abs(r['identity_residual_kcal']) for r in ok]
    eight = [abs((r['d1_kcal']+r['d2_kcal']+r['elst_dft']+r['pauli_dft']+r['oi_dft']+r['disp_dft']+r['cpcm_dft']+r['cds_dft']) - r['barrier_kcal']) for r in ok]
    print(f'ok={len(ok)}')
    print(f'  max |eint_spe − e_bond|     = {max(gaps):.6e} kcal/mol')
    print(f'  max |ASM identity residual| = {max(ids):.6e} kcal/mol')
    print(f'  max |8ch sum − barrier|     = {max(eight):.6e} kcal/mol')
    bsse = [r['bsse_shift_kcal'] for r in ok]
    print(f'  BSSE shift: mean {statistics.mean(bsse):.3f}  sd {statistics.stdev(bsse):.3f}  min {min(bsse):.3f}  max {max(bsse):.3f}')
PY
