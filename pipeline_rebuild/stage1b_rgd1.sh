#!/bin/bash
#SBATCH --job-name=st1b_rgd1
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline_rebuild/logs/st1b_%j.out
#SBATCH --error=pipeline_rebuild/logs/st1b_%j.err

# Stage 1b — download RGD1-CHNO dataset from figshare and pull XYZ triples
# (R, TS, P) for the 200 rgd1 reactions in labels/adf_labels_v6_multifamily.parquet.
#
# Zhao & Savoie 2023, Scientific Data 10:145 — DOI 10.6084/m9.figshare.21066901.v6
# HDF5 layout per reaction key "MR_X_Y":
#   R_E/R_H/R_F (Hartree), P_E/P_H/P_F, TS_E/TS_H/TS_F  (energies)
#   Rsmiles, Psmiles                                    (atom-mapped SMILES)
#   elements                                            (int atomic numbers)
#   RG (Å), PG (Å), TSG (Å)                             (geometries)

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

RAW=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw
mkdir -p "$RAW/rgd1" pipeline_rebuild/logs pipeline_rebuild/results
cd "$RAW/rgd1"

if [ ! -f RGD1_CHNO.h5 ]; then
  echo "[$(date)] downloading RGD1 figshare listing..."
  curl -L -o listing.json "https://api.figshare.com/v2/articles/21066901/files"
  python3 -c "
import json
with open('listing.json') as f: d=json.load(f)
for e in d:
    print(e['id'], e['name'], e['size'], e['download_url'])
" > listing.txt
  cat listing.txt
  echo "---"
  # Grab RGD1_CHNO.h5 + RGD1CHNO_smiles.csv (+ any parse script)
  for name in RGD1_CHNO.h5 RGD1CHNO_smiles.csv; do
    URL=$(python3 -c "
import json
with open('listing.json') as f: d=json.load(f)
for e in d:
    if e['name'] == '$name':
        print(e['download_url']); break
")
    if [ -n "$URL" ]; then
      echo "[$(date)] downloading $name ..."
      curl -L -o "$name" "$URL"
      ls -la "$name"
    else
      echo "!! could not resolve $name in listing"
    fi
  done
fi

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
echo "[$(date)] extracting the 200 rgd1 R/TS/P xyz triples..."
python3 - <<'PY'
import h5py, numpy as np, pandas as pd, json
from pathlib import Path

RAW = Path('/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/rgd1')
H5 = RAW / 'RGD1_CHNO.h5'
OUT = Path('/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/rgd1/extracted_xyz')
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet('labels/adf/adf_labels_v6_multifamily.parquet')
rgd1_rids = df[df.family == 'rgd1'].reaction_id.tolist()
print(f'need {len(rgd1_rids)} rgd1 rids from RGD1_CHNO.h5')

NUM2SYM = {1:'H',6:'C',7:'N',8:'O',9:'F'}

def write_xyz(path, elements, coords, comment=''):
    with open(path, 'w') as f:
        f.write(f'{len(elements)}\n{comment}\n')
        for e, xyz in zip(elements, coords):
            sym = NUM2SYM.get(int(e), 'X')
            f.write(f'{sym:>2s}  {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}\n')

if not H5.exists():
    print(f'!! {H5} missing — did figshare download work?')
    raise SystemExit(1)

got = 0
missing = []
with h5py.File(str(H5), 'r') as hf:
    all_keys = set(hf.keys())
    print(f'HDF5 has {len(all_keys)} reactions total')
    for rid in rgd1_rids:
        key = rid.replace('rgd1_', '', 1)   # 'rgd1_MR_100076_1' -> 'MR_100076_1'
        if key not in hf:
            missing.append(rid); continue
        rxn = hf[key]
        elements = np.array(rxn['elements'])
        rid_dir = OUT / rid
        rid_dir.mkdir(exist_ok=True)
        write_xyz(rid_dir / 'R.xyz', elements, np.array(rxn['RG']), rid + ' reactant')
        write_xyz(rid_dir / 'TS.xyz', elements, np.array(rxn['TSG']), rid + ' TS')
        write_xyz(rid_dir / 'P.xyz', elements, np.array(rxn['PG']), rid + ' product')
        got += 1

print(f'wrote {got}/{len(rgd1_rids)} R/TS/P triples to {OUT}')
if missing:
    print(f'missing {len(missing)}: first 5 = {missing[:5]}')
with open('pipeline_rebuild/results/stage1b_rgd1.json','w') as f:
    json.dump({'wrote': got, 'total_needed': len(rgd1_rids), 'missing': missing}, f, indent=2)
PY

echo "[$(date)] Stage 1b done."
