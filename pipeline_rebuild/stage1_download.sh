#!/bin/bash
#SBATCH --job-name=st1_download
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_rebuild/logs/st1_%j.out
#SBATCH --error=pipeline_rebuild/logs/st1_%j.err

# Stage 1 — download + extract source datasets (dipolar + qmrxn20), then
# match every reaction_id in labels/adf/adf_labels_v6_multifamily.parquet
# against the extracted trees. RGD1 handled separately.

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

RAW=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw
mkdir -p "$RAW/dipolar_cycloaddition" "$RAW/QMrxn20" pipeline_rebuild/logs pipeline_rebuild/results

# ---- Dipolar cycloaddition (Stuyver / Jorner / Coley 2023) ----
if [ ! -d "$RAW/dipolar_cycloaddition/extracted/full_dataset_profiles" ]; then
  echo "[$(date)] downloading dipolar_cycloaddition..."
  cd "$RAW/dipolar_cycloaddition"
  # figshare v5 file listing — DOI 10.6084/m9.figshare.21707888.v5
  # full_dataset_profiles.tar.gz contains geometries + energies per reaction
  curl -L -o listing.json \
    "https://api.figshare.com/v2/articles/21707888/files"
  # Pick the full_dataset_profiles.tar.gz entry
  python3 -c "
import json
with open('listing.json') as f: d=json.load(f)
for e in d:
    if e['name'].startswith('full_dataset_profiles'):
        print(e['download_url'], e['name'])
" > pick.txt
  cat pick.txt
  URL=$(head -1 pick.txt | awk '{print $1}')
  NAME=$(head -1 pick.txt | awk '{print $2}')
  echo "downloading $NAME from $URL"
  curl -L -o "$NAME" "$URL"
  mkdir -p extracted
  echo "[$(date)] extracting $NAME..."
  tar xzf "$NAME" -C extracted
  ls extracted | head -5
  # full_dataset.csv too
  URL_CSV=$(python3 -c "
import json
with open('listing.json') as f: d=json.load(f)
for e in d:
    if e['name'] == 'full_dataset.csv':
        print(e['download_url']); break
")
  curl -L -o full_dataset.csv "$URL_CSV"
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
fi

# ---- QMrxn20 (von Rudorff 2020) ----
if [ ! -d "$RAW/QMrxn20/transition-states" ]; then
  echo "[$(date)] downloading QMrxn20..."
  cd "$RAW/QMrxn20"
  # materialscloud record 2020.55 — geometries.tgz + energies.txt + barriers.txt
  # Query for direct file links via record api
  curl -L -o record.json \
    "https://archive.materialscloud.org/api/records/1eda15fd-b21d-4f11-91be-2fb2d3ce0c95/files"
  cat record.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
for e in d.get('entries', d if isinstance(d, list) else []):
    print(e.get('key'), '->', e.get('links',{}).get('content'))
" > listing.txt
  cat listing.txt
  # If the API changed, try the DOI URL landing page directly
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
fi

# ---- Match 789 reaction_ids to the extracted files ----
echo "[$(date)] matching 789 reaction_ids to on-disk XYZ..."
python3 - <<'PY'
import pandas as pd, json
from pathlib import Path

df = pd.read_parquet('labels/adf/adf_labels_v6_multifamily.parquet')
print(f'{len(df)} rows in labels/adf/adf_labels_v6_multifamily.parquet')
print(df.family.value_counts())

RAW = Path('/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw')
DIP_ROOT = RAW / 'dipolar_cycloaddition' / 'extracted' / 'full_dataset_profiles'
QMR_ROOT = RAW / 'QMrxn20'

hits, misses = {}, {}
for fam, sub in df.groupby('family'):
    ok = miss = 0
    sample_missing = []
    for rid in sub.reaction_id:
        if fam == 'dipolar':
            # rid pattern: dipolar_000027 -> index 27
            idx = int(rid.split('_')[-1])
            d = DIP_ROOT / str(idx)
            if d.exists() and any(d.glob('r0_*.xyz')):
                ok += 1
            else:
                miss += 1
                if len(sample_missing) < 3: sample_missing.append((rid, str(d)))
        elif fam in ('qmrxn20_e2', 'qmrxn20_sn2'):
            subfam = 'e2' if 'e2' in fam else 'sn2'
            # rid pattern: qmrxn20_e2_A_B_A_A_C_B -> label = A_B_A_A_C_B
            label = '_'.join(rid.split('_')[2:])
            ts = QMR_ROOT / 'transition-states' / subfam / f'{label}.xyz'
            if ts.exists(): ok += 1
            else:
                miss += 1
                if len(sample_missing) < 3: sample_missing.append((rid, str(ts)))
        elif fam == 'rgd1':
            miss += 1  # no loader yet — see stage 1b
            if len(sample_missing) < 3: sample_missing.append((rid, 'rgd1 loader TODO'))
    hits[fam] = ok
    misses[fam] = miss
    print(f'  {fam:12s} matched={ok}  missing={miss}   sample_missing={sample_missing}')

with open('pipeline_rebuild/results/stage1_match.json','w') as f:
    json.dump({'hits':hits,'misses':misses,'total_rows':len(df)}, f, indent=2)
print('wrote pipeline_rebuild/results/stage1_match.json')
PY

echo "[$(date)] Stage 1 done."
