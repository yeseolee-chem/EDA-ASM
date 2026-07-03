#!/bin/bash
#SBATCH --job-name=st1d_qmrxn
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline_rebuild/logs/st1d_%j.out
#SBATCH --error=pipeline_rebuild/logs/st1d_%j.err

# Stage 1d — QMrxn20 (Rudorff 2020, materialscloud DOI 2020.55).
# Correct record UUID discovered from stage1c landing-page scrape:
#   gkqvy-3vp74

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p pipeline_rebuild/logs pipeline_rebuild/results

RAW=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/QMrxn20
mkdir -p "$RAW"
cd "$RAW"

BASE="https://archive.materialscloud.org/records/gkqvy-3vp74/files"

for f in geometries.tgz energies.txt.gz barriers.txt.gz; do
  if [ ! -f "$f" ]; then
    echo "[$(date)] downloading $f ..."
    curl -sSL -o "$f" "$BASE/$f?download=1"
    ls -la "$f"
  fi
done

# Decompress the .gz text files
for f in energies.txt barriers.txt; do
  if [ ! -f "$f" ] && [ -f "$f.gz" ]; then
    echo "[$(date)] gunzipping $f.gz"
    gunzip -k "$f.gz"
  fi
done

# Extract geometries.tgz
if [ -f geometries.tgz ] && [ ! -d transition-states ]; then
  echo "[$(date)] extracting geometries.tgz ..."
  tar xzf geometries.tgz
  ls -1 | head
fi

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

# Rematch qmrxn20_e2/sn2 reaction_ids to disk
python3 - <<'PY'
import pandas as pd, json
from pathlib import Path
df = pd.read_parquet('labels/adf/adf_labels_v6_multifamily.parquet')
QMR_ROOT = Path('/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/QMrxn20')

result = {}
for fam in ('qmrxn20_e2', 'qmrxn20_sn2'):
    sub = df[df.family == fam]
    ok = miss = 0
    missing_sample = []
    for rid in sub.reaction_id:
        subfam = 'e2' if 'e2' in fam else 'sn2'
        label = '_'.join(rid.split('_')[2:])
        ts = QMR_ROOT / 'transition-states' / subfam / f'{label}.xyz'
        if ts.exists(): ok += 1
        else:
            miss += 1
            if len(missing_sample) < 3: missing_sample.append((rid, str(ts)))
    print(f'{fam:15s} matched={ok}  missing={miss}   sample_missing={missing_sample}')
    result[fam] = {'matched': ok, 'missing': miss, 'sample_missing': missing_sample}

Path('pipeline_rebuild/results/stage1d_match.json').write_text(json.dumps(result, indent=2))
print('wrote pipeline_rebuild/results/stage1d_match.json')
PY

echo "[$(date)] Stage 1d done."
