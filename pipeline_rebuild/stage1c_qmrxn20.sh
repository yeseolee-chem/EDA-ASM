#!/bin/bash
#SBATCH --job-name=st1c_qmrxn
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline_rebuild/logs/st1c_%j.out
#SBATCH --error=pipeline_rebuild/logs/st1c_%j.err

# Stage 1c — download QMrxn20 (von Rudorff 2020, materialscloud 2020.55).
# The archive is /record/2020.55; direct file downloads use the /record/file
# endpoint with filename+record_id query params (older MC API).

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p pipeline_rebuild/logs pipeline_rebuild/results

RAW=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw
mkdir -p "$RAW/QMrxn20"
cd "$RAW/QMrxn20"

# Files we need per the QMrxn20 loader:
#   geometries.tgz       — nested transition-states/, reactant-complex-*, product-conformers/, ...
#   energies.txt         — per-species DF-LCCSD + MP2 + HF energies
#   barriers.txt         — TS↔reactant-complex pairing with activation energy
#
# Try 4 URL patterns until one succeeds.
try_download() {
  local NAME=$1
  echo "[$(date)] === trying $NAME ==="
  local URLS=(
    "https://archive.materialscloud.org/record/file?filename=${NAME}&record_id=580"
    "https://archive.materialscloud.org/record/file?filename=${NAME}&record_id=2020.55"
    "https://archive.materialscloud.org/records/2020.55/files/${NAME}/content"
    "https://archive.materialscloud.org/api/records/2020.55/files/${NAME}/content"
  )
  for U in "${URLS[@]}"; do
    echo "  trying $U"
    if curl -sSL -f -o "$NAME" "$U"; then
      SZ=$(stat -c '%s' "$NAME")
      echo "  got $NAME ($SZ bytes)"
      if [ "$SZ" -lt 1024 ]; then
        head -c 500 "$NAME"; echo
        rm -f "$NAME"; continue
      fi
      return 0
    fi
  done
  echo "!! all URLs failed for $NAME"
  return 1
}

for f in geometries.tgz energies.txt barriers.txt; do
  if [ ! -f "$f" ]; then
    try_download "$f" || echo "SKIPPED $f"
  fi
done

# Fallback — hit the landing page and scrape file links
if [ ! -f geometries.tgz ] || [ ! -f energies.txt ] || [ ! -f barriers.txt ]; then
  echo "[$(date)] fallback — scraping landing page for direct links"
  curl -sSL -o landing.html "https://archive.materialscloud.org/record/2020.55"
  python3 <<'PY'
import re, urllib.request
html = open('landing.html').read()
# match hrefs to files
matches = re.findall(r'href="([^"]+)"', html)
targets = [u for u in matches if any(k in u for k in ('geometries', 'energies.txt', 'barriers.txt', 'file?'))]
print('found candidates:')
for u in targets: print(' ', u)
PY
fi

# Extract
if [ -f geometries.tgz ]; then
  echo "[$(date)] extracting geometries.tgz ..."
  tar xzf geometries.tgz
  echo "  post-extract layout:"
  ls -1 | head -15
fi

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

# Re-match against the labels parquet, this time including e2/sn2
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
    print(f'{fam}: matched={ok} missing={miss}   sample={missing_sample}')
    result[fam] = {'matched': ok, 'missing': miss, 'sample_missing': missing_sample}

Path('pipeline_rebuild/results/stage1c_match.json').write_text(json.dumps(result, indent=2))
print('wrote pipeline_rebuild/results/stage1c_match.json')
PY

echo "[$(date)] Stage 1c done."
