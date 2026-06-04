#!/bin/bash
# Merge outputs/stage5a_4500/ into outputs/stage5a/ so the existing Flask app
# picks up the 4500 new reactions. Existing 500 entries are untouched.
set -e
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

SRC=outputs/stage5a_4500
DST=outputs/stage5a

if [ ! -d "$SRC/per_reaction" ]; then
    echo "ERROR: $SRC/per_reaction missing — Stage 5a 4500 not complete?"
    exit 1
fi

N_NEW=$(ls $SRC/per_reaction | wc -l)
N_OLD=$(ls $DST/per_reaction | wc -l)
echo "merging: $N_NEW new dirs from $SRC/per_reaction → $DST/per_reaction (existing $N_OLD)"

# Copy per-reaction dirs (skip if dest already exists — shouldn't happen)
for d in $SRC/per_reaction/*/; do
    name=$(basename "$d")
    if [ ! -d "$DST/per_reaction/$name" ]; then
        cp -r "$d" "$DST/per_reaction/"
    fi
done

# Merge fragmentation_summary.json (append new entries, no duplicates)
python3 - <<'PY'
import json
from pathlib import Path
DST = Path("outputs/stage5a/fragmentation_summary.json")
SRC = Path("outputs/stage5a_4500/fragmentation_summary.json")
old = json.load(open(DST)) if DST.exists() else []
new = json.load(open(SRC)) if SRC.exists() else []
old_ids = {r["reaction_id"] for r in old}
add = [r for r in new if r["reaction_id"] not in old_ids]
merged = old + add
DST.write_text(json.dumps(merged, indent=2))
print(f"[OK] fragmentation_summary.json: was {len(old)}, +{len(add)} new → {len(merged)}")
PY

N_FINAL=$(ls $DST/per_reaction | wc -l)
echo "done. $DST/per_reaction now has $N_FINAL reactions."
