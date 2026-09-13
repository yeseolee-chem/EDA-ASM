#!/usr/bin/env python3
"""SPEC18 Step 1 — 200-rxn stratified pilot (40 per fold).

Only rxns that (a) are in the spec17rev2 fold split AND (b) have a
non-null Kabsch RMSD in generation_quality.csv (== succeeded through
Step 9 in the prior stage) are eligible. That way xTB-vs-OTFM(RP)
comparison in Step 3 has a valid reference row for every pilot rxn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"
SEED, PER_FOLD = 42, 40

folds = pd.read_csv(PREV / "artifacts" / "folds.csv")
gq = pd.read_csv(PREV / "artifacts" / "generation_quality.csv")
# 'ok' column not guaranteed; treat rows with a finite rmsd as usable.
if "ok" in gq.columns:
    gq = gq[gq.ok.astype(bool)]
else:
    gq = gq.dropna(subset=[c for c in gq.columns if "rmsd" in c.lower()])

rng = np.random.default_rng(SEED)
rows = []
for k in range(5):
    ids = folds[folds.fold == k].rxn_id.values
    ids = np.intersect1d(ids, gq.rxn_id.values)
    if len(ids) < PER_FOLD:
        print(f"WARN fold{k}: eligible {len(ids)} < PER_FOLD {PER_FOLD}")
    pick = rng.choice(ids, size=min(PER_FOLD, len(ids)), replace=False)
    for r in pick:
        rows.append(dict(rxn_id=int(r), fold=k))

S = pd.DataFrame(rows).sort_values("rxn_id").reset_index(drop=True)
out = BASE / "artifacts" / "pilot_sample.csv"
S.to_csv(out, index=False)
print(f"파일럿 표본: {len(S)}  ->  {out}")
print(S.fold.value_counts().sort_index().to_string())

(BASE / "artifacts" / "GATE1_STATUS.txt").write_text(
    ("PASS" if len(S) == 5 * PER_FOLD else "WARN") + "\n"
    f"n_pilot={len(S)}\n"
    f"per_fold={PER_FOLD}\n"
    f"seed={SEED}\n"
)
