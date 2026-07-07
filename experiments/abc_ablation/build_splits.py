"""Build outer_folds.json for the A/B/C ablation.

- Reaction-level 5-fold, family-stratified, seed = 42.
- Reactions come from the m3 bundle (787 unique reaction_ids).
- Family labels come from the m3 families sidecar json.
- Output layout:
    splits/outer_folds.json  →  {
        "seed": 42,
        "n_folds": 5,
        "families": ["dipolar", "qmrxn20_e2", "qmrxn20_sn2", "rgd1"],
        "folds": [{"fold": F, "test_rids": [...], "train_rids": [...]}, ...],
    }
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BUNDLE = REPO / "pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.pt"
FAMILIES = REPO / "pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.families.json"
OUT = Path(__file__).resolve().parent / "splits" / "outer_folds.json"

SEED = 42
K = 5


def main() -> None:
    b = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    rids = list(b["reaction_ids"])
    fams_map = json.load(open(FAMILIES))
    fams = np.array([fams_map[r] for r in rids])
    n = len(rids)
    print(f"n_reactions={n} (unique reaction_ids)", flush=True)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    folds = []
    for f, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n), fams)):
        train_rids = [rids[i] for i in tr_idx]
        test_rids = [rids[i] for i in te_idx]
        assert set(train_rids).isdisjoint(test_rids), f"fold{f} leakage"
        from collections import Counter
        print(f"  fold{f}: n_train={len(train_rids)} n_test={len(test_rids)} "
              f"test_fam={dict(Counter(fams_map[r] for r in test_rids))}", flush=True)
        folds.append({"fold": f, "train_rids": train_rids, "test_rids": test_rids})

    # Coverage check: every rid appears in exactly one test set.
    covered = set()
    for fd in folds:
        covered |= set(fd["test_rids"])
    assert covered == set(rids), "test coverage != full pool"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seed": SEED,
        "n_folds": K,
        "families": sorted(set(fams_map.values())),
        "folds": folds,
    }, indent=2))
    print(f"wrote → {OUT}", flush=True)


if __name__ == "__main__":
    main()
