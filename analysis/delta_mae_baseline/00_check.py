#!/usr/bin/env python3
"""Step 0: data integrity for SPEC11 δ-MAE baseline.

GATE-0: assert phase5 = 3504 with reaction_number, dedup MMP pairs = 1936,
all rxns in pairs (882) present in phase5.
"""
from pathlib import Path
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
PHASE5 = BASE / "analysis/bath1480_probe/experiments/cohort_v1/phase5_dataset_v2.pkl"
PAIRS = BASE / "analysis/mmp_gate_a/artifacts/mmp_pairs_labeled_v2.pkl"
OUT = BASE / "analysis/delta_mae_baseline/artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_pickle(PHASE5)
    assert len(df) == 3504, f"phase5 rows={len(df)}"
    assert "reaction_number" in df.columns, "reaction_number missing"
    assert df["reaction_number"].nunique() == 3504, "duplicated rxn"
    df["reaction_number"] = df["reaction_number"].astype(int)

    pairs = pd.read_pickle(PAIRS)
    n_raw = len(pairs)
    dedup = pairs.drop_duplicates(subset=["rxn_id_A", "rxn_id_B"]).reset_index(drop=True)
    n_dedup = len(dedup)
    assert n_dedup == 1936, f"deduped pairs = {n_dedup}, expected 1936"

    pair_rxns = set(dedup["rxn_id_A"]) | set(dedup["rxn_id_B"])
    n_pair_rxns = len(pair_rxns)
    print(f"phase5: {len(df)} rxns")
    print(f"MMP pairs: raw={n_raw}  deduped={n_dedup}  unique rxns in pairs={n_pair_rxns}")

    ds_rxns = set(df["reaction_number"])
    missing = pair_rxns - ds_rxns
    assert not missing, f"{len(missing)} pair rxns missing from phase5: {list(missing)[:5]}"

    dedup.to_pickle(OUT / "pairs_dedup.pkl")
    with open(OUT / "GATE0_STATUS.txt", "w") as f:
        f.write(f"PASS phase5={len(df)} pairs_dedup={n_dedup} unique_rxns={n_pair_rxns}\n")
    print(f"=== GATE-0 PASS ===")


if __name__ == "__main__":
    main()
