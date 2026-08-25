#!/usr/bin/env python3
"""Step 0: build pair-level dataset for direct δ learning (SPEC12).

Inputs:
  phase5_dataset_v2.pkl  (3504 rxns × features)
  pairs_dedup.pkl        (1936 MMP pairs, ordering: sub_A<sub_B by lex)

Outputs:
  artifacts/pair_dataset.pkl:  per-pair X_A, X_B, δ per learned channel,
                                swap_type, comp_id (from components.pkl)
  artifacts/dropped_zerovar_columns.txt
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
PHASE5 = BASE / "analysis/bath1480_probe/experiments/cohort_v1/phase5_dataset_v2.pkl"
DMB = BASE / "analysis/delta_mae_baseline/artifacts"
OUT = BASE / "analysis/pairwise_delta/artifacts"
OUT.mkdir(parents=True, exist_ok=True)

LEARNED = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
           "oi_dft", "disp_dft", "cpcm_dft"]

ZERO_VAR_TOL = 1e-12


def get_features(df):
    remove = [c for c in df.columns if "_dft" in c]
    non_num = ["reaction_number", "elements_frag1", "elements_frag2"]
    y = df[remove].copy()
    X = df.drop(columns=remove)
    drop = [c for c in non_num if c in X.columns]
    X = X.drop(columns=drop).select_dtypes(include=[np.number])
    return X, y


def main():
    ds = pd.read_pickle(PHASE5)
    assert len(ds) == 3504, f"phase5 rows={len(ds)}"
    ds["reaction_number"] = ds["reaction_number"].astype(int)
    ds = ds.sort_values("reaction_number").reset_index(drop=True)
    X_all, y_all = get_features(ds)
    print(f"features: {X_all.shape[1]}   targets: {y_all.shape[1]}")

    for t in LEARNED:
        assert t in y_all.columns, f"missing target {t}"

    pairs = pd.read_pickle(DMB / "pairs_dedup.pkl").reset_index(drop=True)
    assert len(pairs) == 1936, f"pairs={len(pairs)}"

    # Index for fast per-rxn lookup
    rxn_to_idx = {int(r): i for i, r in enumerate(ds["reaction_number"])}

    A_idx = pairs["rxn_id_A"].map(rxn_to_idx).astype(int).values
    B_idx = pairs["rxn_id_B"].map(rxn_to_idx).astype(int).values

    X_A = X_all.iloc[A_idx].reset_index(drop=True)
    X_B = X_all.iloc[B_idx].reset_index(drop=True)

    # Zero-variance filter on difference matrix (§0.4)
    diff = X_B.values - X_A.values
    zero_var_cols = [X_all.columns[j] for j in range(X_all.shape[1])
                     if diff[:, j].std() < ZERO_VAR_TOL]
    print(f"Zero-var (diff) columns removed: {len(zero_var_cols)}")
    (OUT / "dropped_zerovar_columns.txt").write_text("\n".join(zero_var_cols) + "\n")

    keep_cols = [c for c in X_all.columns if c not in zero_var_cols]
    X_A = X_A[keep_cols]
    X_B = X_B[keep_cols]
    print(f"kept features per side: {len(keep_cols)}")

    # Build delta labels
    delta = {}
    for t in LEARNED:
        yA = y_all[t].iloc[A_idx].values
        yB = y_all[t].iloc[B_idx].values
        delta[t] = yB - yA

    # Load components → assign comp_id
    with open(DMB / "components.pkl", "rb") as f:
        comps = pickle.load(f)
    rxn_to_comp = {}
    for cid, members in enumerate(comps):
        for r in members:
            rxn_to_comp[int(r)] = cid
    pairs["comp_id"] = pairs["rxn_id_A"].map(rxn_to_comp).astype(int)
    # sanity: A and B share component
    b_comp = pairs["rxn_id_B"].map(rxn_to_comp).astype(int).values
    assert (pairs["comp_id"].values == b_comp).all(), "pair rxn_A, rxn_B differ in component"

    pairs["swap_type"] = pairs["sub_A"] + ">>" + pairs["sub_B"]
    print(f"unique swap_types: {pairs['swap_type'].nunique()}")

    out = {
        "pairs": pairs.reset_index(drop=True),
        "X_A": X_A, "X_B": X_B,
        "delta": pd.DataFrame(delta),
        "feature_cols": keep_cols,
        "learned_channels": LEARNED,
    }
    with open(OUT / "pair_dataset.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"Saved pair_dataset.pkl  X_A={X_A.shape}  delta={pd.DataFrame(delta).shape}")

    with open(OUT / "GATE0_STATUS.txt", "w") as f:
        f.write(f"PASS n_pairs={len(pairs)} unique_rxns={len(set(pairs['rxn_id_A'])|set(pairs['rxn_id_B']))} "
                f"n_swaptypes={pairs['swap_type'].nunique()} kept_features={len(keep_cols)}\n")
    print("=== GATE-0 PASS ===")


if __name__ == "__main__":
    main()
