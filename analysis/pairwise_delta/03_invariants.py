#!/usr/bin/env python3
"""Step 3: invariant checks per arm.

3.1 self-check: predicted δ(A,A) should be 0
3.2 antisymmetry: |δ(A,B) + δ(B,A)| should be 0

Uses a single seed=42, component split, fold=0 hold-out training
to build 3 arm models per channel, then evaluates over all 882 rxns
(self-check) and 1936 pairs (antisym) — INFORMATIONAL only.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta")
OUT = BASE / "artifacts"

SEED = 42
FOLD = 0
CHANNELS_SUB = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
                "oi_dft", "disp_dft", "cpcm_dft"]

XGB_HPS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    min_child_weight=1, early_stopping_rounds=50, tree_method="hist",
    verbosity=0, n_jobs=4,
)


def build(X_A, X_B, arm):
    if arm == "A_diff":
        return X_B - X_A
    if arm == "B_concat":
        return np.hstack([X_A, X_B])
    if arm == "C_symmetric":
        return np.hstack([X_A, X_B, X_B - X_A])


def main():
    with open(OUT / "pair_dataset.pkl", "rb") as f:
        pd_ = pickle.load(f)
    pairs = pd_["pairs"]
    X_A = pd_["X_A"].values
    X_B = pd_["X_B"].values
    delta_df = pd_["delta"]

    # All reactions involved (882) — build unique rxn → X mapping from A/B pools
    rxn_ids = np.unique(np.concatenate([pairs["rxn_id_A"].values, pairs["rxn_id_B"].values]))
    rxn_to_X = {}
    for i in range(len(pairs)):
        rxn_to_X[int(pairs["rxn_id_A"].iloc[i])] = X_A[i]
        rxn_to_X[int(pairs["rxn_id_B"].iloc[i])] = X_B[i]

    train_mask = (pairs["fold_A"] != FOLD).values
    tr_idx = np.where(train_mask)[0]
    tr2, va = train_test_split(tr_idx, test_size=0.1, random_state=SEED)

    records = []
    for arm in ["A_diff", "B_concat", "C_symmetric"]:
        Xtr_raw = build(X_A[tr2], X_B[tr2], arm)
        if arm == "C_symmetric":
            Xtr_raw = np.vstack([Xtr_raw, build(X_B[tr2], X_A[tr2], arm)])
        Xva_raw = build(X_A[va], X_B[va], arm)
        sc_X = StandardScaler()
        Xtr = sc_X.fit_transform(Xtr_raw)
        Xva = sc_X.transform(Xva_raw)

        # Self-check inputs: X_A(rxn) vs X_A(rxn) → for each unique rxn
        rxns_arr = np.array(list(rxn_ids))
        Xs_arr = np.stack([rxn_to_X[int(r)] for r in rxns_arr])
        Xself = build(Xs_arr, Xs_arr, arm)
        Xself = sc_X.transform(Xself)

        # Antisym inputs: reversed pairs
        Xrev = build(X_B, X_A, arm)
        Xrev = sc_X.transform(Xrev)
        Xfwd = sc_X.transform(build(X_A, X_B, arm))

        for ch in CHANNELS_SUB:
            y_all = delta_df[ch].values
            if arm == "C_symmetric":
                y_tr = np.concatenate([y_all[tr2], -y_all[tr2]])
            else:
                y_tr = y_all[tr2]
            y_va = y_all[va]

            sc_y = StandardScaler()
            y_tr_s = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
            y_va_s = sc_y.transform(y_va.reshape(-1, 1)).ravel()

            hp = dict(XGB_HPS); hp["random_state"] = SEED
            m = XGBRegressor(**hp)
            m.fit(Xtr, y_tr_s, eval_set=[(Xva, y_va_s)], verbose=False)

            # 3.1 self-check
            y_self = sc_y.inverse_transform(m.predict(Xself).reshape(-1, 1)).ravel()
            # 3.2 antisym: for each pair, forward + reverse should sum to 0
            y_fwd = sc_y.inverse_transform(m.predict(Xfwd).reshape(-1, 1)).ravel()
            y_rev = sc_y.inverse_transform(m.predict(Xrev).reshape(-1, 1)).ravel()
            asym_err = np.abs(y_fwd + y_rev)

            records.append({
                "arm": arm, "channel": ch,
                "self_mean_abs": float(np.mean(np.abs(y_self))),
                "self_median_abs": float(np.median(np.abs(y_self))),
                "self_p95_abs": float(np.percentile(np.abs(y_self), 95)),
                "asym_mean": float(asym_err.mean()),
                "asym_median": float(np.median(asym_err)),
                "asym_p95": float(np.percentile(asym_err, 95)),
            })
    df = pd.DataFrame(records)
    df.to_csv(OUT / "invariant_check.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
