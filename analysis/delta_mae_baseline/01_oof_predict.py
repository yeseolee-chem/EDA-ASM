#!/usr/bin/env python3
"""Step 1: OOF predictions with 2 split schemes × 5 seeds × 5 folds.

Reuses XGB HPs from bath1480_probe/experiments/scripts/xgb_one_seed.py.

Split (a) kfold_random          — sklearn.KFold(shuffle=True, seed)
Split (b) groupkfold_component  — greedy bin-pack of MMP-connected components

Saves artifacts/oof_predictions.pkl with reaction_number column preserved.
"""
import os
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
PHASE5 = BASE / "analysis/bath1480_probe/experiments/cohort_v1/phase5_dataset_v2.pkl"
OUT = BASE / "analysis/delta_mae_baseline/artifacts"

TARGETS = [
    "d1_own_dft", "d2_own_dft",
    "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft",
    "interaction_own_dft",
]
SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5

XGB_HPS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "verbosity": 0,
    "n_jobs": int(os.environ.get("XGB_NJOBS", "4")),
}


def get_dft_features(df):
    remove_targets = [c for c in df.columns if "_dft" in c]
    y = df[remove_targets].copy()
    non_num_meta = ["reaction_number", "elements_frag1", "elements_frag2"]
    X = df.drop(columns=remove_targets)
    drop = [c for c in non_num_meta if c in X.columns]
    X = X.drop(columns=drop).select_dtypes(include=[np.number])
    return X, y


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_components(rxn_ids, pairs):
    uf = UnionFind(rxn_ids)
    for _, r in pairs.iterrows():
        a, b = int(r["rxn_id_A"]), int(r["rxn_id_B"])
        if a in uf.p and b in uf.p:
            uf.union(a, b)
    comp = defaultdict(list)
    for x in rxn_ids:
        comp[uf.find(x)].append(x)
    return list(comp.values())


def assign_folds_bin_pack(components, n_folds, seed):
    """Sort components by size desc (with seeded jitter on ties), greedy assign to smallest fold."""
    rng = np.random.default_rng(seed)
    # Shuffle order among components of same size
    keyed = [(len(c), rng.random(), i, c) for i, c in enumerate(components)]
    keyed.sort(key=lambda x: (-x[0], x[1]))
    fold_sizes = [0] * n_folds
    fold_of = {}
    for _, _, _, c in keyed:
        f = int(np.argmin(fold_sizes))
        for rxn in c:
            fold_of[rxn] = f
        fold_sizes[f] += len(c)
    return fold_of, fold_sizes


def one_seed_scheme(X, y_all, rxn_ids, fold_of, seed, scheme):
    """Run 5-fold OOF for one (seed, scheme). Returns dict of OOF preds per target."""
    idx_by_fold = defaultdict(list)
    for i, r in enumerate(rxn_ids):
        idx_by_fold[fold_of[r]].append(i)
    oof = {t: np.full(len(rxn_ids), np.nan) for t in TARGETS}
    train_mae_log = {t: [] for t in TARGETS}
    test_mae_log = {t: [] for t in TARGETS}

    for fold in range(N_FOLDS):
        test_idx = np.array(idx_by_fold[fold])
        train_idx = np.array([i for f, ii in idx_by_fold.items() if f != fold for i in ii])
        # Inner val split for early stopping
        train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)

        sc_X = StandardScaler()
        Xtr = sc_X.fit_transform(X.iloc[train_idx2])
        Xva = sc_X.transform(X.iloc[val_idx])
        Xte = sc_X.transform(X.iloc[test_idx])

        for tgt in TARGETS:
            y_tr_raw = y_all[tgt].iloc[train_idx2].values.reshape(-1, 1)
            y_va_raw = y_all[tgt].iloc[val_idx].values.reshape(-1, 1)
            y_te_raw = y_all[tgt].iloc[test_idx].values
            sc_y = StandardScaler()
            y_tr = sc_y.fit_transform(y_tr_raw).ravel()
            y_va = sc_y.transform(y_va_raw).ravel()

            hp = dict(XGB_HPS); hp["random_state"] = seed
            model = XGBRegressor(**hp)
            model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
            y_pred_scaled = model.predict(Xte)
            y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            oof[tgt][test_idx] = y_pred

            y_tr_pred = sc_y.inverse_transform(model.predict(Xtr).reshape(-1, 1)).ravel()
            train_mae_log[tgt].append(mean_absolute_error(y_all[tgt].iloc[train_idx2].values, y_tr_pred))
            test_mae_log[tgt].append(mean_absolute_error(y_te_raw, y_pred))
        print(f"  [{scheme} seed={seed} fold={fold}] done", flush=True)
    train_mae = {t: float(np.mean(train_mae_log[t])) for t in TARGETS}
    test_mae = {t: float(np.mean(test_mae_log[t])) for t in TARGETS}
    return oof, train_mae, test_mae


def main():
    df = pd.read_pickle(PHASE5)
    df = df.sort_values("reaction_number").reset_index(drop=True)
    df["reaction_number"] = df["reaction_number"].astype(int)
    rxn_ids = df["reaction_number"].tolist()
    X, y_all = get_dft_features(df)
    for t in TARGETS:
        assert t in y_all.columns, f"target {t} missing"
    print(f"X: {X.shape}  targets: {TARGETS}")

    pairs = pd.read_pickle(OUT / "pairs_dedup.pkl")

    # Precompute components (independent of seed)
    comps = build_components(rxn_ids, pairs)
    n_comp = len(comps)
    max_comp = max(len(c) for c in comps)
    print(f"Connected components: {n_comp}  max size: {max_comp}")
    with open(OUT / "components.pkl", "wb") as f:
        pickle.dump(comps, f)

    records = []
    train_mae_records = []
    test_mae_records = []

    for scheme in ["kfold_random", "groupkfold_component"]:
        for seed in SEEDS:
            print(f"\n=== scheme={scheme} seed={seed} ===")
            if scheme == "kfold_random":
                kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                fold_of = {}
                for f, (_, test_idx) in enumerate(kf.split(rxn_ids)):
                    for i in test_idx:
                        fold_of[rxn_ids[i]] = f
            else:
                fold_of, sizes = assign_folds_bin_pack(comps, N_FOLDS, seed)
                print(f"  fold sizes: {sizes}")

            oof, tr_mae, te_mae = one_seed_scheme(X, y_all, rxn_ids, fold_of, seed, scheme)

            for i, r in enumerate(rxn_ids):
                row = {"reaction_number": r, "seed": seed, "scheme": scheme,
                       "fold": fold_of[r]}
                for t in TARGETS:
                    row[f"{t}_true"] = float(y_all[t].iloc[i])
                    row[f"{t}_pred"] = float(oof[t][i])
                records.append(row)
            for t in TARGETS:
                train_mae_records.append({"scheme": scheme, "seed": seed, "target": t,
                                          "train_mae": tr_mae[t]})
                test_mae_records.append({"scheme": scheme, "seed": seed, "target": t,
                                         "test_mae": te_mae[t]})

    oof_df = pd.DataFrame(records)
    oof_df.to_pickle(OUT / "oof_predictions.pkl")
    pd.DataFrame(train_mae_records).to_csv(OUT / "train_mae_per_seed.csv", index=False)
    pd.DataFrame(test_mae_records).to_csv(OUT / "test_mae_per_seed.csv", index=False)
    print(f"\nSaved oof_predictions.pkl: {len(oof_df)} rows "
          f"({len(SEEDS)} seeds × 2 schemes × {len(rxn_ids)})")

    # GATE-1 check: absolute MAE within ±10% of Phase 5 reference
    ref = {"d1_own_dft": 1.814, "d2_own_dft": 1.439,
           "elst_dft": 3.722, "pauli_dft": 5.566, "oi_dft": 3.012,
           "disp_dft": 1.010, "cpcm_dft": 2.382, "cds_dft": 0.294,
           "interaction_own_dft": 2.040}
    te = pd.DataFrame(test_mae_records)
    print("\nTest MAE by scheme (seed-averaged):")
    summary = te.groupby(["scheme", "target"])["test_mae"].mean().reset_index()
    summary["ref"] = summary["target"].map(ref)
    summary["ratio"] = summary["test_mae"] / summary["ref"]
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "gate1_absolute_mae_check.csv", index=False)

    kf_bad = summary[(summary["scheme"] == "kfold_random") & (summary["ratio"].between(0.9, 1.1).__invert__())]
    with open(OUT / "GATE1_STATUS.txt", "w") as f:
        if len(kf_bad) == 0:
            f.write("PASS kfold_random within ±10% of phase5 reference\n")
        else:
            f.write(f"WARN kfold_random out of ±10% for: {list(kf_bad['target'])}\n")
    print("=== GATE-1 written ===")


if __name__ == "__main__":
    main()
