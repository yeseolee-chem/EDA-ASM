#!/usr/bin/env python3
"""Step 2: train 3 arms × 2 splits × 5 seeds × 5 folds × 7 channels.

Arm-A: X = X(B) - X(A)                        (dim = D)
Arm-B: X = [X(A), X(B)]                       (dim = 2D)
Arm-C: X = [X(A), X(B), X(B)-X(A)]  +          (dim = 3D)
       antisymmetric augmentation on TRAIN ONLY: (X(B),X(A),X(A)-X(B), -δ)

Each fit uses XGB HPs from xgb_one_seed.py.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/pairwise_delta")
OUT = BASE / "artifacts"

SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5
ARMS = ["A_diff", "B_concat", "C_symmetric"]
SPLITS = ["component", "swaptype_holdout"]

XGB_HPS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1,
    early_stopping_rounds=50, tree_method="hist", verbosity=0,
    n_jobs=int(os.environ.get("XGB_NJOBS", "4")),
)


def build_arm_features(X_A, X_B, arm):
    if arm == "A_diff":
        return X_B - X_A
    if arm == "B_concat":
        return np.hstack([X_A, X_B])
    if arm == "C_symmetric":
        return np.hstack([X_A, X_B, X_B - X_A])
    raise ValueError(arm)


def build_train_X(X_A, X_B, arm):
    """Build X for training. Arm-C: stack original + reversed pairs (antisymmetric augmentation).
    y augmentation is handled separately in the channel loop.
    """
    Xp = build_arm_features(X_A, X_B, arm)
    if arm != "C_symmetric":
        return Xp
    Xr = build_arm_features(X_B, X_A, arm)
    return np.vstack([Xp, Xr])


def main():
    with open(OUT / "pair_dataset.pkl", "rb") as f:
        pd_ = pickle.load(f)
    pairs = pd_["pairs"]
    X_A_full = pd_["X_A"].values
    X_B_full = pd_["X_B"].values
    delta_full = pd_["delta"]
    channels = pd_["learned_channels"]
    n = len(pairs)
    print(f"pairs: {n}  features: {X_A_full.shape[1]}  channels: {channels}")

    fold_cols = {"component": "fold_A", "swaptype_holdout": "fold_B"}
    all_test_records = []
    per_fit_log = []

    for split in SPLITS:
        fold_col = fold_cols[split]
        for arm in ARMS:
            for seed in SEEDS:
                for fold in range(N_FOLDS):
                    test_mask = (pairs[fold_col] == fold).values
                    train_mask = ~test_mask
                    tr_idx = np.where(train_mask)[0]
                    te_idx = np.where(test_mask)[0]
                    if len(te_idx) == 0:
                        continue
                    tr2_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=seed)

                    # Build features per arm
                    X_tr_raw = build_train_X(X_A_full[tr2_idx], X_B_full[tr2_idx], arm)
                    X_va_raw = build_arm_features(X_A_full[va_idx], X_B_full[va_idx], arm)
                    X_te_raw = build_arm_features(X_A_full[te_idx], X_B_full[te_idx], arm)

                    sc_X = StandardScaler()
                    Xtr = sc_X.fit_transform(X_tr_raw)
                    Xva = sc_X.transform(X_va_raw)
                    Xte = sc_X.transform(X_te_raw)

                    for ch in channels:
                        y_all = delta_full[ch].values
                        # augment y for Arm-C
                        if arm == "C_symmetric":
                            y_tr_raw = np.concatenate([y_all[tr2_idx], -y_all[tr2_idx]])
                        else:
                            y_tr_raw = y_all[tr2_idx]
                        y_va_raw = y_all[va_idx]
                        y_te_raw = y_all[te_idx]

                        sc_y = StandardScaler()
                        y_tr = sc_y.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()
                        y_va = sc_y.transform(y_va_raw.reshape(-1, 1)).ravel()

                        hp = dict(XGB_HPS); hp["random_state"] = seed
                        model = XGBRegressor(**hp)
                        model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
                        y_pred_scaled = model.predict(Xte)
                        y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                        # Record per-pair test predictions
                        for k, pair_idx in enumerate(te_idx):
                            all_test_records.append({
                                "arm": arm, "split": split,
                                "seed": seed, "fold": fold,
                                "pair_idx": int(pair_idx),
                                "channel": ch,
                                "delta_true": float(y_te_raw[k]),
                                "delta_pred": float(y_pred[k]),
                            })

                        # Fit-level log
                        y_tr_pred = sc_y.inverse_transform(
                            model.predict(Xtr).reshape(-1, 1)).ravel()
                        per_fit_log.append({
                            "arm": arm, "split": split, "seed": seed, "fold": fold, "channel": ch,
                            "train_mae": float(mean_absolute_error(y_tr_raw, y_tr_pred)),
                            "test_mae": float(mean_absolute_error(y_te_raw, y_pred)),
                            "n_train": len(y_tr_raw), "n_test": len(y_te_raw),
                        })
                    print(f"  [{split}/{arm} seed={seed} fold={fold}] done "
                          f"(train n={len(tr2_idx)}{'x2' if arm=='C_symmetric' else ''} "
                          f"test n={len(te_idx)})", flush=True)

    preds = pd.DataFrame(all_test_records)
    log = pd.DataFrame(per_fit_log)
    with open(OUT / "predictions_all.pkl", "wb") as f:
        pickle.dump(preds, f)
    log.to_csv(OUT / "fit_log.csv", index=False)
    print(f"Saved predictions_all.pkl ({len(preds)} rows), fit_log.csv ({len(log)} rows)")

    # GATE-2: Arm-A vs Arm-0 (baseline from delta_mae_baseline)
    ref = pd.read_csv(BASE.parent / "delta_mae_baseline/artifacts/delta_mae_table.csv")
    ref = ref[(ref["scheme"] == "groupkfold_component")].set_index("channel")["mae_delta"]
    # Aggregate Arm-A component test MAE per (channel, seed) then mean
    a_scores = log[(log["arm"] == "A_diff") & (log["split"] == "component")]
    a_agg = a_scores.groupby("channel")["test_mae"].mean()
    print("\nArm-A vs Arm-0 (component split):")
    wins = 0
    for ch in ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft"]:
        a = a_agg.get(ch, np.nan)
        r = ref.get(ch, np.nan)
        w = "WIN" if a < r else "loss"
        if a < r:
            wins += 1
        print(f"  {ch}: A={a:.3f}  Arm0={r:.3f}  {w}")
    with open(OUT / "GATE2_STATUS.txt", "w") as f:
        if wins >= 5:
            f.write(f"PASS Arm-A wins {wins}/7 channels vs Arm-0\n")
        else:
            f.write(f"FAIL Arm-A wins {wins}/7 (need 5+) — H1 rejected\n")
    print(f"=== GATE-2: Arm-A wins {wins}/7 channels ===")


if __name__ == "__main__":
    main()
