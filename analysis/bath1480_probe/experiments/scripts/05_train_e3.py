#!/usr/bin/env python3
"""Phase 4: Stacked XGBoost + MACE-OFF23 residual training (TS-only, 8 targets).

Env:
  FOLD: 0..4
  SEED: 0..4
Runs one cell (fold × seed). Total: 25 cells, submit as 5×5 array.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FOLD = int(os.environ["FOLD"])
SEED = int(os.environ["SEED"])

COHORT_DIR = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
OUT_ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase4_stacked_xgb_mace")
OUT_DIR = OUT_ROOT / f"fold_{FOLD}" / f"seed_{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["d1_own", "d2_own", "elst_dft", "pauli_dft",
               "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
EPOCHS_MAX = 100_000
PATIENCE = 10_000
BATCH = 16
LR = 1e-5
WEIGHT_DECAY = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    labels = pd.read_pickle(COHORT_DIR / "labels_v1.pkl")
    physics = pd.read_pickle(COHORT_DIR / "physics_24.pkl")
    df = labels.merge(physics, on="reaction_number")
    return df


def pad_mace(feats_list, max_atoms):
    """Pad list of (N_atoms, 256) tensors to (max_atoms, 256), return tensor + mask."""
    N = len(feats_list)
    D = feats_list[0].shape[1]
    padded = torch.zeros((N, max_atoms, D), dtype=torch.float32)
    mask = torch.zeros((N, max_atoms), dtype=torch.bool)
    for i, f in enumerate(feats_list):
        n = f.shape[0]
        padded[i, :n] = f
        mask[i, :n] = True
    return padded, mask


def load_mace_features(rxn_ids):
    mace_dir = COHORT_DIR / "mace_ts_v1"
    feats = []
    for rxn in rxn_ids:
        pt_path = mace_dir / f"{rxn:04d}.pt"
        d = torch.load(pt_path, map_location="cpu")
        feats.append(d["mace_feats"])
    max_atoms = max(f.shape[0] for f in feats)
    return pad_mace(feats, max_atoms)


class ResidualNet(nn.Module):
    def __init__(self, in_dim=256, hidden=128, n_targets=8, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # attention pool via learned query
        self.attn_query = nn.Parameter(torch.randn(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, n_targets),
        )

    def forward(self, mace_feats, mask):
        h = self.proj(mace_feats)  # (B, N_atoms, hidden)
        # attention weights (learned query dot each atom)
        scores = torch.einsum("bnh,h->bn", h, self.attn_query)  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (B, N)
        v = torch.einsum("bnh,bn->bh", h, weights)  # (B, hidden)
        return self.head(v)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = load_data()
    print(f"Loaded: {df.shape}")

    # Fold split: use FOLD as test, others as train
    test_mask = df["fold_id"] == FOLD
    train_mask = ~test_mask
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)
    print(f"Train: {len(df_train)}  Test: {len(df_test)}")

    # Physics features
    physics_cols = [c for c in df.columns
                    if c not in TARGET_COLS + ["reaction_number", "fold_id", "n_atoms",
                                                "e_ab_eh", "e_frag1_dist_tzvp_eh",
                                                "e_frag2_dist_tzvp_eh", "e_frag1_rel_tzvp_eh",
                                                "e_frag2_rel_tzvp_eh", "interaction_own",
                                                "barrier_own_reconstructed", "closure_gap_kcal",
                                                "d1_dft_espley", "d2_dft_espley", "sum_dist_espley",
                                                "interaction_espley", "barrier_e_espley", "barrier_q_espley",
                                                "eint_dft"]]
    print(f"Physics cols: {physics_cols[:5]}... ({len(physics_cols)} total)")
    X_train_phy = df_train[physics_cols].values.astype(np.float32)
    X_test_phy = df_test[physics_cols].values.astype(np.float32)
    y_train = df_train[TARGET_COLS].values.astype(np.float32)
    y_test = df_test[TARGET_COLS].values.astype(np.float32)

    # ---- Stage A: XGBoost baseline (per target) ----
    print("\n=== Stage A: XGBoost baseline ===")
    xgb_models = {}
    yhat_baseline_train = np.zeros_like(y_train)
    yhat_baseline_test = np.zeros_like(y_test)
    for i, t in enumerate(TARGET_COLS):
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            random_state=SEED, tree_method="hist", n_jobs=4)
        xgb.fit(X_train_phy, y_train[:, i])
        yhat_baseline_train[:, i] = xgb.predict(X_train_phy)
        yhat_baseline_test[:, i] = xgb.predict(X_test_phy)
        xgb_models[t] = xgb
        mae = mean_absolute_error(y_test[:, i], yhat_baseline_test[:, i])
        print(f"  {t}: XGB baseline MAE = {mae:.4f}")

    # Residuals for training
    r_train = y_train - yhat_baseline_train  # (N, 8)

    # ---- Stage B: MACE residual network ----
    print("\n=== Stage B: MACE residual network ===")
    mace_train, mask_train = load_mace_features(df_train["reaction_number"].tolist())
    mace_test, mask_test = load_mace_features(df_test["reaction_number"].tolist())

    # Normalize residual by train fold std per target
    sigma_c = r_train.std(axis=0) + 1e-8  # (8,)
    r_train_normed = r_train / sigma_c

    mace_train = mace_train.to(device)
    mask_train = mask_train.to(device)
    mace_test = mace_test.to(device)
    mask_test = mask_test.to(device)
    r_train_t = torch.from_numpy(r_train_normed).to(device)

    net = ResidualNet(in_dim=mace_train.shape[-1], n_targets=8).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_train = len(df_train)
    best_loss = float("inf")
    patience = 0
    for epoch in range(EPOCHS_MAX):
        net.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i+BATCH]
            pred = net(mace_train[idx], mask_train[idx])
            loss = (pred - r_train_t[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1

        if epoch % 500 == 0:
            print(f"  epoch {epoch:>6d}  loss={epoch_loss:.4f}  best={best_loss:.4f}  patience={patience}")

        if patience >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    # ---- Final prediction ----
    net.eval()
    with torch.no_grad():
        delta_test_normed = net(mace_test, mask_test).cpu().numpy()
    delta_test = delta_test_normed * sigma_c
    y_pred = yhat_baseline_test + delta_test

    # Metrics per target
    metrics = {}
    for i, t in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        metrics[t] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  {t}: final MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.3f}")

    # Save
    with open(OUT_DIR / "xgb_models.pkl", "wb") as f:
        pickle.dump(xgb_models, f)
    torch.save(net.state_dict(), OUT_DIR / "delta_net.pt")
    pd.DataFrame({
        "reaction_number": np.repeat(df_test["reaction_number"].values, 8),
        "target": TARGET_COLS * len(df_test),
        "y_true": y_test.flatten(),
        "y_pred": y_pred.flatten(),
        "y_baseline": yhat_baseline_test.flatten(),
        "delta": delta_test.flatten(),
    }).to_parquet(OUT_DIR / "predictions.parquet", index=False)
    with open(OUT_DIR / "metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
