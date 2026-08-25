#!/usr/bin/env python3
"""Phase 6: Phase 5 XGB base (78 features, paper 80/10/10 split) + MACE-OFF23 residual head.

Same split methodology as Phase 5 (xgb_one_seed.py) → direct comparison to
Phase 5 XGB per-target MAE with matching seeds. 5 cells (one per seed).

Env:
  SEED: 22|23|14|1|2  (Phase 5 seed pool)
"""
import os
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = int(os.environ["SEED"])

COHORT_DIR = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
OUT_ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase6_p5xgb_mace_v1")
OUT_DIR = OUT_ROOT / f"seed_{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Idempotent skip
metrics_path = OUT_DIR / "metrics.json"
if metrics_path.exists():
    try:
        m = json.loads(metrics_path.read_text())
        if len(m) == 8:
            print(f"SKIP seed={SEED} (metrics.json complete)")
            raise SystemExit(0)
    except Exception:
        pass

TARGET_COLS = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
               "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]

XGB_HPS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1,
    "early_stopping_rounds": 50, "tree_method": "hist",
    "random_state": SEED, "verbosity": 0, "n_jobs": 4,
}

EPOCHS_MAX = 100_000
PATIENCE = 10_000
BATCH = 16
LR = 1e-5
WEIGHT_DECAY = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dft_features(df):
    remove = [c for c in df.columns if "_dft" in c]
    y = df[remove].copy()
    non_num = ["reaction_number", "elements_frag1", "elements_frag2"]
    X = df.drop(columns=remove)
    drop = [c for c in non_num if c in X.columns]
    X = X.drop(columns=drop).select_dtypes(include=[np.number])
    return X, y


def pad_mace(feats_list, max_atoms):
    N, D = len(feats_list), feats_list[0].shape[1]
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
        d = torch.load(mace_dir / f"{int(rxn):04d}.pt", map_location="cpu")
        feats.append(d["mace_feats"])
    max_atoms = max(f.shape[0] for f in feats)
    return pad_mace(feats, max_atoms)


class ResidualNet(nn.Module):
    def __init__(self, in_dim=256, hidden=128, n_targets=8, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.attn_query = nn.Parameter(torch.randn(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, n_targets),
        )

    def forward(self, mace_feats, mask):
        h = self.proj(mace_feats)
        scores = torch.einsum("bnh,h->bn", h, self.attn_query)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        v = torch.einsum("bnh,bn->bh", h, weights)
        return self.head(v)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_pickle(COHORT_DIR / "phase5_dataset_v2.pkl")
    df["reaction_number"] = df["reaction_number"].astype(int)
    assert len(df) == 3504
    print(f"Loaded phase5_dataset_v2: {df.shape}")

    X_all, y_all = get_dft_features(df)
    for t in TARGET_COLS:
        assert t in y_all.columns, f"missing target {t}"
    y_all = y_all[TARGET_COLS]
    rxn_all = df["reaction_number"].values

    # Paper 80/10/10 split (matches xgb_one_seed.py)
    X_tr, X_te, y_tr, y_te, rxn_tr, rxn_te = train_test_split(
        X_all, y_all, rxn_all, test_size=0.2, random_state=SEED)
    X_te, X_va, y_te, y_va, rxn_te, rxn_va = train_test_split(
        X_te, y_te, rxn_te, test_size=0.5, random_state=SEED)
    print(f"train/val/test: {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    sc_X = StandardScaler()
    X_tr_s = sc_X.fit_transform(X_tr)
    X_va_s = sc_X.transform(X_va)
    X_te_s = sc_X.transform(X_te)

    # ---- Stage A: XGB baseline ----
    print("\n=== Stage A: XGB baseline (phase5 features) ===")
    xgb_models = {}
    yhat_train = np.zeros((len(X_tr), len(TARGET_COLS)), dtype=np.float32)
    yhat_test = np.zeros((len(X_te), len(TARGET_COLS)), dtype=np.float32)
    for i, t in enumerate(TARGET_COLS):
        y_tr_raw = y_tr[t].values.reshape(-1, 1)
        y_va_raw = y_va[t].values.reshape(-1, 1)
        sc_y = StandardScaler()
        y_tr_s = sc_y.fit_transform(y_tr_raw).ravel()
        y_va_s = sc_y.transform(y_va_raw).ravel()

        m = XGBRegressor(**XGB_HPS)
        m.fit(X_tr_s, y_tr_s, eval_set=[(X_va_s, y_va_s)], verbose=False)

        yhat_train[:, i] = sc_y.inverse_transform(m.predict(X_tr_s).reshape(-1, 1)).ravel()
        yhat_test[:, i] = sc_y.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()
        xgb_models[t] = (m, sc_y)
        mae = mean_absolute_error(y_te[t].values, yhat_test[:, i])
        print(f"  {t}: XGB baseline MAE = {mae:.4f}")

    y_tr_np = y_tr.values.astype(np.float32)
    y_te_np = y_te.values.astype(np.float32)

    # ---- Stage B: MACE residual on train only ----
    print("\n=== Stage B: MACE residual network ===")
    r_train = y_tr_np - yhat_train
    sigma_c = r_train.std(axis=0) + 1e-8
    r_train_normed = r_train / sigma_c

    mace_train, mask_train = load_mace_features(rxn_tr.tolist())
    mace_test, mask_test = load_mace_features(rxn_te.tolist())
    mace_train = mace_train.to(device); mask_train = mask_train.to(device)
    mace_test = mace_test.to(device); mask_test = mask_test.to(device)
    r_train_t = torch.from_numpy(r_train_normed).to(device)

    net = ResidualNet(in_dim=mace_train.shape[-1], n_targets=len(TARGET_COLS)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_train = len(X_tr)
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

    net.eval()
    with torch.no_grad():
        delta_test_normed = net(mace_test, mask_test).cpu().numpy()
    delta_test = delta_test_normed * sigma_c
    y_pred = yhat_test + delta_test

    metrics = {}
    for i, t in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_te_np[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_te_np[:, i], y_pred[:, i]))
        r2 = r2_score(y_te_np[:, i], y_pred[:, i])
        metrics[t] = {
            "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
            "xgb_baseline_mae": float(mean_absolute_error(y_te_np[:, i], yhat_test[:, i])),
        }
        print(f"  {t}: final MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.3f}")

    with open(OUT_DIR / "xgb_models.pkl", "wb") as f:
        pickle.dump(xgb_models, f)
    torch.save(net.state_dict(), OUT_DIR / "delta_net.pt")
    pd.DataFrame({
        "reaction_number": np.repeat(rxn_te, len(TARGET_COLS)),
        "target": TARGET_COLS * len(X_te),
        "y_true": y_te_np.flatten(),
        "y_pred": y_pred.flatten(),
        "y_baseline": yhat_test.flatten(),
        "delta": delta_test.flatten(),
    }).to_parquet(OUT_DIR / "predictions.parquet", index=False)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
