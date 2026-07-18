"""SPEC_06 B1 — family-restricted 2-step (xgb28 base + δ) trainer, one cell.

Same math and hyperparameters as train_xgb28_delta.py, but with training
restricted to a single family's ~150 reactions.

Differences from spec06 main trainer:
  - Splits come from splits/family_folds/{family}_outer_folds.json (5-fold KFold).
  - X28 / bundle are pre-sliced to family rxns before the fold indexing.
  - Cross-fit gate #3: soft (warn + record) instead of raise, since with ~130
    inner-train rxns the residuals can be genuinely small without meaning the
    δ target is degenerate. If med|r_train| < 5% of med|y − ȳ|, we still train
    δ but tag `cross_fit_gate` in the JSON so aggregate can flag those cells.

CLI:
  --family FAM      (dipolar | qmrxn20_e2 | qmrxn20_sn2 | rgd1)
  --fold N          (0..4; default $SLURM_ARRAY_TASK_ID)
  --member N        (default 0)
  --device cuda|cpu

Idempotent: skip if oof/family_restricted/{fam}/fold{f}/member{m}.json exists.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC = REPO / "spec/spec09_per_family_xgb28_delta"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
sys.path.insert(0, str(SPEC / "code"))

from eda_asm.asr_v1.models_delta import ModelM1Delta
from eda_asm.asr_v1.baseline_physics import LinearBaseline
from eda_asm.asr_v1.training_delta import (
    CachedFeatureBundleDelta, TrainConfigDelta, train_one_model_delta,
)
from baselines import cross_fit_oof, fit_full, predict_full  # noqa: E402
from descriptors28 import build_X28  # noqa: E402

BUNDLE_PT = Path(os.environ.get(
    "BUNDLE_PT",
    "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt"))
FAM_SPLITS_DIR = SPEC / "splits/family_folds"
OUT_ROOT = SPEC / "oof/family_restricted"

LR = 1.0e-5
EPOCHS_MAX = 100_000
PATIENCE = 10_000
BATCH = 16
WD = 1.0e-3
M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
SEED_BASE = 42
CHANNELS = ["strain", "Pauli", "elst", "oi", "disp"]


def slice_bundle(bundle, rids_keep_set):
    keep = [i for i, r in enumerate(bundle.reaction_ids) if r in rids_keep_set]
    return CachedFeatureBundleDelta(
        reaction_ids=[bundle.reaction_ids[i] for i in keep],
        R_features=[bundle.R_features[i] for i in keep],
        TS_features=[bundle.TS_features[i] for i in keep],
        P_features=[bundle.P_features[i] for i in keep],
        labels=bundle.labels[keep],
        descriptors=bundle.descriptors[keep],
        feature_dim=bundle.feature_dim,
    )


def train_one_family_cell(bundle_full, X_fam, Y_fam, rids_fam,
                          family, fold_idx, tr_idx, te_idx,
                          member, device):
    # seed convention: family-hash to keep seeds distinct across families
    seed = SEED_BASE + member * 1000 + fold_idx * 100 + 50 + (hash(family) % 47) * 10
    print(f"[{family} f{fold_idx} m{member}] seed={seed}", flush=True)

    b_oof = cross_fit_oof("xgb", X_fam[tr_idx], Y_fam[tr_idx], K=5, seed=seed)
    r_train = Y_fam[tr_idx] - b_oof
    med_r = float(np.median(np.abs(r_train)))
    med_y = float(np.median(np.abs(Y_fam[tr_idx] - Y_fam[tr_idx].mean(0))))
    gate_ratio = med_r / max(med_y, 1e-9)
    gate_ok = med_r >= 0.05 * med_y
    print(f"  med|r_train|={med_r:.3f}  med|y-ȳ|={med_y:.3f}  "
          f"ratio={gate_ratio:.3f}  gate_ok={gate_ok}", flush=True)
    if not gate_ok:
        print(f"  WARN: cross-fit gate #3 tripped (ratio<{0.05}); "
              f"continuing (soft gate, family-restricted).", flush=True)

    b_full = fit_full("xgb", X_fam[tr_idx], Y_fam[tr_idx])
    b_val = predict_full("xgb", b_full, X_fam[te_idx])

    for c_i, ch in enumerate(CHANNELS):
        yt_c = Y_fam[tr_idx][:, c_i]; yp_c = b_oof[:, c_i]
        mad = float(np.mean(np.abs(yt_c - yt_c.mean())))
        nm = float(np.mean(np.abs(yt_c - yp_c)) / (mad + 1e-12))
        print(f"    b_oof NMAE[{ch}] = {nm:.3f}", flush=True)

    rids_tr = set(rids_fam[tr_idx].tolist()); rids_te = set(rids_fam[te_idx].tolist())
    keep_set = rids_tr | rids_te
    bundle = slice_bundle(bundle_full, keep_set)
    r2i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    train_positions_all = [r2i[r] for r in rids_fam[tr_idx] if r in r2i]
    test_positions = [r2i[r] for r in rids_fam[te_idx] if r in r2i]

    rng = np.random.default_rng(seed)
    arr = list(train_positions_all); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    train_pos, val_pos = arr[n_val:], arr[:n_val]

    injected_train = {rids_fam[tr_idx][i]: b_oof[i] for i in range(len(tr_idx))}
    injected_test  = {rids_fam[te_idx][i]: b_val[i] for i in range(len(te_idx))}
    injected_all = {**injected_train, **injected_test}

    class FixedBaseline(LinearBaseline):
        def __init__(self, rid_map, bundle_rids):
            super().__init__(alpha=1.0)
            arr = np.array([rid_map[r] for r in bundle_rids], dtype=np.float32)
            self._arr = arr
            self.W_ = np.zeros((1, 5), dtype=np.float32)
            self.d_in_ = 1
            self.d_mean_ = np.zeros(1, dtype=np.float32)
            self.d_std_ = np.ones(1, dtype=np.float32)

        def fit(self, D_train, Y_train): return self
        def predict(self, D):
            n = D.shape[0]
            if n == self._arr.shape[0]:
                return self._arr.copy()
            return np.zeros((n, 5), dtype=np.float32)

    from eda_asm.asr_v1 import training_delta as tdmod
    orig_lb = tdmod.LinearBaseline
    tdmod.LinearBaseline = lambda **kw: FixedBaseline(injected_all, bundle.reaction_ids)

    try:
        factory = lambda F: ModelM1Delta(feature_dim=F, **M1_HP)
        cfg = TrainConfigDelta(
            epochs=EPOCHS_MAX, batch_size=BATCH, lr=LR, weight_decay=WD,
            early_stop_patience=PATIENCE, device=device, baseline_ridge_alpha=1.0,
        )
        model, fr = train_one_model_delta(bundle, factory, train_pos, val_pos, cfg, seed=seed)
    finally:
        tdmod.LinearBaseline = orig_lb

    model.eval()
    yt = bundle.labels[test_positions].numpy()
    baseline_test = np.array([injected_test[r] for r in
                              [bundle.reaction_ids[i] for i in test_positions]])
    preds = []
    with torch.no_grad():
        for i in test_positions:
            r_f = bundle.R_features[i].to(device).unsqueeze(0)
            t_f = bundle.TS_features[i].to(device).unsqueeze(0)
            p_f = bundle.P_features[i].to(device).unsqueeze(0)
            r_m = torch.ones(r_f.shape[:2], dtype=torch.bool, device=device)
            t_m = torch.ones(t_f.shape[:2], dtype=torch.bool, device=device)
            p_m = torch.ones(p_f.shape[:2], dtype=torch.bool, device=device)
            delta = model(r_f, r_m, t_f, t_m, p_f, p_m).cpu().numpy().flatten()
            preds.append(delta)
    delta_test = np.array(preds)
    yp = baseline_test + delta_test
    return (yt, yp,
            [bundle.reaction_ids[i] for i in test_positions],
            baseline_test, delta_test,
            fr, med_r, med_y, gate_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["dipolar", "qmrxn20_e2", "qmrxn20_sn2", "rgd1"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.fold is None:
        args.fold = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    print(f"[B1] family={args.family}  fold={args.fold}  member={args.member}  device={args.device}",
          flush=True)

    out_dir = OUT_ROOT / args.family / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member{args.member}.json"
    if out_path.exists():
        print(f"[skip] {out_path}", flush=True); return

    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    all_rids = np.asarray(b["reaction_ids"])
    X24_all = b["descriptors"].numpy().astype(np.float64)
    Y_all = b["labels"].numpy().astype(np.float64)
    X28_all, ok_all = build_X28(all_rids, X24_all)

    with open(FAM_SPLITS_DIR / f"{args.family}_outer_folds.json") as fh:
        fam_folds = json.load(fh)
    fam_rids = np.asarray(fam_folds["all_rids"])
    n_fam = len(fam_rids)
    # positions of family rxns in the master arrays (order preserved from fam_folds.all_rids)
    all_r2i = {r: i for i, r in enumerate(all_rids)}
    fam_pos = np.array([all_r2i[r] for r in fam_rids])
    X_fam = X28_all[fam_pos]; Y_fam = Y_all[fam_pos]
    n_zero_fill_fam = int((~ok_all[fam_pos]).sum())
    print(f"  family cohort n={n_fam}  proxies_zero_filled={n_zero_fill_fam}", flush=True)

    fam_r2i = {r: i for i, r in enumerate(fam_rids)}
    tr_rids = fam_folds["folds"][str(args.fold)]["train"]
    te_rids = fam_folds["folds"][str(args.fold)]["test"]
    tr_idx = np.array([fam_r2i[r] for r in tr_rids])
    te_idx = np.array([fam_r2i[r] for r in te_rids])
    print(f"  train={len(tr_idx)}  test={len(te_idx)}", flush=True)

    bundle_full = CachedFeatureBundleDelta(
        reaction_ids=b["reaction_ids"],
        R_features=b["R_features"], TS_features=b["TS_features"], P_features=b["P_features"],
        labels=b["labels"].float(), descriptors=b["descriptors"].float(),
        feature_dim=int(b["feature_dim"]),
    )

    t0 = time.time()
    (yt, yp, te_rids_ordered, b_val, delta_test, fr,
     med_r, med_y, gate_ok) = train_one_family_cell(
        bundle_full, X_fam, Y_fam, fam_rids,
        args.family, args.fold, tr_idx, te_idx, args.member, args.device,
    )
    bt = yt.sum(axis=1); bp = yp.sum(axis=1)
    result = {
        "arm": "xgb28_delta_family",
        "family": args.family,
        "fold": args.fold, "member": args.member,
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "reaction_ids": te_rids_ordered,
        **{f"y_true_{c}": [float(v) for v in yt[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"y_pred_{c}": [float(v) for v in yp[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"b_test_{c}": [float(v) for v in b_val[:, i]] for i, c in enumerate(CHANNELS)},
        **{f"delta_test_{c}": [float(v) for v in delta_test[:, i]] for i, c in enumerate(CHANNELS)},
        "barrier_true": [float(v) for v in bt],
        "barrier_pred": [float(v) for v in bp],
        "best_epoch": int(fr.best_epoch), "final_epoch": int(fr.final_epoch),
        "early_stopped": bool(fr.early_stopped),
        "cross_fit_median_r_train": med_r,
        "cross_fit_median_y_bar": med_y,
        "cross_fit_gate_ok": bool(gate_ok),
        "proxies_zero_filled_in_family": n_zero_fill_fam,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path} ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
