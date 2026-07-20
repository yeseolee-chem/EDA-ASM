"""SPEC_08 whole-dataset LC — one (size, fold, member) cell.

Trains both arms on the same size-N whole-cohort family-stratified train
subset and evaluates on the fold's fixed test:
  - xgb28_base:  fit XGB on X28[train_subset], predict on X28[test]
  - xgb28_delta: fit XGB with cross-fit-oof on X28[train_subset] for the
                 δ target, train ModelM1Delta on MACE features of
                 train_subset (spec06 recipe), predict ŷ = b_val + δ.

Idempotent: skips if oof/size{N}/fold{f}/member{m}.json exists.

CLI:
  --size N       target size (must be in splits/lc_splits.json's sizes)
  --fold F       0..4
  --member M     default 0
  --device       default cuda if available
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
SPEC = REPO / "spec/spec08_whole_dataset_learning_curve"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "spec/spec02_abc_ablation/code"))
sys.path.insert(0, str(REPO / "spec/spec06_2step_xgb28_delta/code"))

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
LC_SPLITS = SPEC / "splits/lc_splits.json"
OUT_ROOT = SPEC / "oof"

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


def channel_metrics(y_true, y_pred):
    out = {}
    for c_i, ch in enumerate(CHANNELS):
        yt = y_true[:, c_i]; yp = y_pred[:, c_i]
        mad = float(np.mean(np.abs(yt - yt.mean())))
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        out[f"nmae_{ch}"] = mae / (mad + 1e-12)
        out[f"mae_{ch}"] = mae
        out[f"rmse_{ch}"] = rmse
    bt = y_true.sum(axis=1); bp = y_pred.sum(axis=1)
    mad_b = float(np.mean(np.abs(bt - bt.mean())))
    mae_b = float(np.mean(np.abs(bt - bp)))
    rmse_b = float(np.sqrt(np.mean((bt - bp) ** 2)))
    out["nmae_barrier"] = mae_b / (mad_b + 1e-12)
    out["mae_barrier"] = mae_b
    out["rmse_barrier"] = rmse_b
    return out


def train_delta_arm(bundle_full, rids_all, X28_all, Y_all, tr_idx, te_idx,
                    seed, device):
    b_oof = cross_fit_oof("xgb", X28_all[tr_idx], Y_all[tr_idx], K=5, seed=seed)
    r_train = Y_all[tr_idx] - b_oof
    med_r = float(np.median(np.abs(r_train)))
    med_y = float(np.median(np.abs(Y_all[tr_idx] - Y_all[tr_idx].mean(0))))
    gate_ratio = med_r / max(med_y, 1e-9)
    gate_ok = med_r >= 0.05 * med_y
    print(f"  δ cross-fit gate: ratio={gate_ratio:.3f}  gate_ok={gate_ok} (soft)",
          flush=True)

    b_full = fit_full("xgb", X28_all[tr_idx], Y_all[tr_idx])
    b_val = predict_full("xgb", b_full, X28_all[te_idx])

    rids_tr = set(rids_all[tr_idx].tolist()); rids_te = set(rids_all[te_idx].tolist())
    keep_set = rids_tr | rids_te
    bundle = slice_bundle(bundle_full, keep_set)
    r2i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    train_positions_all = [r2i[r] for r in rids_all[tr_idx] if r in r2i]
    test_positions = [r2i[r] for r in rids_all[te_idx] if r in r2i]

    rng = np.random.default_rng(seed)
    arr = list(train_positions_all); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    train_pos, val_pos = arr[n_val:], arr[:n_val]

    injected_train = {rids_all[tr_idx][i]: b_oof[i] for i in range(len(tr_idx))}
    injected_test = {rids_all[te_idx][i]: b_val[i] for i in range(len(te_idx))}
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

        def fit(self, D_train, Y_train):
            return self

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
    te_rids_ord = [bundle.reaction_ids[i] for i in test_positions]
    baseline_test = np.array([injected_test[r] for r in te_rids_ord])
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
    return {
        "b_val": b_val, "delta_test": delta_test, "yp": yp,
        "te_rids_ordered": te_rids_ord,
        "fit_result": fr,
        "cross_fit_median_r_train": med_r,
        "cross_fit_median_y_bar": med_y,
        "cross_fit_gate_ok": gate_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[spec08 whole LC] size={args.size} fold={args.fold} "
          f"member={args.member} device={args.device}", flush=True)

    out_dir = OUT_ROOT / f"size{args.size}" / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member{args.member}.json"
    if out_path.exists():
        print(f"[skip] {out_path}", flush=True); return

    with open(LC_SPLITS) as fh:
        lc = json.load(fh)
    if args.size not in lc["sizes"]:
        raise ValueError(f"size {args.size} not in {lc['sizes']}")
    sub = lc["subsamples"][str(args.size)][str(args.fold)]
    train_rids = sub["train_rids"]
    test_rids = lc["folds"][str(args.fold)]["test"]
    print(f"  train_n={len(train_rids)} (target={args.size} "
          f"capped={sub['capped_to_fold_train']})  test_n={len(test_rids)}",
          flush=True)
    print(f"  per_family={sub['per_family']}", flush=True)

    b = torch.load(str(BUNDLE_PT), weights_only=False, map_location="cpu")
    rids = np.asarray(b["reaction_ids"])
    X24 = b["descriptors"].numpy().astype(np.float64)
    Y = b["labels"].numpy().astype(np.float64)
    X28, ok = build_X28(rids, X24)

    tr_idx = np.array([np.where(rids == r)[0][0] for r in train_rids])
    te_idx = np.array([np.where(rids == r)[0][0] for r in test_rids])

    size_idx = lc["sizes"].index(args.size)
    seed = (SEED_BASE + args.member * 10_000 + args.fold * 1000
            + size_idx * 100 + 50)
    print(f"  seed={seed}", flush=True)

    bundle_full = CachedFeatureBundleDelta(
        reaction_ids=b["reaction_ids"],
        R_features=b["R_features"], TS_features=b["TS_features"], P_features=b["P_features"],
        labels=b["labels"].float(), descriptors=b["descriptors"].float(),
        feature_dim=int(b["feature_dim"]),
    )

    t0 = time.time()

    # ---- arm 1: xgb28_base ----
    m_base = fit_full("xgb", X28[tr_idx], Y[tr_idx])
    b_test_base = predict_full("xgb", m_base, X28[te_idx])
    Y_test = Y[te_idx]
    base_metrics = channel_metrics(Y_test, b_test_base)
    print(f"  [xgb_base] NMAE barrier={base_metrics['nmae_barrier']:.3f}", flush=True)

    # ---- arm 2: xgb28_delta ----
    delta_res = train_delta_arm(
        bundle_full, rids, X28, Y, tr_idx, te_idx, seed, args.device,
    )
    te_rids_ord = delta_res["te_rids_ordered"]
    te_idx_ord = np.array([np.where(rids == r)[0][0] for r in te_rids_ord])
    Y_test_ord = Y[te_idx_ord]
    delta_metrics = channel_metrics(Y_test_ord, delta_res["yp"])
    print(f"  [xgb_delta] NMAE barrier={delta_metrics['nmae_barrier']:.3f}", flush=True)

    # reorder xgb_base for direct comparison
    b_test_base_ord = predict_full("xgb", m_base, X28[te_idx_ord])
    base_metrics = channel_metrics(Y_test_ord, b_test_base_ord)

    fr = delta_res["fit_result"]
    result = {
        "spec": "spec08_whole_lc",
        "size_target": args.size,
        "size_actual": int(len(tr_idx)),
        "capped_to_fold_train": bool(sub["capped_to_fold_train"]),
        "per_family": sub["per_family"],
        "fold": args.fold, "member": args.member, "seed": seed,
        "reaction_ids_test": te_rids_ord,
        "xgb_base": {
            "y_pred": {ch: [float(v) for v in b_test_base_ord[:, i]]
                       for i, ch in enumerate(CHANNELS)},
            "metrics": base_metrics,
        },
        "xgb_delta": {
            "b_test": {ch: [float(v) for v in delta_res["b_val"][:, i]]
                       for i, ch in enumerate(CHANNELS)},
            "delta_test": {ch: [float(v) for v in delta_res["delta_test"][:, i]]
                           for i, ch in enumerate(CHANNELS)},
            "y_pred": {ch: [float(v) for v in delta_res["yp"][:, i]]
                       for i, ch in enumerate(CHANNELS)},
            "metrics": delta_metrics,
            "best_epoch": int(fr.best_epoch), "final_epoch": int(fr.final_epoch),
            "early_stopped": bool(fr.early_stopped),
            "cross_fit_median_r_train": delta_res["cross_fit_median_r_train"],
            "cross_fit_median_y_bar": delta_res["cross_fit_median_y_bar"],
            "cross_fit_gate_ok": bool(delta_res["cross_fit_gate_ok"]),
        },
        "y_true": {ch: [float(v) for v in Y_test_ord[:, i]]
                   for i, ch in enumerate(CHANNELS)},
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path} ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
