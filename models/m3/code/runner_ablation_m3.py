"""spec2 ablation runner for m3: probe the delta / b decomposition.

Ablation modes (env MODE):
  full           y = b + delta                 (reference; reuses m3 arch)
  delta_only     y = delta        (baseline pinned to 0)
  baseline_only  y = b            (no ML head, ridge only)

Fold is fixed to 0 (single-fold speed-up). Members 0..4 give ensemble std.

Output:
  m3/code/trackB_ablation/{MODE}/fold0/member{M}.json

SLURM_ARRAY_TASK_ID -> member (0..4) for delta_only / full.
baseline_only runs all 5 members in a single CPU task.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent   # models/m3/code -> models/m3 -> models -> repo root

sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.asr_v1.data import ASR_COMPONENTS
from eda_asm.asr_v1.models_delta import ModelM1Delta
from eda_asm.asr_v1.baseline_physics import LinearBaseline
from eda_asm.asr_v1.training_delta import (
    CachedFeatureBundleDelta, TrainConfigDelta, FoldResultDelta,
    _IndexedDatasetDelta, _collate_delta, _component_mae,
)

MODE = os.environ.get("MODE", "full")
assert MODE in {"full", "delta_only", "baseline_only"}, f"bad MODE={MODE}"

BASELINE = os.environ.get("BASELINE", "xtb_geom6_plus_v2")
BUNDLE = HERE / "bundles" / f"features_v6_delta_{BASELINE}.pt"
SUBSAMPLES_TAG = os.environ.get("SUBSAMPLES_TAG", "v9_all")
SUBSAMPLES = ROOT / "outputs/asr_v1/phase3/subsamples" / SUBSAMPLES_TAG
OUT_TAG = os.environ.get("OUT_TAG", "ablation")
# For baseline_only, tag the output subdir with the ridge alpha so a grid
# sweep can save all alphas side-by-side without clobbering.
if MODE == "baseline_only":
    _alpha_tag = os.environ.get("RIDGE_ALPHA_TAG",
                                f"a{float(os.environ.get('RIDGE_ALPHA', 1.0)):g}")
    OUT_BASE = HERE / f"trackB_{OUT_TAG}" / MODE / _alpha_tag
else:
    OUT_BASE = HERE / f"trackB_{OUT_TAG}" / MODE

LR_LOW = float(os.environ.get("LR_LOW", 1.0e-5))
EPOCHS_MAX = int(os.environ.get("EPOCHS_MAX", 100_000))
PATIENCE = int(os.environ.get("PATIENCE", 10_000))
BATCH = 16
WD = 1.0e-3

M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", 1.0))
SIZE_FULL = int(os.environ.get("SIZE_FULL", 626))
SEED_BASE = 42


def slice_delta(bundle: CachedFeatureBundleDelta, rids: list[str]) -> CachedFeatureBundleDelta:
    rid_to_i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    keep = [rid_to_i[r] for r in rids if r in rid_to_i]
    return CachedFeatureBundleDelta(
        reaction_ids=[bundle.reaction_ids[i] for i in keep],
        R_features=[bundle.R_features[i] for i in keep],
        TS_features=[bundle.TS_features[i] for i in keep],
        P_features=[bundle.P_features[i] for i in keep],
        labels=bundle.labels[keep],
        descriptors=bundle.descriptors[keep],
        feature_dim=bundle.feature_dim,
    )


def load_indices_trackB(fold: int):
    fold_dir = SUBSAMPLES / f"fold{fold}"
    test_rids = json.load(open(fold_dir / "test_rids.json"))
    p_default = fold_dir / f"size_{SIZE_FULL}.json"
    if p_default.exists():
        train_rids = json.load(open(p_default))
    else:
        candidates = sorted(fold_dir.glob("size_*.json"),
                            key=lambda p: int(p.stem.split("_")[1]),
                            reverse=True)
        if not candidates:
            raise FileNotFoundError(f"no size_*.json in {fold_dir}")
        train_rids = json.load(open(candidates[0]))
    return train_rids, test_rids


def make_val_split(train_pos, fold, member):
    rng = np.random.default_rng(SEED_BASE + fold * 100 + member * 17)
    arr = list(train_pos); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    return arr[n_val:], arr[:n_val]


# --- ablation-aware training loop -------------------------------------------
def train_one_model_ablation(
    bundle: CachedFeatureBundleDelta,
    model_factory: Callable[[int], nn.Module],
    train_idx: list[int],
    val_idx: list[int],
    cfg: TrainConfigDelta,
    baseline_all_np: np.ndarray,   # (N, 5) — may be all zeros for delta_only
    baseline_state: dict,          # for on-disk serialization only
    seed: int = 0,
) -> tuple[nn.Module, FoldResultDelta]:
    torch.manual_seed(seed); np.random.seed(seed)
    Y_np = bundle.labels.numpy()
    baseline_all = torch.from_numpy(baseline_all_np).float()
    val_base_mae = np.abs(baseline_all_np[val_idx] - Y_np[val_idx]).mean(axis=0)

    model = model_factory(bundle.feature_dim).to(cfg.device)
    if hasattr(model, "input_std"):
        train_features = (
            [bundle.R_features[i] for i in train_idx]
            + [bundle.P_features[i] for i in train_idx]
        )
        model.input_std.fit_from(train_features)

    sigma_c = bundle.labels[train_idx].std(dim=0).clamp_min(1e-6).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    train_ds = _IndexedDatasetDelta(bundle, train_idx, baseline_all)
    val_ds = _IndexedDatasetDelta(bundle, val_idx, baseline_all)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=_collate_delta, drop_last=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=_collate_delta, num_workers=0)

    best_val = float("inf"); best_state = None
    best_epoch = -1; final_epoch = 0; early_stopped = False
    epochs_since_best = 0; history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_seen = 0.0, 0
        for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in train_dl:
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            y = y.to(cfg.device); b = b.to(cfg.device)
            delta = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
            pred = b + delta
            loss = (pred - y).abs().div(sigma_c).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()
            ep_loss += loss.item() * y.shape[0]; n_seen += y.shape[0]
        train_mae = ep_loss / max(n_seen, 1)

        model.eval()
        vp_list, vt_list = [], []
        with torch.no_grad():
            for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in val_dl:
                R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
                T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
                P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
                b = b.to(cfg.device)
                delta = model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)
                pred = b + delta
                vp_list.append(pred.cpu()); vt_list.append(y)
        vp = torch.cat(vp_list, dim=0); vt = torch.cat(vt_list, dim=0)
        per_comp = _component_mae(vp, vt)
        val_mae = per_comp.mean().item()

        history.append({"epoch": epoch, "train_mae": train_mae,
                        "val_mae": val_mae,
                        "val_mae_per_comp": per_comp.tolist()})
        final_epoch = epoch
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stop_patience:
                early_stopped = True; break

    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    vp_list, vt_list = [], []
    with torch.no_grad():
        for R_feat, R_mask, T_feat, T_mask, P_feat, P_mask, y, b in val_dl:
            R_feat, R_mask = R_feat.to(cfg.device), R_mask.to(cfg.device)
            T_feat, T_mask = T_feat.to(cfg.device), T_mask.to(cfg.device)
            P_feat, P_mask = P_feat.to(cfg.device), P_mask.to(cfg.device)
            b = b.to(cfg.device)
            vp_list.append(
                (b + model(R_feat, R_mask, T_feat, T_mask, P_feat, P_mask)).cpu()
            )
            vt_list.append(y)
    vp = torch.cat(vp_list, dim=0); vt = torch.cat(vt_list, dim=0)
    per_comp = _component_mae(vp, vt).numpy()
    result = FoldResultDelta(
        train_indices=train_idx, val_indices=val_idx,
        val_mae_per_component=per_comp,
        val_mae_overall=float(per_comp.mean()),
        val_mae_baseline_only=val_base_mae,
        best_epoch=best_epoch, final_epoch=final_epoch,
        early_stopped=early_stopped, history=history,
        baseline_state=baseline_state,
    )
    return model, result


def evaluate_predictions(model, bundle, test_pos, baseline_pred_np, device):
    model.eval()
    y_true = bundle.labels[test_pos].numpy()
    preds = []
    with torch.no_grad():
        for i in test_pos:
            r_f = bundle.R_features[i].to(device).unsqueeze(0)
            t_f = bundle.TS_features[i].to(device).unsqueeze(0)
            p_f = bundle.P_features[i].to(device).unsqueeze(0)
            r_m = torch.ones(r_f.shape[:2], dtype=torch.bool, device=device)
            t_m = torch.ones(t_f.shape[:2], dtype=torch.bool, device=device)
            p_m = torch.ones(p_f.shape[:2], dtype=torch.bool, device=device)
            delta = model(r_f, r_m, t_f, t_m, p_f, p_m).cpu().numpy().flatten()
            preds.append(baseline_pred_np[i] + delta)
    y_pred = np.array(preds)
    test_mae = np.abs(y_pred - y_true).mean(axis=0)
    return y_true, y_pred, test_mae


def run_one(args):
    assert 0 <= args.fold <= 4 and 0 <= args.member <= 4
    out_dir = OUT_BASE / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member{args.member}.json"
    if out_path.exists():
        print(f"[skip] {out_path} exists", flush=True); return

    print(f"=== MODE={MODE} fold={args.fold} member={args.member} | "
          f"baseline={BASELINE} lr={LR_LOW} epochs={EPOCHS_MAX} pat={PATIENCE} ===",
          flush=True)
    t0 = time.time()

    full = CachedFeatureBundleDelta.load(BUNDLE)
    print(f"  bundle: {len(full.reaction_ids)} rxns, feat_dim={full.feature_dim}",
          flush=True)

    train_rids, test_rids = load_indices_trackB(args.fold)
    keep = list(set(train_rids) | set(test_rids))
    bundle = slice_delta(full, keep)
    rid_to_i = {r: i for i, r in enumerate(bundle.reaction_ids)}
    train_pos_all = [rid_to_i[r] for r in train_rids if r in rid_to_i]
    test_pos = [rid_to_i[r] for r in test_rids if r in rid_to_i]
    train_pos, val_pos = make_val_split(train_pos_all, args.fold, args.member)
    print(f"  N: train={len(train_pos)} val={len(val_pos)} test={len(test_pos)}",
          flush=True)

    D_np = bundle.descriptors.numpy()
    Y_np = bundle.labels.numpy()

    # --- build baseline vector according to MODE
    if MODE in {"full", "baseline_only"}:
        bl = LinearBaseline(alpha=RIDGE_ALPHA).fit(D_np[train_pos], Y_np[train_pos])
        baseline_all_np = bl.predict(D_np)
        baseline_state = bl.state_dict()
    elif MODE == "delta_only":
        baseline_all_np = np.zeros((D_np.shape[0], 5), dtype=np.float32)
        baseline_state = {"alpha": RIDGE_ALPHA, "W": None,
                          "d_in": D_np.shape[1], "d_mean": None, "d_std": None,
                          "note": "zeroed for delta_only ablation"}
    else:
        raise ValueError(MODE)

    seed = SEED_BASE + args.member * 1000 + args.fold * 100

    if MODE == "baseline_only":
        # No ML — just baseline on test.
        y_true = bundle.labels[test_pos].numpy()
        y_pred = baseline_all_np[test_pos]
        test_mae = np.abs(y_pred - y_true).mean(axis=0)
        val_true = bundle.labels[val_pos].numpy()
        val_pred = baseline_all_np[val_pos]
        val_mae_pc = np.abs(val_pred - val_true).mean(axis=0)
        result_extra = dict(
            val_mae_per_channel=list(map(float, val_mae_pc)),
            val_mae_baseline_only=list(map(float, val_mae_pc)),
            best_epoch=0, final_epoch=0, early_stopped=False,
        )
    else:
        cfg = TrainConfigDelta(
            epochs=EPOCHS_MAX, batch_size=BATCH, lr=LR_LOW,
            weight_decay=WD, early_stop_patience=PATIENCE,
            device=args.device, baseline_ridge_alpha=RIDGE_ALPHA,
        )
        factory = lambda F: ModelM1Delta(feature_dim=F, **M1_HP)
        model, fr = train_one_model_ablation(
            bundle, factory, train_pos, val_pos, cfg,
            baseline_all_np, baseline_state, seed=seed,
        )
        y_true, y_pred, test_mae = evaluate_predictions(
            model, bundle, test_pos, baseline_all_np, args.device,
        )
        result_extra = dict(
            val_mae_per_channel=list(map(float, fr.val_mae_per_component)),
            val_mae_baseline_only=list(map(float, fr.val_mae_baseline_only)),
            best_epoch=int(fr.best_epoch),
            final_epoch=int(fr.final_epoch),
            early_stopped=bool(fr.early_stopped),
        )

    test_rids_in_order = [bundle.reaction_ids[i] for i in test_pos]
    barrier_true = y_true.sum(axis=1)
    barrier_pred = y_pred.sum(axis=1)
    barrier_mae = float(np.abs(barrier_pred - barrier_true).mean())
    barrier_rmse = float(np.sqrt(np.mean((barrier_pred - barrier_true) ** 2)))

    result = {
        "mode": MODE,
        "fold": args.fold, "member": args.member, "seed": seed,
        "n_train": len(train_pos), "n_val": len(val_pos), "n_test": len(test_pos),
        "reaction_ids": test_rids_in_order,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "components": list(ASR_COMPONENTS),
        "barrier_true": barrier_true.tolist(),
        "barrier_pred": barrier_pred.tolist(),
        "test_barrier_mae": barrier_mae,
        "test_barrier_rmse": barrier_rmse,
        "test_mae_per_channel": list(map(float, test_mae)),
        "test_mae_mean_kcal": float(test_mae.mean()),
        "elapsed_s": time.time() - t0,
        "hp": {
            "mode": MODE,
            "lr": LR_LOW if MODE != "baseline_only" else None,
            "epochs_max": EPOCHS_MAX if MODE != "baseline_only" else 0,
            "early_stop_patience": PATIENCE if MODE != "baseline_only" else 0,
            "batch_size": BATCH, "weight_decay": WD,
            "arch": "m3 ablation",
            "baseline": BASELINE,
            "ridge_alpha": RIDGE_ALPHA,
            "d_model": M1_HP["d_model"], "n_heads": M1_HP["n_heads"],
        },
        **result_extra,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  test MAE per channel: {[f'{v:.2f}' for v in test_mae]}", flush=True)
    print(f"  test MAE mean: {float(test_mae.mean()):.3f} kcal/mol", flush=True)
    print(f"  test barrier MAE: {barrier_mae:.3f}  RMSE: {barrier_rmse:.3f}", flush=True)
    print(f"  wrote {out_path}  ({result['elapsed_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=int(os.environ.get("FOLD", 0)))
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--all-members", action="store_true",
                    help="Loop members 0..4 in this process (for baseline_only CPU job).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.all_members:
        for m in range(5):
            args2 = argparse.Namespace(**vars(args))
            args2.member = m
            run_one(args2)
        return

    if args.member is None:
        m = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        args.member = m
        print(f"[array] SLURM_ARRAY_TASK_ID={m} -> member={m}", flush=True)
    run_one(args)


if __name__ == "__main__":
    main()
