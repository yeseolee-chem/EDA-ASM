"""SPEC_02 — retrain M_δ (b ≡ 0). Same arch/HP/split/seeds as m3 M_bδ.

Writes results/spec02_bdelta/m_delta/fold{F}/member{M}.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))

from eda_asm.asr_v1.data import ASR_COMPONENTS
from eda_asm.asr_v1.models_delta import ModelM1Delta
from eda_asm.asr_v1.training_delta import (CachedFeatureBundleDelta, TrainConfigDelta,
                                            train_one_model_delta)

BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt")
SPLITS = REPO / "pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood"
OUT = REPO / "results" / "spec02_bdelta" / "m_delta"

SEED_BASE = 42; SIZE_FULL = 509; BATCH = 16; WD = 1e-3; LR = 1e-5
EPOCHS_MAX = int(os.environ.get("EPOCHS_MAX", 100_000))
PATIENCE = int(os.environ.get("PATIENCE", 10_000))
M1_HP = dict(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)


def load_indices(fold):
    fdir = SPLITS / f"fold{fold}"
    return json.load(open(fdir / f"size_{SIZE_FULL}.json")), json.load(open(fdir / "test_rids.json"))


def make_val(train_pos, fold, member):
    rng = np.random.default_rng(SEED_BASE + fold * 100 + member * 17)
    arr = list(train_pos); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * 0.15))
    return arr[n_val:], arr[:n_val]


class _ZeroBaseline:
    def fit(self, D, Y): return self
    def predict(self, D): return np.zeros((D.shape[0], 5), dtype=np.float32)
    def state_dict(self): return {}


def evaluate(model, bundle, test_pos, baseline_pred_np, device):
    model.eval()
    y_true = bundle.labels[test_pos].numpy(); preds = []
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
    return y_true, np.array(preds)


def run_one(fold, member):
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT / f"fold{fold}"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"member{member}.json"
    if out_path.exists(): print(f"[skip] {out_path}"); return
    seed = SEED_BASE + fold * 100 + member
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== M_δ fold={fold} member={member} lr={LR} epochs={EPOCHS_MAX} pat={PATIENCE} ===")
    bundle = CachedFeatureBundleDelta.load(BUNDLE)
    print(f"  bundle: {len(bundle.reaction_ids)} rxns")
    train_rids, test_rids = load_indices(fold)
    train_pos_all = [i for i, r in enumerate(bundle.reaction_ids) if r in set(train_rids)]
    test_pos = [i for i, r in enumerate(bundle.reaction_ids) if r in set(test_rids)]
    train_pos, val_pos = make_val(train_pos_all, fold, member)
    print(f"  N: tr={len(train_pos)} va={len(val_pos)} te={len(test_pos)}")

    cfg = TrainConfigDelta(epochs=EPOCHS_MAX, batch_size=BATCH, lr=LR, weight_decay=WD,
                           early_stop_patience=PATIENCE, device=device, baseline_ridge_alpha=1.0)
    def factory(fd): return ModelM1Delta(feature_dim=fd, **M1_HP)

    import eda_asm.asr_v1.training_delta as td
    _orig = td.LinearBaseline
    td.LinearBaseline = lambda **kw: _ZeroBaseline()
    try:
        t0 = time.time()
        model, fr = train_one_model_delta(bundle, factory, train_pos, val_pos, cfg, seed=seed)
    finally:
        td.LinearBaseline = _orig

    baseline_all = np.zeros((len(bundle.reaction_ids), 5), dtype=np.float32)
    y_true, y_pred = evaluate(model, bundle, test_pos, baseline_all, device)
    test_mae = np.abs(y_pred - y_true).mean(axis=0)
    bt = y_true.sum(axis=1); bp = y_pred.sum(axis=1)
    out = {"variant": "M_delta_only", "fold": fold, "member": member, "seed": seed,
           "n_train": len(train_pos), "n_val": len(val_pos), "n_test": len(test_pos),
           "reaction_ids": [bundle.reaction_ids[i] for i in test_pos],
           "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
           "components": list(ASR_COMPONENTS),
           "barrier_true": bt.tolist(), "barrier_pred": bp.tolist(),
           "test_mae_per_channel": list(map(float, test_mae)),
           "test_mae_mean_kcal": float(test_mae.mean()),
           "test_barrier_mae": float(np.abs(bp - bt).mean()),
           "test_barrier_rmse": float(np.sqrt(np.mean((bp - bt) ** 2))),
           "best_epoch": int(fr.best_epoch), "final_epoch": int(fr.final_epoch),
           "early_stopped": bool(fr.early_stopped), "elapsed_s": time.time() - t0,
           "hp": {"lr": LR, "epochs_max": EPOCHS_MAX, "patience": PATIENCE,
                  "batch_size": BATCH, "weight_decay": WD, "baseline": "ZERO"}}
    out_path.write_text(json.dumps(out, indent=2))
    # Save trained model state_dict for reproducibility (~1 MB per checkpoint).
    ckpt_path = out_dir / f"member{member}.ckpt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": seed, "fold": fold, "member": member,
        "hp": out["hp"], "variant": "M_delta_only",
    }, ckpt_path)
    print(f"  wrote {out_path} + {ckpt_path.name}  {out['elapsed_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--member", type=int, default=None)
    args = ap.parse_args()
    if args.fold is None:
        args.fold = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if args.member is None:
        for m in range(5): run_one(args.fold, m)
        return
    run_one(args.fold, args.member)


if __name__ == "__main__":
    main()
