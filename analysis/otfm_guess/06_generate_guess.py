#!/usr/bin/env python3
"""SPEC18 Step 6 — cross-fit TS generation with GUESS-mode models.

Fork of spec17rev2 Step 7 (`07_generate_crossfit.py`), differences:
  - Model checkpoints live under `ckpt/guess_fold{K}/`.
  - `test.pkl` in each fold's data dir already carries the `ts_guess_xtbpath`
    key (Step 4 verified). React-OT's DataLoader emits it in each batch's
    `conditions` dict; the model's `eval_sample_batch` consumes it as x1.

The rest of the pipeline (setup → test_dataloader → eval_sample_batch →
split by x0_size → write per-rxn xyz) is identical to Step 7.
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"
ROT = REPO / "external" / "react-ot"
GEN = BASE / "generated" / "guess"
MANIFEST = GEN.parent / "manifest.csv"
N_FOLDS = 5

for _d in ("artifacts", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / _d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PREV))
from _rot_patches import apply_all as apply_rot_patches  # noqa: E402

SAMPLER_NFE = int(os.environ.get("SAMPLER_NFE", "25"))
SAMPLER_MAX_NUM = int(os.environ.get("SAMPLER_MAX_NUM", "1600"))
BATCH_SIZE_TEST = int(os.environ.get("BATCH_SIZE_TEST", "8"))

Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
               15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def write_xyz(path: Path, syms, xyz: np.ndarray, comment: str = "") -> None:
    tmp = path.with_suffix(".xyz.tmp")
    n = len(syms)
    with open(tmp, "w") as f:
        f.write(f"{n}\n{comment}\n")
        for s, (x, y, z) in zip(syms, xyz):
            f.write(f"{s:<3}{x:16.8f}{y:16.8f}{z:16.8f}\n")
    tmp.replace(path)


def already_generated(rid: int, fold: int) -> bool:
    p = GEN / f"rxn_{rid:04d}.xyz"
    if not p.exists():
        return False
    try:
        head = p.read_text().splitlines()[1]
        m = re.search(r"fold=(\d+)", head)
        return m is not None and int(m.group(1)) == fold
    except Exception:
        return False


def append_manifest(rows) -> None:
    header = ["rxn_id", "generating_model", "n_atoms", "path"]
    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def find_checkpoint(fold_dir: Path) -> Path:
    """Pick best-val ckpt in fold_dir/checkpoint/**."""
    hits = list(fold_dir.rglob("sb-*.ckpt"))
    if not hits:
        raise FileNotFoundError(f"no .ckpt under {fold_dir}")

    def _key(p):
        m = re.search(r"val_ep_scaled_err=(-?\d+\.\d+)", p.name)
        return float(m.group(1)) if m else float("inf")

    return sorted(hits, key=_key)[0]


def load_model(ckpt_path: Path):
    """Load SBModule (or DDPMModule) — see spec17rev2 fix history for
    why we print all attempts' tracebacks."""
    import traceback
    sys.path.insert(0, str(ROT))
    import torch  # noqa: F401

    errors = []
    for mod_name, cls_name in (
        ("reactot.trainer.pl_trainer", "SBModule"),
        ("reactot.trainer.pl_trainer", "DDPMModule"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            m = cls.load_from_checkpoint(str(ckpt_path), map_location="cpu")
            print(f"[loaded] {cls_name} from {ckpt_path.name}")
            return m
        except Exception as e:  # noqa: BLE001
            errors.append((mod_name, cls_name, e, traceback.format_exc()))
    for mod_name, cls_name, e, tb in errors:
        print(f"\n=== attempt: {mod_name}.{cls_name} ===")
        print(tb, flush=True)
    raise RuntimeError(f"could not load {ckpt_path}")


def load_fold_test(fold: int) -> dict:
    p = BASE / "ckpt" / f"guess_fold{fold}" / "data" / f"test_fold{fold}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"missing test pkl: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    args = ap.parse_args()
    fold = args.fold

    GEN.mkdir(parents=True, exist_ok=True)
    fold_dir = BASE / "ckpt" / f"guess_fold{fold}"
    ckpt = find_checkpoint(fold_dir)
    apply_rot_patches(ROT)
    model = load_model(ckpt)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_data = str(fold_dir / "data") + "/"
    model.training_config["datadir"] = fold_data
    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    model.setup(stage="test", device=device, swapping_react_prod=False)
    model = model.to(device).eval()
    model.nfe = SAMPLER_NFE

    # SPEC17rev2 discovery: `EnSB.sample()` references self.opt; attach it.
    class _OPT:
        def __init__(self):
            self.solver = "ode"
            self.method = "euler"
            self.atol = 1e-2
            self.rtol = 1e-2

    model.ddpm.opt = _OPT()
    if not hasattr(model, "ot_ode"):
        model.ot_ode = True

    ds = load_fold_test(fold)
    folds = pd.read_csv(PREV / "artifacts" / "folds.csv").set_index("rxn_id")
    test_rids = list(ds["rxn_id"])
    for rid in test_rids:
        if folds.at[rid, "fold"] != fold:
            print(f"[FATAL] rxn {rid} not in fold {fold}", file=sys.stderr)
            return 1
    # Assert ts_guess_xtbpath is present in this fold's test data so GUESS-mode
    # sampling actually has an x1 anchor.
    assert "ts_guess_xtbpath" in ds, \
        f"test_fold{fold}.pkl missing ts_guess_xtbpath — rerun Step 4"

    from torch.utils.data import DataLoader
    loader = DataLoader(
        model.test_dataset,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=0,
        collate_fn=model.test_dataset.collate_fn,
    )

    new_rows = []
    n_skip = 0
    cursor = 0
    for batch_idx, batch in enumerate(loader):
        this_batch_rids = test_rids[cursor:cursor + BATCH_SIZE_TEST]
        if all(already_generated(rid, fold) for rid in this_batch_rids):
            n_skip += len(this_batch_rids)
            cursor += len(this_batch_rids)
            continue
        r_pos, x0_pred, p_pos, x0_size, x0_other, rmsds = model.eval_sample_batch(
            batch, return_all=True,
        )
        pos_np = x0_pred.detach().cpu().numpy()
        sizes = [int(s) for s in x0_size]
        assert sum(sizes) == pos_np.shape[0], \
            f"size mismatch: sizes={sizes} pos_len={pos_np.shape[0]}"
        offset = 0
        for k, n in enumerate(sizes):
            if k >= len(this_batch_rids):
                break
            rid = this_batch_rids[k]
            if already_generated(rid, fold):
                offset += n
                continue
            rxn_pos = pos_np[offset:offset + n]
            offset += n
            chg = ds["reactant"]["charges"][cursor + k]
            syms = [Z_TO_SYMBOL[z] for z in chg]
            if len(syms) != n:
                print(f"[WARN] rid {rid}: syms={len(syms)} vs x0_size={n}",
                      file=sys.stderr)
                continue
            out = GEN / f"rxn_{rid:04d}.xyz"
            write_xyz(out, syms, rxn_pos,
                      comment=f"rid={rid} fold={fold} nfe={SAMPLER_NFE} mode=GUESS")
            new_rows.append(dict(rxn_id=rid, generating_model=fold,
                                  n_atoms=n, path=str(out)))
        cursor += len(this_batch_rids)
        if (batch_idx + 1) % 20 == 0:
            print(f"  fold {fold}: batch {batch_idx+1}  cursor {cursor}/{len(test_rids)}",
                  flush=True)

    if new_rows:
        append_manifest(new_rows)
    print(f"fold {fold}: generated {len(new_rows)}  skipped {n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
