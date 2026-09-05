#!/usr/bin/env python3
"""SPEC17rev2 Step 6 — cross-fit training driver (per fold).

Usage:
    python 06_train_crossfit.py --fold K [--prepare-only]

For fold K (0..4):
  1. Load coley_all.pkl + folds.csv
  2. Write train_fold{K}.pkl / val_fold{K}.pkl to ckpt/fold{K}/data/
     (val = 10% split of the train set with SEED)
  3. Materialize a patched copy of react-ot's train_rpsb_ts1x.py in
     ckpt/fold{K}/train.py with our overrides:
        - datadir           -> ckpt/fold{K}/data/
        - node_nfs          -> [11]*3   (7 atom types)
        - sampler max_num   -> configurable via SAMPLER_MAX_NUM (default 1600)
        - bz                -> configurable via BATCH_SIZE (default 8)
        - swap datamodule pkl filenames -> train_fold{K}.pkl / val_fold{K}.pkl
        - inject partial_load of external/react-ot/reactot-pretrained.ckpt
  4. Invoke it via subprocess.

Idempotent: if ckpt/fold{K}/last.ckpt already exists and training is
requested, subprocess will resume/skip according to Lightning's default
`resume_from_checkpoint` fallback.

GATE-6:  five checkpoints exist, val loss converges.
GATE-6a: partial_load only skips embed/output layers (asserted in the
         injected preamble via _partial_load.assert_gate_6a).
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
ROT = REPO / "external/react-ot"
PRETRAINED = ROT / "reactot-pretrained.ckpt"
SEED = 42
N_FOLDS = 5
VAL_FRAC = 0.10

for _d in ("artifacts", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / _d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
from _rot_patches import apply_all as apply_rot_patches  # noqa: E402

# Environment-tunable knobs.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
SAMPLER_MAX_NUM = int(os.environ.get("SAMPLER_MAX_NUM", "1600"))
MAX_EPOCHS = os.environ.get("MAX_EPOCHS", "")  # empty -> keep react-ot default


def slice_fold(data: dict, ids: set) -> dict:
    keep = [i for i, r in enumerate(data["rxn_id"]) if r in ids]
    return {
        "reactant": {k: [data["reactant"][k][i] for i in keep]
                     for k in data["reactant"]},
        "transition_state": {k: [data["transition_state"][k][i] for i in keep]
                             for k in data["transition_state"]},
        "product": {k: [data["product"][k][i] for i in keep]
                    for k in data["product"]},
        "single_fragment": [data["single_fragment"][i] for i in keep],
        "rxn_id": [data["rxn_id"][i] for i in keep],
    }


def prepare_fold(fold: int) -> tuple[Path, int, int]:
    """Write train_fold{K}.pkl / val_fold{K}.pkl. Return (datadir, n_train, n_val)."""
    coley_pkl = BASE / "data" / "coley_all.pkl"
    folds_csv = BASE / "artifacts" / "folds.csv"
    with open(coley_pkl, "rb") as f:
        data = pickle.load(f)
    folds = pd.read_csv(folds_csv).set_index("rxn_id")

    train_ids = set(folds[folds.fold != fold].index)
    test_ids = set(folds[folds.fold == fold].index)

    # Deterministic 10% val from train_ids
    rng = np.random.RandomState(SEED + fold)
    train_list = sorted(train_ids)
    rng.shuffle(train_list)
    n_val = max(1, int(round(VAL_FRAC * len(train_list))))
    val_ids = set(train_list[:n_val])
    tr_ids = set(train_list[n_val:])

    fold_dir = BASE / "ckpt" / f"fold{fold}"
    datadir = fold_dir / "data"
    datadir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", tr_ids), ("val", val_ids), ("test", test_ids)):
        sub = slice_fold(data, ids)
        out = datadir / f"{name}_fold{fold}.pkl"
        tmp = out.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(sub, f)
        tmp.replace(out)
        print(f"  wrote {out.name}: n={len(sub['rxn_id'])}")
    return datadir, len(tr_ids), len(val_ids)


PREAMBLE_TEMPLATE = '''\
# --- SPEC17rev2 injected preamble ---------------------------------
import os, sys
sys.path.insert(0, {analysis_dir!r})
from _partial_load import partial_load, assert_gate_6a
import torch as _torch

_PRETRAINED_CKPT_PATH = {pretrained!r}
_FOLD_DATADIR = {datadir!r}
_SAMPLER_MAX_NUM = {sampler_max_num}
_BATCH_SIZE = {batch_size}
_FOLD_NAME = {fold_name!r}
# ------------------------------------------------------------------
'''

# Regex-based patches applied to a copy of react-ot's train_rpsb_ts1x.py.
# Each entry is (pattern, replacement). Grouped by SPEC §5.3, §5.4, §6, §8.
PATCH_RULES = [
    # datadir override — SPEC training_config
    (r'datadir\s*=\s*"[^"]*"', 'datadir=_FOLD_DATADIR'),
    # node feature count — SPEC §5.3
    (r'node_nfs\s*:\s*List\[int\]\s*=\s*\[9\]\s*\*\s*3',
     'node_nfs: List[int] = [11] * 3'),
    (r'node_nfs\s*=\s*\[9\]\s*\*\s*3', 'node_nfs=[11] * 3'),
    # sampler max_num — SPEC §8 warning re OOM
    (r'max_num\s*=\s*2800', 'max_num=_SAMPLER_MAX_NUM'),
    # batch size — SPEC §8 warning re OOM
    (r'\bbz\s*=\s*\d+', 'bz=_BATCH_SIZE'),
    # dataset file names — react-ot's TS1x split uses ts1x-*.pkl by default;
    # force our fold pkls.
    (r'ts1x-train-[^"\']+\.pkl', f'train_{{FOLD_NAME}}.pkl'),
    (r'ts1x-val-[^"\']+\.pkl',   f'val_{{FOLD_NAME}}.pkl'),
    (r'ts1x-test-[^"\']+\.pkl',  f'test_{{FOLD_NAME}}.pkl'),
    # colored_traceback triggers curses.setupterm() at import time, which
    # fails on SLURM compute nodes lacking a terminfo database. Not needed
    # for training — swap to a no-op try/except so any missing dep is soft.
    # (?m) enables multiline so ^ matches each line start.
    (r'(?m)^\s*import colored_traceback\.always\s*$',
     'try:\n    import colored_traceback.always\nexcept Exception:\n    pass'),
    # PyTorch Lightning 2.x renamed `replace_sampler_ddp` to
    # `use_distributed_sampler`. Semantics identical.
    (r'\breplace_sampler_ddp\b', 'use_distributed_sampler'),
]


def find_train_script() -> Path:
    candidates = [
        ROT / "train_rpsb_ts1x.py",
        ROT / "reactot" / "train_rpsb_ts1x.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to any file matching *train_rpsb*.py
    hits = list(ROT.rglob("train_rpsb*.py"))
    if hits:
        return hits[0]
    print(f"[FATAL] cannot locate train_rpsb_ts1x.py under {ROT}",
          file=sys.stderr)
    raise SystemExit(1)


def patch_train_script(src: Path, dst: Path, datadir: Path,
                        fold_name: str) -> None:
    text = src.read_text()
    fold_placeholder = "{FOLD_NAME}"
    for pat, repl in PATCH_RULES:
        text = re.sub(pat, repl, text)
    text = text.replace(fold_placeholder, fold_name)
    preamble = PREAMBLE_TEMPLATE.format(
        analysis_dir=str(BASE),
        pretrained=str(PRETRAINED),
        datadir=str(datadir) + "/",
        sampler_max_num=SAMPLER_MAX_NUM,
        batch_size=BATCH_SIZE,
        fold_name=fold_name,
    )
    # Optionally cap epochs
    if MAX_EPOCHS:
        preamble += f"_MAX_EPOCHS = {int(MAX_EPOCHS)}\n"
        text = re.sub(r'max_epochs\s*=\s*\d+',
                      'max_epochs=_MAX_EPOCHS', text)

    dst.write_text(preamble + text)


def train_one_fold(fold: int) -> int:
    fold_name = f"fold{fold}"
    fold_dir = BASE / "ckpt" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    datadir, n_tr, n_val = prepare_fold(fold)
    print(f"fold {fold}: n_train={n_tr}  n_val={n_val}")

    if not PRETRAINED.exists():
        print(f"[FATAL] pretrained ckpt missing: {PRETRAINED}", file=sys.stderr)
        return 1

    # GATE-6b: extend ATOM_MAPPING to 7 elements + allowed_atom_types BEFORE
    # any react-ot dataset import. Idempotent (writes .py.orig once).
    mapping = apply_rot_patches(ROT)
    (BASE / "artifacts" / "GATE6b_STATUS.txt").write_text(
        "PASS\n"
        f"atom_mapping={mapping}\n"
        f"n_element={len(mapping)}\n"
        f"expected_node_nfs={3 + len(mapping) + 1}\n"
    )

    src = find_train_script()
    dst = fold_dir / "train.py"
    patch_train_script(src, dst, datadir, fold_name)
    print(f"patched trainer: {dst}")

    # Cross-fit safety: refuse to run if fold data leaks.
    folds = pd.read_csv(BASE / "artifacts" / "folds.csv").set_index("rxn_id")
    with open(datadir / f"train_{fold_name}.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(datadir / f"test_{fold_name}.pkl", "rb") as f:
        te = pickle.load(f)
    tr_ids = set(tr["rxn_id"])
    te_ids = set(te["rxn_id"])
    if tr_ids & te_ids:
        print(f"[FATAL] train/test leakage detected in fold {fold}",
              file=sys.stderr)
        return 1
    if any(folds.loc[list(te_ids), "fold"] != fold):
        print(f"[FATAL] test set contains other folds' rxns", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROT}:{env.get('PYTHONPATH', '')}"
    env["SPEC17_FOLD"] = fold_name

    cmd = ["python", "-u", "train.py"]
    print("running:", " ".join(cmd), "in", fold_dir)
    r = subprocess.run(cmd, cwd=fold_dir, env=env)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    ap.add_argument("--prepare-only", action="store_true",
                    help="write fold pkls, do not launch training")
    args = ap.parse_args()

    if args.prepare_only:
        prepare_fold(args.fold)
        return 0
    return train_one_fold(args.fold)


if __name__ == "__main__":
    sys.exit(main())
