#!/usr/bin/env python3
"""SPEC17rev2 Step 8 — final deployment model trained on all 5269 rxns.

Reuses Step 6's patching machinery with `--fold final` semantics:
  train = all rxns except a small held-out val (SEED=42)
  no test set (the 960 downstream reactions are the "real" held-out)

Output: ckpt/final/last.ckpt
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
ROT = REPO / "external/react-ot"
SEED = 42
VAL_FRAC = 0.05

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
SAMPLER_MAX_NUM = int(os.environ.get("SAMPLER_MAX_NUM", "1600"))
MAX_EPOCHS = os.environ.get("MAX_EPOCHS", "")

for _d in ("artifacts", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / _d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
from _rot_patches import apply_all as apply_rot_patches  # noqa: E402


def slice_data(data: dict, ids: set) -> dict:
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


def main() -> int:
    coley_pkl = BASE / "data" / "coley_all.pkl"
    with open(coley_pkl, "rb") as f:
        data = pickle.load(f)

    all_ids = list(data["rxn_id"])
    rng = np.random.RandomState(SEED)
    shuf = list(all_ids)
    rng.shuffle(shuf)
    n_val = max(1, int(round(VAL_FRAC * len(shuf))))
    val_ids = set(shuf[:n_val])
    train_ids = set(shuf[n_val:])
    print(f"final: n_train={len(train_ids)}  n_val={len(val_ids)}")

    final_dir = BASE / "ckpt" / "final"
    datadir = final_dir / "data"
    datadir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids)):
        sub = slice_data(data, ids)
        out = datadir / f"{name}_final.pkl"
        tmp = out.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(sub, f)
        tmp.replace(out)
        print(f"  wrote {out.name}: n={len(sub['rxn_id'])}")
    # empty test set — some react-ot loaders require the key
    tmp = datadir / "test_final.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(slice_data(data, set()), f)
    tmp.replace(datadir / "test_final.pkl")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROT}:{env.get('PYTHONPATH', '')}"

    # GATE-6b: patch ATOM_MAPPING before spawning the trainer.
    apply_rot_patches(ROT)

    import re

    src_candidates = [
        ROT / "train_rpsb_ts1x.py",
        ROT / "reactot" / "train_rpsb_ts1x.py",
    ]
    src = next((p for p in src_candidates if p.exists()), None)
    if src is None:
        hits = list(ROT.rglob("train_rpsb*.py"))
        src = hits[0] if hits else None
    if src is None:
        print("[FATAL] train_rpsb_ts1x.py not found", file=sys.stderr)
        return 1

    text = src.read_text()
    for pat, repl in [
        (r'datadir\s*=\s*"[^"]*"', 'datadir=_FOLD_DATADIR'),
        (r'node_nfs\s*:\s*List\[int\]\s*=\s*\[9\]\s*\*\s*3',
         'node_nfs: List[int] = [11] * 3'),
        (r'node_nfs\s*=\s*\[9\]\s*\*\s*3', 'node_nfs=[11] * 3'),
        (r'max_num\s*=\s*2800', 'max_num=_SAMPLER_MAX_NUM'),
        (r'\bbz\s*=\s*\d+', 'bz=_BATCH_SIZE'),
        (r'ts1x-train-[^"\']+\.pkl', 'train_final.pkl'),
        (r'ts1x-val-[^"\']+\.pkl',   'val_final.pkl'),
        (r'ts1x-test-[^"\']+\.pkl',  'test_final.pkl'),
    ]:
        text = re.sub(pat, repl, text)
    if MAX_EPOCHS:
        text = re.sub(r'max_epochs\s*=\s*\d+',
                      f'max_epochs={int(MAX_EPOCHS)}', text)

    preamble = (
        f"import sys\n"
        f"sys.path.insert(0, {str(BASE)!r})\n"
        f"from _partial_load import partial_load, assert_gate_6a\n"
        f"_PRETRAINED_CKPT_PATH = {str(ROT / 'reactot-pretrained.ckpt')!r}\n"
        f"_FOLD_DATADIR = {str(datadir) + '/'!r}\n"
        f"_SAMPLER_MAX_NUM = {SAMPLER_MAX_NUM}\n"
        f"_BATCH_SIZE = {BATCH_SIZE}\n"
        f"_FOLD_NAME = 'final'\n"
    )
    dst = final_dir / "train.py"
    dst.write_text(preamble + text)
    print(f"patched trainer: {dst}")

    cmd = ["python", "-u", "train.py"]
    print("running:", " ".join(cmd), "in", final_dir)
    r = subprocess.run(cmd, cwd=final_dir, env=env)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
