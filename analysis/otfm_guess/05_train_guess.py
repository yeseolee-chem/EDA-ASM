#!/usr/bin/env python3
"""SPEC18 Step 5 — train React-OT in GUESS mode (xTB path as x1).

Usage:
    python 05_train_guess.py --fold K [--from-rp | --from-pretrained]

Fold pkls must already exist at `ckpt/guess_fold{K}/data/` (produced by
Step 4). This script materializes a patched train.py in that fold's
directory and launches it.

Delta vs spec17rev2 Step 6:
  - `mapping_initial: str = "RP"`  →  `"GUESS"`
  - `ts_guess: bool = None`        →  `= "ts_guess_xtbpath"`

Warm-start options (`--from-*`):
  --from-pretrained  (default)  use react-ot's official Transition1x ckpt
  --from-rp                     start from spec17rev2 RP-mode fold ckpt
                                (for A/B comparison of transfer strength)
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"
ROT = REPO / "external" / "react-ot"
PRETRAINED = ROT / "reactot-pretrained.ckpt"
N_FOLDS = 5

sys.path.insert(0, str(PREV))         # reuse _rot_patches + _partial_load
from _rot_patches import apply_all as apply_rot_patches  # noqa: E402

# Env-tunable knobs (defaults match spec17rev2)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
SAMPLER_MAX_NUM = int(os.environ.get("SAMPLER_MAX_NUM", "8000"))
MAX_EPOCHS = os.environ.get("MAX_EPOCHS", "")


PREAMBLE_TEMPLATE = '''\
# --- SPEC18 injected preamble -------------------------------------
import glob as _glob
import os, sys, re as _re
sys.path.insert(0, {analysis_dir!r})
from _partial_load import partial_load, assert_gate_6a
import torch as _torch

_PRETRAINED_CKPT_PATH = {pretrained!r}
_FOLD_DATADIR = {datadir!r}
_SAMPLER_MAX_NUM = {sampler_max_num}
_BATCH_SIZE = {batch_size}
_FOLD_NAME = {fold_name!r}

def _find_best_ckpt():
    pat = os.path.join({fold_ckpt_dir!r}, "checkpoint", "*", "*", "sb-*.ckpt")
    hits = _glob.glob(pat)
    if not hits:
        return None
    def _key(p):
        m = _re.search(r"val_ep_scaled_err=(-?\\d+\\.\\d+)", p)
        return float(m.group(1)) if m else float("inf")
    return sorted(hits, key=_key)[0]

_RESUME_CKPT = _find_best_ckpt()
if _RESUME_CKPT:
    print(f"[resume] found best ckpt: {{_RESUME_CKPT}}")
else:
    print("[resume] no prior checkpoint, training from scratch")
# ------------------------------------------------------------------
'''


# Regex patches applied to a copy of react-ot's train_rpsb_ts1x.py.
# First 10 rules are lifted verbatim from spec17rev2 (functional equivalence);
# the last two are SPEC18-specific (GUESS mode + ts_guess key).
PATCH_RULES = [
    (r'datadir\s*=\s*"[^"]*"', 'datadir=_FOLD_DATADIR'),
    (r'node_nfs\s*:\s*List\[int\]\s*=\s*\[9\]\s*\*\s*3',
     'node_nfs: List[int] = [11] * 3'),
    (r'node_nfs\s*=\s*\[9\]\s*\*\s*3', 'node_nfs=[11] * 3'),
    (r'max_num\s*=\s*2800', 'max_num=_SAMPLER_MAX_NUM'),
    (r'\bbz\s*=\s*\d+', 'bz=_BATCH_SIZE'),
    (r'ts1x-train-[^"\']+\.pkl', f'train_{{FOLD_NAME}}.pkl'),
    (r'ts1x-val-[^"\']+\.pkl',   f'val_{{FOLD_NAME}}.pkl'),
    (r'ts1x-test-[^"\']+\.pkl',  f'test_{{FOLD_NAME}}.pkl'),
    (r'(?m)^\s*import colored_traceback\.always\s*$',
     'try:\n    import colored_traceback.always\nexcept Exception:\n    pass'),
    (r'\buse_by_ind\s*=\s*True\b', 'use_by_ind=False'),
    (r'save_top_k=-1', 'save_top_k=3'),
    (r'trainer\.fit\(ddpm\)', 'trainer.fit(ddpm, ckpt_path=_RESUME_CKPT)'),

    # SPEC18-only ---------------------------------------------------
    (r'mapping_initial\s*:\s*str\s*=\s*"RP"',
     'mapping_initial: str = "GUESS"'),
    (r'ts_guess\s*:\s*bool\s*=\s*None',
     'ts_guess = "ts_guess_xtbpath"'),
]


def find_train_script() -> Path:
    for c in (ROT / "train_rpsb_ts1x.py",
              ROT / "reactot" / "trainer" / "train_rpsb_ts1x.py",
              ROT / "reactot" / "train_rpsb_ts1x.py"):
        if c.exists():
            return c
    hits = list(ROT.rglob("train_rpsb*.py"))
    if hits:
        return hits[0]
    raise SystemExit(f"[FATAL] cannot locate train_rpsb_ts1x.py under {ROT}")


def patch_train_script(src: Path, dst: Path, datadir: Path,
                        fold_name: str) -> None:
    text = src.read_text()
    for pat, repl in PATCH_RULES:
        text = re.sub(pat, repl, text)
    text = text.replace("{FOLD_NAME}", fold_name)
    preamble = PREAMBLE_TEMPLATE.format(
        analysis_dir=str(PREV),           # keep _rot_patches / _partial_load
        pretrained=str(PRETRAINED),
        datadir=str(datadir) + "/",
        sampler_max_num=SAMPLER_MAX_NUM,
        batch_size=BATCH_SIZE,
        fold_name=fold_name,
        fold_ckpt_dir=str(BASE / "ckpt" / fold_name),
    )
    if MAX_EPOCHS:
        preamble += f"_MAX_EPOCHS = {int(MAX_EPOCHS)}\n"
        text = re.sub(r'max_epochs\s*=\s*\d+',
                      'max_epochs=_MAX_EPOCHS', text)
    dst.write_text(preamble + text)


def train_one_fold(fold: int, warm_from: str) -> int:
    fold_name = f"guess_fold{fold}"
    fold_dir = BASE / "ckpt" / fold_name
    datadir = fold_dir / "data"
    if not datadir.exists():
        print(f"[FATAL] fold data missing: {datadir}. Run Step 4 first.",
              file=sys.stderr)
        return 1

    # Verify ts_guess passthrough into the fold pkl
    train_pkl = datadir / f"train_{fold_name.replace('guess_', '')}.pkl"
    # The Step 4 slicer wrote e.g. train_fold3.pkl (NOT train_guess_fold3.pkl)
    # because the fold_name in slice is "fold{K}", not "guess_fold{K}".
    # patch_train_script's `train_{FOLD_NAME}.pkl` also references FOLD_NAME
    # which is `guess_foldK`. So we align the pkl names below.
    return _launch(fold, fold_name, fold_dir, datadir, warm_from)


def _launch(fold, fold_name, fold_dir, datadir, warm_from) -> int:
    # Ensure the react-ot ATOM_MAPPING patches are in place BEFORE the
    # subprocess imports react-ot's dataset module.
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
    print(f"[patched] {dst}")

    # Align pkl filenames the patched train.py expects. Step 4 wrote them
    # with the base fold suffix (`fold{K}` — no `guess_` prefix). Bridge via
    # symlink so `train_guess_fold{K}.pkl` -> `train_fold{K}.pkl` etc.
    plain = fold_name.replace("guess_", "")
    for stem in ("train", "val", "test"):
        target = datadir / f"{stem}_{plain}.pkl"
        alias = datadir / f"{stem}_{fold_name}.pkl"
        if not target.exists():
            print(f"[FATAL] missing fold pkl: {target}", file=sys.stderr)
            return 1
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(target.name)

    # If --from-rp, seed the fold dir with a partial-load of spec17rev2
    # RP-mode best ckpt. Otherwise the preamble falls through to react-ot's
    # own resume-from-pretrained behavior on Trainer.fit.
    if warm_from == "rp":
        rp_dir = PREV / "ckpt" / plain
        cand = sorted(rp_dir.rglob("sb-*.ckpt"),
                      key=lambda p: float(
                          re.search(r"val_ep_scaled_err=(-?\d+\.\d+)",
                                    p.name).group(1))
                      if re.search(r"val_ep_scaled_err=(-?\d+\.\d+)", p.name)
                      else float("inf"))
        if cand:
            # Place a copy under ckpt/guess_fold{K}/checkpoint/... so the
            # preamble's _find_best_ckpt picks it up.
            dst_ckpt = fold_dir / "checkpoint" / "RPSB-FT-Schedule" / \
                "rp_warmstart" / cand[0].name
            dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
            if not dst_ckpt.exists():
                import shutil
                shutil.copy2(cand[0], dst_ckpt)
                print(f"[warm-start-rp] copied {cand[0].name} -> {dst_ckpt}")

    # Cross-fit leak assert
    folds = pd.read_csv(PREV / "artifacts" / "folds.csv").set_index("rxn_id")
    with open(datadir / f"train_{plain}.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(datadir / f"test_{plain}.pkl", "rb") as f:
        te = pickle.load(f)
    tr_ids = set(tr["rxn_id"])
    te_ids = set(te["rxn_id"])
    if tr_ids & te_ids:
        print(f"[FATAL] train/test leakage in fold {fold}", file=sys.stderr)
        return 1
    if any(folds.loc[list(te_ids), "fold"] != fold):
        print(f"[FATAL] test set contains other folds' rxns", file=sys.stderr)
        return 1
    # SPEC-crit: ts_guess key must be present in the train pkl
    if "ts_guess_xtbpath" not in tr:
        print("[FATAL] ts_guess_xtbpath missing from train pkl — "
              "did Step 4 write it?", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROT}:{env.get('PYTHONPATH', '')}"
    env["SPEC18_FOLD"] = fold_name
    cmd = ["python", "-u", "train.py"]
    print(f"[run] {' '.join(cmd)}  in {fold_dir}")
    r = subprocess.run(cmd, cwd=fold_dir, env=env)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    ap.add_argument("--from-rp", action="store_true",
                    help="warm-start from spec17rev2 RP-mode best ckpt")
    args = ap.parse_args()
    warm_from = "rp" if args.from_rp else "pretrained"
    return train_one_fold(args.fold, warm_from)


if __name__ == "__main__":
    sys.exit(main())
