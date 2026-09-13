#!/usr/bin/env python3
"""SPEC18 Step 4 — build coley_all_guess.pkl and slice into per-fold pkls.

Attaches two new keys to the Coley dataset:
  ts_guess_xtbpath          — [n_atoms, 3] list per rxn (xTB TS or (R+P)/2 fallback)
  ts_guess_xtbpath_source   — "xtb" | "rp_fallback"

Then reproduces spec17rev2's per-fold slicing (reactants-only scaffold
split from `PREV/artifacts/folds.csv`) into `ckpt/guess_fold{K}/data/`
so Step 5 training can point at those pkls without re-splitting.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"

GUESS_KEY = "ts_guess_xtbpath"
SRC_KEY = GUESS_KEY + "_source"
SEED = 42
VAL_FRAC = 0.10


def slice_fold(data: dict, ids: set) -> dict:
    """Same schema as otfm_train Step 6 slice_fold, PLUS ts_guess passthrough."""
    keep = [i for i, r in enumerate(data["rxn_id"]) if r in ids]

    def slice_frag(frag: dict) -> dict:
        sub = {k: [frag[k][i] for i in keep] for k in frag}
        sub["num_atoms"] = [len(c) for c in sub["charges"]]
        return sub

    out = {
        "reactant":         slice_frag(data["reactant"]),
        "transition_state": slice_frag(data["transition_state"]),
        "product":          slice_frag(data["product"]),
        "single_fragment": [data["single_fragment"][i] for i in keep],
        "rxn_id":          [data["rxn_id"][i] for i in keep],
    }
    # Pass through per-rxn top-level keys (ts_guess + its source metadata).
    for k in (GUESS_KEY, SRC_KEY):
        if k in data:
            out[k] = [data[k][i] for i in keep]
    return out


def main() -> int:
    with open(PREV / "data" / "coley_all.pkl", "rb") as f:
        data = pickle.load(f)
    rxn_ids = data["rxn_id"]

    full_csv = BASE / "artifacts" / "xtb_path_full.csv"
    if not full_csv.exists():
        print(f"missing {full_csv} — Step 2 (mode=full) must complete first",
              file=sys.stderr)
        return 1
    res = pd.read_csv(full_csv).set_index("rxn_id")

    guess = []
    source = []
    for i, rid in enumerate(rxn_ids):
        npy = BASE / "xtb_runs" / f"rxn_{rid:04d}" / "ts_xtb.npy"
        R = np.array(data["reactant"]["positions"][i])
        P = np.array(data["product"]["positions"][i])
        if rid in res.index and res.at[rid, "status"] == "ok" and npy.exists():
            g = np.load(npy)
            if g.shape != R.shape:
                print(f"rxn {rid}: shape {g.shape} vs R {R.shape} — falling back",
                      file=sys.stderr)
                guess.append(((R + P) / 2).tolist()); source.append("rp_fallback")
                continue
            guess.append(g.tolist()); source.append("xtb")
        else:
            guess.append(((R + P) / 2).tolist()); source.append("rp_fallback")

    data[GUESS_KEY] = guess
    data[SRC_KEY] = source
    n_xtb = source.count("xtb")
    n_fb = source.count("rp_fallback")
    print(f"ts_guess: xTB {n_xtb}, (R+P)/2 fallback {n_fb} "
          f"({n_fb/max(1,len(source)):.1%})")

    out_pkl = BASE / "data" / "coley_all_guess.pkl"
    tmp = out_pkl.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(out_pkl)

    # Verify by round-tripping
    with open(out_pkl, "rb") as f:
        chk = pickle.load(f)
    assert GUESS_KEY in chk and SRC_KEY in chk
    assert len(chk[GUESS_KEY]) == len(chk["rxn_id"])
    for i in range(0, len(rxn_ids), 500):
        assert len(chk[GUESS_KEY][i]) == len(chk["transition_state"]["positions"][i]), \
            f"rxn {rxn_ids[i]}: ts_guess len mismatch"
    print(f"coley_all_guess.pkl OK  ({out_pkl.stat().st_size/1e6:.1f} MB)")

    # Per-fold split
    folds = pd.read_csv(PREV / "artifacts" / "folds.csv").set_index("rxn_id")
    for fold in range(5):
        train_ids = set(folds[folds.fold != fold].index)
        test_ids = set(folds[folds.fold == fold].index)
        rng = np.random.RandomState(SEED + fold)
        train_list = sorted(train_ids)
        rng.shuffle(train_list)
        n_val = max(1, int(round(VAL_FRAC * len(train_list))))
        val_ids = set(train_list[:n_val])
        tr_ids = set(train_list[n_val:])

        datadir = BASE / "ckpt" / f"guess_fold{fold}" / "data"
        datadir.mkdir(parents=True, exist_ok=True)
        for name, ids in (("train", tr_ids), ("val", val_ids), ("test", test_ids)):
            sub = slice_fold(chk, ids)
            out = datadir / f"{name}_fold{fold}.pkl"
            tmp = out.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(sub, f)
            tmp.replace(out)
            # SPEC-crit: verify ts_guess passed through slice
            assert GUESS_KEY in sub, f"{name}_fold{fold}: {GUESS_KEY} missing"
            print(f"  fold{fold} {name}: n={len(sub['rxn_id'])}  "
                  f"ts_guess_present={GUESS_KEY in sub}")

        # react-ot expected filename symlinks (see otfm_train/06 line 117-124)
        for our, rot in (("train", "train_rpsb_all"),
                         ("val",   "valid_rpsb_all"),
                         ("test",  "test")):
            src_name = f"{our}_fold{fold}.pkl"
            link = datadir / f"{rot}.pkl"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src_name)

    (BASE / "artifacts" / "GATE4_STATUS.txt").write_text(
        "PASS\n"
        f"n_rxns={len(rxn_ids)}\n"
        f"n_xtb={n_xtb}\n"
        f"n_rp_fallback={n_fb}\n"
        f"fallback_rate={n_fb/max(1,len(rxn_ids)):.4f}\n"
    )
    print("=== GATE-4 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
