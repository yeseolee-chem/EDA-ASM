"""Reload TS positions from raw TS.xyz for all dipolar reactions and overwrite
the (possibly-corrupted) TS field in .pt files.

Some dipolar .pt files had TS accidentally stored as raw r0+r1 concat coordinates
(overlapping atoms at origin) instead of the actual TS geometry. This restores
the correct TS positions.

Also verifies min pairwise distance is normal (>= 0.7 Å) after the fix.
"""
from __future__ import annotations
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
COHORT_V7 = REPO / "outputs/frag_review/cohort_v7.parquet"


def load_raw_ts(rid: str):
    idx = int(rid.split("_")[-1])
    d = RAW / str(idx)
    cand = [f for f in d.glob("TS_*.xyz") if "imag_mode" in f.name]
    if not cand:
        cand = list(d.glob("TS_*.xyz"))
    if not cand:
        return None
    return ase.io.read(str(cand[0]))


def main():
    c7 = pd.read_parquet(COHORT_V7)
    dip = c7[c7.family == "dipolar"].reaction_id.tolist()
    print(f"dipolar in cohort_v7: {len(dip)}", flush=True)

    n_ok = n_replaced = n_err = n_skip_ok = 0
    for rid in dip:
        pt = FEAT_DIR / f"{rid}.pt"
        if not pt.exists():
            n_err += 1; print(f"[ERR] {rid}: no .pt"); continue
        try:
            d = torch.load(str(pt), map_location="cpu", weights_only=False)
            if "TS" not in d:
                n_err += 1; print(f"[ERR] {rid}: no TS field"); continue
            pos_pt = np.asarray(d["TS"]["pos"])
            min_d = float(pdist(pos_pt).min()) if len(pos_pt) >= 2 else 999
            # Also reload from raw if positions look suspicious even at 0.9 Å.
            # Real minimum atomic distance in organic molecules is ~1.0 Å.
            if min_d >= 0.90:
                n_skip_ok += 1
                continue
            # Corrupted — reload from raw
            raw = load_raw_ts(rid)
            if raw is None:
                n_err += 1; print(f"[ERR] {rid}: no raw TS.xyz"); continue
            z_raw = np.asarray(raw.get_atomic_numbers(), int)
            pos_raw = raw.get_positions()
            # Sanity: match element counts (order can differ)
            from collections import Counter
            if Counter(int(z) for z in z_raw) != Counter(int(z) for z in np.asarray(d["TS"]["z"], int)):
                n_err += 1
                print(f"[ERR] {rid}: element count mismatch pt vs raw")
                continue
            d["TS"] = {"z": z_raw, "pos": pos_raw.astype(np.float32)}
            torch.save(d, str(pt))
            n_replaced += 1
            print(f"[FIX] {rid}: min_pdist {min_d:.3f} → {float(pdist(pos_raw).min()):.3f} Å", flush=True)
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {rid}: {exc}")

    print(f"\ndone: fixed={n_replaced}  already_ok={n_skip_ok}  errors={n_err}")


if __name__ == "__main__":
    main()
