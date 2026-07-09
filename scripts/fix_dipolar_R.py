"""Recompute R geometry for dipolar reactions with r0 + r1 properly offset.

Original stage2_mace_features.py did `r0 + r1` by direct concat — each xyz was
optimised in its own reference frame near origin, so the two reactant molecules
overlap in the stored R geometry. This corrupts the visualisation (atoms
interpenetrate) and any downstream distance-based analysis on R.

This script:
  - Iterates over every dipolar reaction in cohort_v7.parquet
  - Loads raw r0.xyz + r1.xyz separately
  - Centres r1 at origin, then translates it by (r0_bounding_radius +
    r1_bounding_radius + 5 Å) along +x → guarantees a ≥5 Å intermolecular gap
  - Overwrites the R.pos field in the reaction's .pt file (feat is DROPPED
    since old features were computed on the overlapped geometry — flag it so
    downstream code can skip if needed).
"""
from __future__ import annotations
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
DIP_ROOT = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
COHORT_V7 = REPO / "outputs/frag_review/cohort_v7.parquet"

GAP_A = 5.0


def _single_match(rxn_dir: Path, pattern: str):
    matches = list(rxn_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    primary = [p for p in matches if "_alt" not in p.stem]
    return primary[0] if len(primary) == 1 else matches[0]


def compose_R(rid: str) -> tuple[np.ndarray, np.ndarray] | None:
    idx = int(rid.split("_")[-1])
    d = DIP_ROOT / str(idx)
    r0p = _single_match(d, "r0_*.xyz")
    r1p = _single_match(d, "r1_*.xyz")
    if r0p is None:
        return None
    r0 = ase.io.read(str(r0p))
    if r1p is None:
        return (np.asarray(r0.get_atomic_numbers(), int), r0.get_positions())
    r1 = ase.io.read(str(r1p))

    p0 = r0.get_positions()
    p1 = r1.get_positions()
    # Centre r1 at origin, then shift so it clears r0 by GAP_A.
    p1 = p1 - p1.mean(axis=0)
    p0_ctr = p0 - p0.mean(axis=0)
    r0_radius = np.linalg.norm(p0_ctr, axis=1).max()
    r1_radius = np.linalg.norm(p1, axis=1).max()
    shift = np.array([r0_radius + r1_radius + GAP_A, 0.0, 0.0])
    p1 = p1 + p0.mean(axis=0) + shift

    z_full = np.concatenate([np.asarray(r0.get_atomic_numbers(), int),
                              np.asarray(r1.get_atomic_numbers(), int)])
    pos_full = np.vstack([p0, p1])
    return z_full, pos_full


def main():
    c7 = pd.read_parquet(COHORT_V7)
    dip = c7[c7.family == "dipolar"].reaction_id.tolist()
    print(f"dipolar cohort_v7 count: {len(dip)}", flush=True)

    n_ok = n_skip = n_err = 0
    for rid in dip:
        pt = FEAT_DIR / f"{rid}.pt"
        if not pt.exists():
            n_err += 1
            print(f"[ERR] {rid}: no .pt", flush=True)
            continue
        composed = compose_R(rid)
        if composed is None:
            n_err += 1
            print(f"[ERR] {rid}: compose_R failed", flush=True)
            continue
        z, pos = composed

        d = torch.load(str(pt), map_location="cpu", weights_only=False)
        if "R" in d:
            n_old = len(d["R"]["z"])
            if n_old != len(z):
                n_err += 1
                print(f"[ERR] {rid}: atom count mismatch old={n_old} new={len(z)}", flush=True)
                continue
        # Preserve elements order; only overwrite positions (feat becomes stale).
        d["R"] = {"z": z, "pos": pos.astype(np.float32), "feat_stale": True}
        # Drop stale feat if present.
        torch.save(d, str(pt))
        n_ok += 1

    print(f"done: ok={n_ok}  skipped={n_skip}  errors={n_err}")


if __name__ == "__main__":
    main()
