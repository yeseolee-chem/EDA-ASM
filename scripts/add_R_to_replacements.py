"""Extend the 143 replacement .pt files with R (reactant) geometry from raw sources.

Copies stage2_mace_features.load_triple's per-family logic (without running
MACE embeddings — only z + pos). Skips reactions already having R.

Emits atom-count sanity warnings when R != TS length.
"""
from __future__ import annotations
import json
from pathlib import Path

import ase
import ase.io
import h5py
import numpy as np
import torch

RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
DIP_ROOT = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
QMR_ROOT = RAW / "QMrxn20"
RGD1_H5 = RAW / "rgd1" / "RGD1_CHNO.h5"

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
REPLACE_JSON = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/frag_review/replacements_need_features.json")


def _single_match(rxn_dir: Path, pattern: str):
    matches = list(rxn_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    primary = [p for p in matches if "_alt" not in p.stem]
    return primary[0] if len(primary) == 1 else matches[0]


def _compose_dipolar_R(r0_atoms, r1_atoms, gap=5.0):
    p0 = r0_atoms.get_positions()
    p1 = r1_atoms.get_positions()
    p1 = p1 - p1.mean(axis=0)
    p0_ctr = p0 - p0.mean(axis=0)
    r0_r = np.linalg.norm(p0_ctr, axis=1).max()
    r1_r = np.linalg.norm(p1, axis=1).max()
    p1 = p1 + p0.mean(axis=0) + np.array([r0_r + r1_r + gap, 0.0, 0.0])
    z = np.concatenate([np.asarray(r0_atoms.get_atomic_numbers(), int),
                        np.asarray(r1_atoms.get_atomic_numbers(), int)])
    pos = np.vstack([p0, p1])
    return ase.Atoms(numbers=z, positions=pos)


def load_R(rid: str, family: str):
    if family == "dipolar":
        idx = int(rid.split("_")[-1])
        d = DIP_ROOT / str(idx)
        r0 = ase.io.read(str(_single_match(d, "r0_*.xyz")))
        r1_path = _single_match(d, "r1_*.xyz")
        if r1_path is None:
            return r0
        r1 = ase.io.read(str(r1_path))
        return _compose_dipolar_R(r0, r1)
    if family in ("qmrxn20_e2", "qmrxn20_sn2"):
        subfam = "e2" if "e2" in family else "sn2"
        label = "_".join(rid.split("_")[2:])
        rc = QMR_ROOT / "reactant-complex-constrained-conformers" / subfam / label
        r_path = rc / "00.xyz"
        if not r_path.exists():
            r_path = next(iter(rc.glob("*.xyz")), None)
        return ase.io.read(str(r_path))
    if family == "rgd1":
        key = "_".join(rid.split("_")[1:])
        with h5py.File(RGD1_H5, "r") as f:
            g = f[key]
            z = np.asarray(g["elements"], int)
            pos = np.asarray(g["RG"], float)
        return ase.Atoms(numbers=z, positions=pos)
    raise ValueError(family)


def main():
    with open(REPLACE_JSON) as f:
        replacements = json.load(f)

    n_ok = n_skip = n_err = n_mismatch = 0
    for entry in replacements:
        rid = entry["reaction_id"]
        fam = entry["family"]
        pt_path = FEAT_DIR / f"{rid}.pt"
        try:
            d = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except FileNotFoundError:
            n_err += 1
            print(f"[ERR] {rid}: no .pt file", flush=True)
            continue
        if "R" in d:
            n_skip += 1
            continue
        try:
            R = load_R(rid, fam)
            nR = len(R)
            nTS = len(d["TS"]["z"])
            if nR != nTS:
                n_mismatch += 1
                print(f"[MISMATCH] {rid}: R has {nR} atoms, TS has {nTS} — skipping R attach", flush=True)
                continue
            d["R"] = {
                "z": np.asarray(R.get_atomic_numbers(), int),
                "pos": R.get_positions().astype(float),
            }
            torch.save(d, str(pt_path))
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {rid} ({fam}): {exc}", flush=True)

    print(f"done: ok={n_ok}  skipped={n_skip}  errors={n_err}  R!=TS_len={n_mismatch}")


if __name__ == "__main__":
    main()
