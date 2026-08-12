#!/usr/bin/env python3
"""Phase 4 preparation: extract MACE-OFF23_medium features on TS geometry.

Reads cohort_subset.parquet + kisti_results eda.inp (TS with fragment labels).
Writes per-rxn tensor to mace_ts_v1/{rxn_id}.pt

Env: SEED_TASK_ID (0-2) for 3-way parallelization
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
NTASK = 3

COHORT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/cohort_subset.parquet")
KISTI = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")
OUT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/mace_ts_v1")
MACE_MODEL = Path.home() / ".cache/mace/MACE-OFF23_medium.model"

ATOM_RE = re.compile(r"^\s*([A-Za-z]+)\(([12])\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_ts_xyz(eda_inp_path):
    atoms = []
    for ln in eda_inp_path.read_text().splitlines():
        m = ATOM_RE.match(ln)
        if m:
            elem = m.group(1)
            x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
            atoms.append((elem, x, y, z))
    return atoms


def build_ase_atoms(atoms_list):
    from ase import Atoms
    elems = [a[0] for a in atoms_list]
    coords = np.array([[a[1], a[2], a[3]] for a in atoms_list])
    return Atoms(elems, positions=coords)


def extract_mace_features(atoms, calc):
    """Return per-atom MACE features tensor (N_atoms, 256)."""
    atoms.calc = calc
    # trigger single-point calculation
    atoms.get_potential_energy()
    # get node features (invariant part)
    # MACE stores in calc.results['node_feats'] or similar; use hook
    feats = calc.results.get("node_feats")
    if feats is None:
        # fallback: manually descriptor call
        from mace.calculators import MACECalculator
        descriptors = calc.get_descriptors(atoms, invariants_only=True)
        return torch.from_numpy(descriptors)
    return torch.from_numpy(np.asarray(feats))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(COHORT)
    # round-robin distribution over 3 tasks
    my_rxns = cohort[cohort["reaction_number"] % NTASK == TASK_ID]["reaction_number"].tolist()
    print(f"Task {TASK_ID}: handling {len(my_rxns)} rxns")

    # Load MACE
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=str(MACE_MODEL), device="cuda",
                          default_dtype="float32")

    n_done = 0
    for rxn in my_rxns:
        out_path = OUT / f"{rxn:04d}.pt"
        if out_path.exists():
            continue
        try:
            eda_inp = KISTI / f"rxn_{rxn:04d}" / "eda.inp"
            atoms_list = parse_ts_xyz(eda_inp)
            atoms = build_ase_atoms(atoms_list)
            # invariant descriptors
            descriptors = calc.get_descriptors(atoms, invariants_only=True)
            # shape: (n_atoms, feature_dim)
            torch.save({
                "rxn_id": rxn,
                "elements": [a[0] for a in atoms_list],
                "coords":   torch.tensor([[a[1], a[2], a[3]] for a in atoms_list], dtype=torch.float32),
                "mace_feats": torch.from_numpy(descriptors).float(),
            }, out_path)
            n_done += 1
            if n_done % 50 == 0:
                print(f"  [{n_done}/{len(my_rxns)}] latest: rxn_{rxn:04d} shape={descriptors.shape}")
        except Exception as e:
            print(f"  FAIL rxn_{rxn:04d}: {e}")

    print(f"Task {TASK_ID} done: {n_done} rxns processed")


if __name__ == "__main__":
    main()
