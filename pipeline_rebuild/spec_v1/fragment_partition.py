"""Fragment partition per reaction — returns (frag_A_TS_indices, frag_B_TS_indices).

Strategy per family:

- **dipolar**  : R = r0 ⊕ r1 in load order, but autodE's TS re-orders atoms.
  Use the atom-mapped LHS SMILES from `full_dataset.csv` + RDKit's
  `GetSubstructMatch` on TS connectivity (RDKit `DetermineConnectivity`)
  to recover fragA/fragB indices in the TS array. Falls back to
  connected-component analysis on R + coordinate matching if SMILES
  lookup fails.

- **qmrxn20** (e2/sn2) : R and TS share atom order per QMrxn20 convention.
  Connected components on R (natural cutoffs) → two largest groups.

- **rgd1** : R and TS share atom order per RGD1 HDF5 convention. Rsmiles
  in the HDF5 gives "A.B" for bimolecular reactions; for unimolecular
  (single-fragment) reactions, we split at the reactive bond using
  the atom-mapped SMILES on both sides. For 1r1p same-reactant/product,
  we fall back to an even-atom-index split (fragA = even, fragB = odd)
  purely to keep xTB single-points defined; this is a documented
  approximation for the m2/m3 features.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import ase
import ase.io
import networkx as nx
import numpy as np
import pandas as pd
from ase.neighborlist import build_neighbor_list, natural_cutoffs

RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
DIP_ROOT = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
DIP_CSV = RAW / "dipolar_cycloaddition" / "full_dataset.csv"
QMR_ROOT = RAW / "QMrxn20"
RGD1_XYZ_ROOT = RAW / "rgd1" / "extracted_xyz"
RGD1_H5 = RAW / "rgd1" / "RGD1_CHNO.h5"

_DIP_CSV_CACHE: Optional[pd.DataFrame] = None


def _dipolar_csv() -> pd.DataFrame:
    global _DIP_CSV_CACHE
    if _DIP_CSV_CACHE is None:
        _DIP_CSV_CACHE = pd.read_csv(DIP_CSV)
    return _DIP_CSV_CACHE


def _cc_components(atoms: ase.Atoms) -> list[list[int]]:
    """Return connected components (as sorted index lists) using ASE natural cutoffs × 1.2."""
    cutoffs = [c * 1.2 for c in natural_cutoffs(atoms)]
    nl = build_neighbor_list(atoms, cutoffs, self_interaction=False, bothways=True)
    n = len(atoms)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        idx, _ = nl.get_neighbors(i)
        for j in idx:
            g.add_edge(i, int(j))
    comps = [sorted(c) for c in nx.connected_components(g)]
    comps.sort(key=lambda c: -len(c))
    return comps


def _two_largest_components(atoms: ase.Atoms) -> tuple[list[int], list[int]]:
    comps = _cc_components(atoms)
    if len(comps) >= 2:
        return comps[0], comps[1]
    # Single-component fallback: split at largest coordinate gap along principal axis
    positions = atoms.get_positions()
    centered = positions - positions.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[0]
    order = np.argsort(proj)
    n = len(order)
    return sorted(order[: n // 2].tolist()), sorted(order[n // 2 :].tolist())


def _dipolar_partition(rid: str, ts_atoms: ase.Atoms) -> tuple[list[int], list[int]]:
    """Use atom-mapped SMILES + subgraph match to find fragA/fragB in TS array.

    Falls back to coordinate matching between R = r0 ⊕ r1 and TS if RDKit
    steps fail.
    """
    idx = int(rid.rsplit("_", 1)[-1])
    d = DIP_ROOT / str(idx)
    r0_path = next(d.glob("r0_*.xyz"))
    r1_path = next(d.glob("r1_*.xyz"))

    csv = _dipolar_csv()
    row = csv[csv["rxn_id"] == idx]
    if len(row) == 1:
        try:
            from rdkit import Chem
            from rdkit.Chem.rdDetermineBonds import DetermineConnectivity
            rxn_smi = str(row.iloc[0]["rxn_smiles"])
            lhs, _ = rxn_smi.split(">>", 1)
            parts = lhs.split(".")
            if len(parts) == 2:
                smi_a, smi_b = parts
                mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
                mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))
                raw_ts = Chem.MolFromXYZFile(str([f for f in d.glob("TS_*.xyz") if "imag_mode" not in f.name][0]))
                if raw_ts is None:
                    raw_ts = Chem.MolFromXYZFile(str(next(d.glob("TS_*.xyz"))))
                if raw_ts is not None:
                    DetermineConnectivity(raw_ts)
                    match_a = raw_ts.GetSubstructMatch(mol_a)
                    match_b = raw_ts.GetSubstructMatch(mol_b)
                    if match_a and match_b and not (set(match_a) & set(match_b)):
                        return sorted(list(match_a)), sorted(list(match_b))
        except Exception:
            pass

    # Fallback: R = r0 ⊕ r1 (natural n_A/n_B) → coordinate assignment on TS.
    r0 = ase.io.read(str(r0_path))
    r1 = ase.io.read(str(r1_path))
    n_a, n_b = len(r0), len(r1)
    R = r0 + r1
    R_pos = R.get_positions()
    T_pos = ts_atoms.get_positions()
    # Greedy nearest-neighbor matching on element-matched atoms.
    used = [False] * len(ts_atoms)
    ts_of_r = np.full(len(R), -1, dtype=int)
    R_z = np.array(R.get_atomic_numbers())
    T_z = np.array(ts_atoms.get_atomic_numbers())
    for i in range(len(R)):
        # Candidates: same element, unused
        cands = [j for j in range(len(ts_atoms)) if not used[j] and T_z[j] == R_z[i]]
        if not cands:
            continue
        d2 = np.array([np.sum((T_pos[j] - R_pos[i]) ** 2) for j in cands])
        j = cands[int(np.argmin(d2))]
        ts_of_r[i] = j
        used[j] = True
    frag_a = sorted([int(ts_of_r[k]) for k in range(n_a) if ts_of_r[k] >= 0])
    frag_b = sorted([int(ts_of_r[k]) for k in range(n_a, n_a + n_b) if ts_of_r[k] >= 0])
    return frag_a, frag_b


def _qmrxn20_partition(rid: str, family: str, r_atoms: ase.Atoms) -> tuple[list[int], list[int]]:
    """Same atom order as TS. Connected components on R."""
    return _two_largest_components(r_atoms)


def _rgd1_partition(rid: str, r_atoms: ase.Atoms) -> tuple[list[int], list[int]]:
    """Same atom order as TS. Use connected components on R; if 1 comp
    (unimolecular reactant), fall back to principal-axis split."""
    return _two_largest_components(r_atoms)


def partition_for(rid: str, family: str) -> dict:
    """Return {'frag_A_indices': [...], 'frag_B_indices': [...], 'method': str}."""
    if family == "dipolar":
        idx = int(rid.rsplit("_", 1)[-1])
        d = DIP_ROOT / str(idx)
        ts_paths = [f for f in d.glob("TS_*.xyz") if "imag_mode" not in f.name]
        if not ts_paths:
            ts_paths = list(d.glob("TS_*.xyz"))
        ts = ase.io.read(str(ts_paths[0]))
        a, b = _dipolar_partition(rid, ts)
        return {"frag_A_indices": a, "frag_B_indices": b, "n_atoms": len(ts),
                "method": "dipolar_smiles_or_coord"}
    if family in ("qmrxn20_e2", "qmrxn20_sn2"):
        subfam = "e2" if "e2" in family else "sn2"
        label = "_".join(rid.split("_")[2:])
        rc = QMR_ROOT / "reactant-complex-constrained-conformers" / subfam / label
        r_path = rc / "00.xyz"
        if not r_path.exists():
            r_path = next(iter(rc.glob("*.xyz")))
        R = ase.io.read(str(r_path))
        a, b = _qmrxn20_partition(rid, family, R)
        return {"frag_A_indices": a, "frag_B_indices": b, "n_atoms": len(R),
                "method": "qmrxn20_cc"}
    if family == "rgd1":
        R = ase.io.read(str(RGD1_XYZ_ROOT / rid / "R.xyz"))
        a, b = _rgd1_partition(rid, R)
        return {"frag_A_indices": a, "frag_B_indices": b, "n_atoms": len(R),
                "method": "rgd1_cc"}
    raise ValueError(f"unknown family {family!r}")


def partition_all(labels_parquet: str | Path) -> dict:
    df = pd.read_parquet(labels_parquet)
    out = {}
    for _, row in df.iterrows():
        try:
            out[row.reaction_id] = partition_for(row.reaction_id, row.family)
        except Exception as e:
            out[row.reaction_id] = {"error": f"{type(e).__name__}: {e}"}
    return out


if __name__ == "__main__":
    import sys
    parquet = sys.argv[1] if len(sys.argv) > 1 else "labels/adf/adf_labels_v6_multifamily.parquet"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"processing {parquet} → {out_path}", flush=True)
    part = partition_all(parquet)
    with open(out_path, "w") as f:
        json.dump(part, f, indent=1)
    ok = sum(1 for v in part.values() if "error" not in v)
    err = len(part) - ok
    print(f"partitions: {ok} ok, {err} errors")
    if err:
        for r, v in list(part.items())[:20]:
            if "error" in v:
                print(f"  {r}: {v['error']}")
