"""Fragment-assignment helpers for ADF EDA-NOCV input preparation.

For dipolar cycloaddition reactions, fragments come for free from the
`ReactionRecord.fragments` dict (r0 atoms = A, r1 atoms = B). For QMrxn20
E2 / SN2 reactions, the TS file is a single XYZ with the substrate (R-X) and
incoming nucleophile/base (Y) jumbled together; this module produces the
two atom-index sets.

Strategy for QMrxn20:
    1. Determine connectivity on the reactant-complex XYZ (rcc/rcu) using
       RDKit DetermineConnectivity with charge=-1.  At the bound-complex
       stage Y is approached but rarely directly bonded to R, so the mol
       typically splits into 2 connected components.
    2. The smaller component is fragment B (Y nucleophile / base, usually
       1-3 atoms). The larger is fragment A (substrate).
    3. Apply the *same* atom-index split to the TS atom array. The two
       XYZ files share atom ordering because they come from the same
       computation (just different geometries).
    4. If step 1 fails to yield two components, fall back to a heuristic
       cut at the longest "bond" reported by DetermineConnectivity.

For the relaxed-fragment references:
    - Fragment A (substrate) relaxed: load
      `reactant-conformers/{substrate_label}/00.xyz` where substrate_label
      is the first 5 components of the TS label (positions 1-4 substituents
      + position 5 leaving group X; no Y).
    - Fragment B (Y) relaxed: Y is usually a single atom (Cl-, F-, etc.) or
      a small species. Its relaxed geometry is treated as the same as in
      the TS-frozen state for monatomic ions, since there's no internal
      geometry. For multi-atomic Y we use the TS-frozen positions as a
      conservative approximation (∆E_strain_B ≈ 0). The runner records
      this as a provenance flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ase.io
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity


@dataclass
class QMrxn20Fragmentation:
    fragA_indices: np.ndarray              # substrate
    fragB_indices: np.ndarray              # nucleophile/base
    method: str                            # how we obtained the split
    substrate_label: str                   # for sourcing the relaxed-A reference
    fragA_relaxed_xyz: Path | None         # path to substrate-only XYZ (if found)


def _ts_label_to_substrate_label(ts_label: str) -> str:
    parts = ts_label.split("_")
    if len(parts) != 6:
        raise ValueError(f"unexpected QMrxn20 TS label: {ts_label}")
    return "_".join(parts[:5])


def _mol_from_xyz_with_connectivity(xyz_path: Path, charge: int) -> Chem.Mol | None:
    raw = Chem.MolFromXYZFile(str(xyz_path))
    if raw is None:
        return None
    mol = Chem.Mol(raw)
    try:
        DetermineConnectivity(mol, charge=charge)
    except (ValueError, RuntimeError):
        return None
    return mol


def fragment_qmrxn20(
    qm_root: Path,
    reaction: str,
    ts_label: str,
    rcc_xyz: Path | None = None,
) -> QMrxn20Fragmentation:
    """Determine fragmentA / fragmentB indices for a QMrxn20 (reaction, ts_label).

    Parameters
    ----------
    qm_root : Path
        The QMrxn20 dataset root (contains transition-states/, reactant-complex-*/, etc.)
    reaction : str  ("e2" | "sn2")
    ts_label : str  ("A_B_C_D_E_F")
    rcc_xyz : Path | None
        Path to the reactant-complex XYZ to derive connectivity from. If None,
        we try the default rcc/{label}/00.xyz then rcu fallback.

    Returns
    -------
    QMrxn20Fragmentation
    """
    qm_root = Path(qm_root)
    if rcc_xyz is None:
        for kind in ("constrained", "unconstrained"):
            cand = qm_root / f"reactant-complex-{kind}-conformers" / reaction / ts_label / "00.xyz"
            if cand.is_file():
                rcc_xyz = cand
                break
    if rcc_xyz is None or not rcc_xyz.is_file():
        raise FileNotFoundError(
            f"no reactant-complex XYZ found for {reaction}/{ts_label}"
        )

    mol = _mol_from_xyz_with_connectivity(rcc_xyz, charge=-1)
    method = "rdkit_connectivity_charge-1"
    if mol is None:
        mol = _mol_from_xyz_with_connectivity(rcc_xyz, charge=0)
        method = "rdkit_connectivity_charge0_fallback"
    if mol is None:
        raise ValueError(f"DetermineConnectivity failed on {rcc_xyz}")

    fragments = Chem.GetMolFrags(mol, asMols=False)
    if len(fragments) >= 2:
        # Use the two largest by atom count; sort smallest fragment first.
        sized = sorted(fragments, key=len)
        fragB = sized[0]
        fragA = sized[-1]
        # If there are residual very small fragments, lump them with the smallest
        # so we end up with exactly two groups.
        if len(fragments) > 2:
            others = [a for grp in sized[1:-1] for a in grp]
            fragB = tuple(list(fragB) + others)
            method += "_collapsed_extras"
    else:
        # Single component: cut at the longest bond using geometric heuristic.
        positions = np.asarray([mol.GetConformer().GetAtomPosition(i)
                                for i in range(mol.GetNumAtoms())])
        positions = np.stack([np.array([p.x, p.y, p.z]) for p in positions])
        bonds = list(mol.GetBonds())
        if not bonds:
            raise ValueError(f"no bonds found in {rcc_xyz}; can't split")
        lengths = [
            (b, float(np.linalg.norm(positions[b.GetBeginAtomIdx()]
                                     - positions[b.GetEndAtomIdx()])))
            for b in bonds
        ]
        longest = max(lengths, key=lambda x: x[1])[0]
        # Remove the longest bond and recompute fragments
        editable = Chem.EditableMol(mol)
        editable.RemoveBond(longest.GetBeginAtomIdx(), longest.GetEndAtomIdx())
        cut_mol = editable.GetMol()
        frags = Chem.GetMolFrags(cut_mol, asMols=False)
        if len(frags) < 2:
            raise ValueError(f"longest-bond cut failed for {rcc_xyz}")
        sized = sorted(frags, key=len)
        fragB = sized[0]
        fragA = sized[-1]
        method = "longest_bond_cut"

    fragA_idx = np.array(sorted(fragA), dtype=np.int64)
    fragB_idx = np.array(sorted(fragB), dtype=np.int64)

    substrate_label = _ts_label_to_substrate_label(ts_label)
    relaxed_xyz_candidate = qm_root / "reactant-conformers" / substrate_label / "00.xyz"
    relaxed_xyz = relaxed_xyz_candidate if relaxed_xyz_candidate.is_file() else None

    return QMrxn20Fragmentation(
        fragA_indices=fragA_idx,
        fragB_indices=fragB_idx,
        method=method,
        substrate_label=substrate_label,
        fragA_relaxed_xyz=relaxed_xyz,
    )


def read_xyz_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    atoms = ase.io.read(str(path), format="xyz")
    return (
        np.asarray(atoms.numbers, dtype=np.int64),
        np.asarray(atoms.positions, dtype=np.float64),
    )


def align_substrate_to_TS(
    substrate_xyz: Path,
    ts_positions_subset: np.ndarray,
    ts_numbers_subset: np.ndarray,
) -> np.ndarray:
    """Reorder substrate atoms to match the TS atom ordering of its substrate subset.

    Both the substrate-only relaxed XYZ and the substrate subset of the TS
    have the SAME composition. We need positions in the same atom order.
    A simple Hungarian-style match on element + nearest distance works.
    Returns the (n, 3) array of substrate-relaxed positions in TS-substrate order.
    """
    sub_numbers, sub_positions = read_xyz_positions(substrate_xyz)
    if len(sub_numbers) != len(ts_numbers_subset):
        raise ValueError(
            f"substrate atom count {len(sub_numbers)} != "
            f"TS subset atom count {len(ts_numbers_subset)}"
        )
    n = len(sub_numbers)
    used = [False] * n
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        target_z = int(ts_numbers_subset[i])
        target_pos = ts_positions_subset[i]
        # find the closest substrate atom with matching Z that hasn't been used
        best_j, best_d = -1, np.inf
        for j in range(n):
            if used[j] or int(sub_numbers[j]) != target_z:
                continue
            d = float(np.linalg.norm(sub_positions[j] - target_pos))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j < 0:
            raise ValueError(
                f"no substrate atom matched TS atom {i} (Z={target_z})"
            )
        out[i] = sub_positions[best_j]
        used[best_j] = True
    return out
